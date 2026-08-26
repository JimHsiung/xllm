/* Copyright 2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/xLLM-AI/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "runtime/prepared_task/prepared_input_arena.h"

#include <glog/logging.h>

#include <algorithm>
#include <array>
#include <limits>
#include <vector>

#include "common/metrics.h"

namespace xllm {
namespace {

torch::Tensor wrap_host_int_vector(const std::vector<int32_t>& values) {
  if (values.empty()) {
    return torch::Tensor();
  }
  return torch::from_blob(
      const_cast<int32_t*>(values.data()),
      {static_cast<int64_t>(values.size())},
      torch::TensorOptions().dtype(torch::kInt).device(torch::kCPU));
}

torch::Tensor make_host_int64_vector(const std::vector<int64_t>& values,
                                     torch::ScalarType dtype) {
  if (values.empty()) {
    return torch::Tensor();
  }
  if (dtype == torch::kInt64) {
    return torch::from_blob(
        const_cast<int64_t*>(values.data()),
        {static_cast<int64_t>(values.size())},
        torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
  }
  return torch::tensor(values,
                       torch::TensorOptions().dtype(dtype).device(torch::kCPU));
}

torch::Tensor prefer_host_int_vector(const std::vector<int32_t>& host_values,
                                     const torch::Tensor& device_source) {
  if (!host_values.empty()) {
    return wrap_host_int_vector(host_values);
  }
  return device_source;
}

torch::Tensor prefer_host_tensor(const torch::Tensor& host_source,
                                 const torch::Tensor& device_source) {
  if (host_source.defined() && host_source.device().is_cpu()) {
    return host_source;
  }
  return device_source;
}

bool tensor_points_into_buffer(const torch::Tensor& tensor,
                               const torch::Tensor& buffer) {
  if (!tensor.defined() || tensor.numel() == 0 || !buffer.defined() ||
      buffer.numel() == 0 || tensor.device() != buffer.device()) {
    return false;
  }
  const uintptr_t buffer_begin = reinterpret_cast<uintptr_t>(buffer.data_ptr());
  const uintptr_t buffer_end =
      buffer_begin +
      static_cast<uintptr_t>(buffer.numel()) * buffer.element_size();
  const uintptr_t tensor_begin = reinterpret_cast<uintptr_t>(tensor.data_ptr());
  const uintptr_t tensor_end =
      tensor_begin +
      static_cast<uintptr_t>(tensor.numel()) * tensor.element_size();
  return tensor_begin >= buffer_begin && tensor_end <= buffer_end;
}

bool add_generated_tensor_to_plan(const torch::Tensor& source,
                                  torch::Tensor* target,
                                  const torch::Tensor& arena_buffer,
                                  detail::ForwardInputBufferPlan& plan) {
  CHECK(target != nullptr);
  if (target->defined() && target->device() == arena_buffer.device() &&
      (target->numel() == 0 ||
       tensor_points_into_buffer(*target, arena_buffer))) {
    return true;
  }
  return plan.add(source, target);
}

void release_rebound_attention_buffers(AttentionInput& attention) {
  attention.attention_host_buffer = torch::Tensor();
  const bool owns_external_history =
      tensor_points_into_buffer(attention.device.history_compressed_kv,
                                attention.attention_device_buffer) ||
      tensor_points_into_buffer(attention.device.history_k_rope,
                                attention.attention_device_buffer);
  if (owns_external_history) {
    return;
  }
  attention.attention_device_buffer = torch::Tensor();
  attention.attention_buffer_bytes = 0;
  attention.attention_buffer_capacity = 0;
  attention.attention_buffer_owner = std::make_shared<int>(0);
}

bool prepared_external_sources_supported(const ForwardInput& input) {
  const ModelInputParams& params = input.input_params;
  if (params.parallel.cp_plan.enabled() || params.mtp_topk_state != nullptr) {
    return false;
  }
  if (params.is_spec_verify &&
      (!params.graph.spec_verify_draft_token_sources.empty() ||
       params.graph.spec_verify_static_graph_tasks_prepared)) {
    return false;
  }
  if (params.graph.input_tokens_override.defined() &&
      (!input.token_ids.defined() ||
       !params.graph.input_tokens_override.is_same(input.token_ids))) {
    return false;
  }
  return params.graph.spec_verify_draft_token_sources.empty() ||
         params.graph.spec_verify_source_addresses_stable;
}

bool add_prepared_host_model_tensors_to_plan(
    const ModelInputParams& source,
    ModelInputParams& target,
    detail::ForwardInputBufferPlan& plan) {
  target.multi_block_tables.clear();
  target.multi_block_tables.resize(source.multi_block_tables.size());
  for (size_t manager_index = 0;
       manager_index < source.multi_block_tables.size();
       ++manager_index) {
    if (!plan.add(source.multi_block_tables[manager_index],
                  &target.multi_block_tables[manager_index])) {
      return false;
    }
  }
  return true;
}

uint64_t combine_layout_signatures(uint64_t current_signature,
                                   uint64_t appended_signature,
                                   uint64_t appended_offset) {
  constexpr uint64_t kHashSeed = 0x9e3779b97f4a7c15ULL;
  uint64_t signature = current_signature;
  signature ^=
      appended_signature + kHashSeed + (signature << 6) + (signature >> 2);
  signature ^=
      appended_offset + kHashSeed + (signature << 6) + (signature >> 2);
  return signature;
}

void record_transfer_stats(const detail::ForwardInputBufferPlan& plan,
                           const detail::ForwardInputHostCopyStats& h2d_stats,
                           bool accumulate,
                           ForwardInput& staged_input) {
  const uint64_t d2d_bytes = plan.device_source_bytes();
  const int32_t d2d_copies = plan.device_source_count();
  if (accumulate) {
    staged_input.prepared_arena_h2d_bytes += h2d_stats.bytes;
    staged_input.prepared_arena_h2d_copies += h2d_stats.copies;
    staged_input.prepared_arena_d2d_bytes += d2d_bytes;
    staged_input.prepared_arena_d2d_copies += d2d_copies;
  } else {
    staged_input.prepared_arena_h2d_bytes = h2d_stats.bytes;
    staged_input.prepared_arena_h2d_copies = h2d_stats.copies;
    staged_input.prepared_arena_d2d_bytes = d2d_bytes;
    staged_input.prepared_arena_d2d_copies = d2d_copies;
  }
  COUNTER_ADD(prepared_task_staging_h2d_bytes_total, h2d_stats.bytes);
  COUNTER_ADD(prepared_task_staging_h2d_copies_total, h2d_stats.copies);
  COUNTER_ADD(prepared_task_staging_d2d_bytes_total, d2d_bytes);
  COUNTER_ADD(prepared_task_staging_d2d_copies_total, d2d_copies);
}

bool add_prepared_parallel_to_plan(const ParallelInput& source,
                                   ParallelInput& target,
                                   const torch::Tensor& arena_buffer,
                                   detail::ForwardInputBufferPlan& plan) {
  const DpEpPaddingData& source_padding = source.dp_ep_padding_data;
  DpEpPaddingData& target_padding = target.dp_ep_padding_data;
  return add_generated_tensor_to_plan(source_padding.attn_padding_idx(),
                                      &target_padding.attn_padding_idx(),
                                      arena_buffer,
                                      plan) &&
         add_generated_tensor_to_plan(source_padding.attn_unpadding_idx(),
                                      &target_padding.attn_unpadding_idx(),
                                      arena_buffer,
                                      plan) &&
         add_generated_tensor_to_plan(source_padding.ffn_padding_idx(),
                                      &target_padding.ffn_padding_idx(),
                                      arena_buffer,
                                      plan) &&
         add_generated_tensor_to_plan(source_padding.ffn_unpadding_idx(),
                                      &target_padding.ffn_unpadding_idx(),
                                      arena_buffer,
                                      plan) &&
         add_generated_tensor_to_plan(
             source_padding.lm_head_skip_padding_token_indices(),
             &target_padding.lm_head_skip_padding_token_indices(),
             arena_buffer,
             plan) &&
         add_generated_tensor_to_plan(source_padding.gather_prenorm_idx(),
                                      &target_padding.gather_prenorm_idx(),
                                      arena_buffer,
                                      plan) &&
         add_generated_tensor_to_plan(source_padding.padding_idx(),
                                      &target_padding.padding_idx(),
                                      arena_buffer,
                                      plan) &&
         add_generated_tensor_to_plan(source_padding.un_padding_idx(),
                                      &target_padding.un_padding_idx(),
                                      arena_buffer,
                                      plan) &&
         add_generated_tensor_to_plan(source_padding.dynamic_ep_idx(),
                                      &target_padding.dynamic_ep_idx(),
                                      arena_buffer,
                                      plan) &&
         add_generated_tensor_to_plan(source_padding.moe_idx(),
                                      &target_padding.moe_idx(),
                                      arena_buffer,
                                      plan) &&
         add_generated_tensor_to_plan(source_padding.expert_array(),
                                      &target_padding.expert_array(),
                                      arena_buffer,
                                      plan) &&
         add_generated_tensor_to_plan(
             source_padding.post_lmhead_gather_indices(),
             &target_padding.post_lmhead_gather_indices(),
             arena_buffer,
             plan);
}

bool add_prepared_generated_model_tensors_to_plan(
    const ModelInputParams& source,
    ModelInputParams& target,
    const torch::Tensor& arena_buffer,
    detail::ForwardInputBufferPlan& plan) {
  const torch::Tensor linear_state_indices = prefer_host_int_vector(
      source.embedding.linear_state_ids, source.embedding.linear_state_indices);
  const torch::Tensor expanded_kv_seq_lens = prefer_host_int_vector(
      source.graph.expanded_kv_seq_lens_vec, source.graph.expanded_kv_seq_lens);
  torch::Tensor num_accepted_tokens = source.num_accepted_tokens;
  if (!source.num_accepted_tokens_host.empty()) {
    const torch::ScalarType accepted_token_dtype =
        source.num_accepted_tokens.defined()
            ? source.num_accepted_tokens.scalar_type()
            : torch::kLong;
    num_accepted_tokens = make_host_int64_vector(
        source.num_accepted_tokens_host, accepted_token_dtype);
  }
  return add_generated_tensor_to_plan(linear_state_indices,
                                      &target.embedding.linear_state_indices,
                                      arena_buffer,
                                      plan) &&
         add_generated_tensor_to_plan(source.embedding.mtp_shifted_token_ids,
                                      &target.embedding.mtp_shifted_token_ids,
                                      arena_buffer,
                                      plan) &&
         add_prepared_parallel_to_plan(
             source.parallel, target.parallel, arena_buffer, plan) &&
         add_generated_tensor_to_plan(source.expert.expert_array,
                                      &target.expert.expert_array,
                                      arena_buffer,
                                      plan) &&
         add_generated_tensor_to_plan(source.expert.eplb_decode_token_mask,
                                      &target.expert.eplb_decode_token_mask,
                                      arena_buffer,
                                      plan) &&
         add_generated_tensor_to_plan(source.graph.attn_mask,
                                      &target.graph.attn_mask,
                                      arena_buffer,
                                      plan) &&
         add_generated_tensor_to_plan(source.graph.tiling_data,
                                      &target.graph.tiling_data,
                                      arena_buffer,
                                      plan) &&
         add_generated_tensor_to_plan(expanded_kv_seq_lens,
                                      &target.graph.expanded_kv_seq_lens,
                                      arena_buffer,
                                      plan) &&
         add_generated_tensor_to_plan(source.graph.expanded_block_tables,
                                      &target.graph.expanded_block_tables,
                                      arena_buffer,
                                      plan) &&
         add_generated_tensor_to_plan(source.graph.expanded_paged_kv_indptr,
                                      &target.graph.expanded_paged_kv_indptr,
                                      arena_buffer,
                                      plan) &&
         add_generated_tensor_to_plan(source.graph.expanded_paged_kv_indices,
                                      &target.graph.expanded_paged_kv_indices,
                                      arena_buffer,
                                      plan) &&
         add_generated_tensor_to_plan(
             source.graph.expanded_paged_kv_last_page_len,
             &target.graph.expanded_paged_kv_last_page_len,
             arena_buffer,
             plan) &&
         add_generated_tensor_to_plan(source.graph.expanded_tiling_data,
                                      &target.graph.expanded_tiling_data,
                                      arena_buffer,
                                      plan) &&
         add_generated_tensor_to_plan(num_accepted_tokens,
                                      &target.num_accepted_tokens,
                                      arena_buffer,
                                      plan) &&
         add_generated_tensor_to_plan(source.mtp_shifted_token_ids,
                                      &target.mtp_shifted_token_ids,
                                      arena_buffer,
                                      plan);
}

bool add_prepared_model_tensors_to_plan(const ModelInputParams& source,
                                        ModelInputParams& target,
                                        detail::ForwardInputBufferPlan& plan) {
  // These two tensors may be persistent model outputs or embedding-cache
  // state without a Host mirror. Every other packed input remains Host-only
  // and must be deliberately classified here before it may introduce D2D.
  if (!plan.add_external_device_source(source.embedding.input_embedding,
                                       &target.embedding.input_embedding) ||
      !plan.add_external_device_source(
          source.embedding.mtp_bootstrap_embeddings,
          &target.embedding.mtp_bootstrap_embeddings)) {
    return false;
  }
  const torch::Tensor no_device_source;
  const torch::Tensor linear_state_indices = prefer_host_int_vector(
      source.embedding.linear_state_ids, source.embedding.linear_state_indices);
  const torch::Tensor expanded_kv_seq_lens = prefer_host_int_vector(
      source.graph.expanded_kv_seq_lens_vec, source.graph.expanded_kv_seq_lens);
  torch::Tensor num_accepted_tokens = source.num_accepted_tokens;
  if (!source.num_accepted_tokens_host.empty()) {
    const torch::ScalarType accepted_token_dtype =
        source.num_accepted_tokens.defined()
            ? source.num_accepted_tokens.scalar_type()
            : torch::kLong;
    num_accepted_tokens = make_host_int64_vector(
        source.num_accepted_tokens_host, accepted_token_dtype);
  }
  const detail::ModelTensorPlanSourceOverrides overrides{&no_device_source,
                                                         &linear_state_indices,
                                                         &no_device_source,
                                                         &expanded_kv_seq_lens,
                                                         &num_accepted_tokens};
  return detail::add_model_tensors_to_plan(source, target, plan, &overrides);
}

bool add_prepared_attention_to_plan(
    const AttentionInput& source,
    AttentionInput& target,
    detail::ForwardInputBufferPlan& plan,
    const torch::Tensor* arena_buffer = nullptr) {
  // Speculative builders already keep the scalar attention metadata on Host.
  // Prefer those mirrors so Prepared staging performs one H2D directly into
  // the fixed Slot Arena instead of first building a temporary Device buffer
  // and then copying it Device-to-Device into the Arena.
  auto add_tensor = [&](const torch::Tensor& tensor, torch::Tensor* target) {
    if (arena_buffer != nullptr) {
      return add_generated_tensor_to_plan(tensor, target, *arena_buffer, plan);
    }
    return plan.add(tensor, target);
  };
  return add_tensor(prefer_host_int_vector(source.host.q_seq_lens,
                                           source.device.q_seq_lens),
                    &target.device.q_seq_lens) &&
         add_tensor(prefer_host_int_vector(source.host.kv_seq_lens,
                                           source.device.kv_seq_lens),
                    &target.device.kv_seq_lens) &&
         add_tensor(prefer_host_int_vector(source.host.q_cu_seq_lens,
                                           source.device.q_cu_seq_lens),
                    &target.device.q_cu_seq_lens) &&
         add_tensor(prefer_host_int_vector(source.host.new_cache_slots,
                                           source.device.new_cache_slots),
                    &target.device.new_cache_slots) &&
         add_tensor(prefer_host_tensor(source.host.block_tables,
                                       source.device.block_tables),
                    &target.device.block_tables) &&
         add_tensor(source.device.paged_kv_indptr,
                    &target.device.paged_kv_indptr) &&
         add_tensor(source.device.paged_kv_indices,
                    &target.device.paged_kv_indices) &&
         add_tensor(source.device.paged_kv_last_page_len,
                    &target.device.paged_kv_last_page_len) &&
         add_tensor(source.device.new_cache_slot_offsets,
                    &target.device.new_cache_slot_offsets) &&
         add_tensor(source.device.kv_cache_start_offsets,
                    &target.device.kv_cache_start_offsets) &&
         add_tensor(prefer_host_int_vector(source.host.kv_cache_tokens_nums,
                                           source.device.kv_cache_tokens_nums),
                    &target.device.kv_cache_tokens_nums) &&
         add_tensor(prefer_host_int_vector(source.host.ring_cur_seqlen,
                                           source.device.ring_cur_seqlen),
                    &target.device.ring_cur_seqlen) &&
         add_tensor(prefer_host_int_vector(source.host.ring_cache_seqlen,
                                           source.device.ring_cache_seqlen),
                    &target.device.ring_cache_seqlen) &&
         add_tensor(source.device.in_prefix_slots,
                    &target.device.in_prefix_slots);
}

}  // namespace

bool prepared_spec_verify_graph_contract_is_complete(
    const ForwardInput& input) {
  const ModelInputParams& params = input.input_params;
  const GraphInput& graph = params.graph;
  const AttentionInput& attention = params.attention;
  if (!params.is_spec_verify ||
      !params.meta.batch_forward_type.is_chunked_prefill() ||
      !graph.prepared_spec_verify_direct_bind ||
      !graph.spec_verify_draft_token_sources.empty() ||
      graph.spec_verify_source_addresses_stable ||
      graph.spec_verify_static_graph_tasks_prepared ||
      graph.expanded_tiling_data.defined() ||
      !graph.input_tokens_override.defined() ||
      !graph.input_tokens_override.is_same(input.token_ids) ||
      !input.device_input_buffer.defined() ||
      input.prepared_input_layout_signature == 0 ||
      !params.multi_block_tables.empty() ||
      !params.device_multi_block_tables.empty()) {
    return false;
  }

  const int64_t num_sequences = params.meta.num_sequences;
  const int64_t spec_width = params.meta.q_max_seq_len;
  if (num_sequences <= 0 || spec_width <= 0) {
    return false;
  }
  const int64_t expected_tokens = num_sequences * spec_width;
  if (!input.token_ids.defined() || !input.positions.defined() ||
      input.token_ids.numel() != expected_tokens ||
      input.positions.numel() != expected_tokens ||
      !tensor_points_into_buffer(input.token_ids, input.device_input_buffer) ||
      !tensor_points_into_buffer(input.positions, input.device_input_buffer)) {
    return false;
  }

  if (attention.host.q_seq_lens.size() != static_cast<size_t>(num_sequences) ||
      attention.host.kv_seq_lens.size() != static_cast<size_t>(num_sequences)) {
    return false;
  }
  int64_t total_query_tokens = 0;
  int32_t max_query_tokens = 0;
  for (int32_t query_tokens : attention.host.q_seq_lens) {
    if (query_tokens <= 0) {
      return false;
    }
    total_query_tokens += query_tokens;
    max_query_tokens = std::max(max_query_tokens, query_tokens);
  }
  if (total_query_tokens != expected_tokens || max_query_tokens != spec_width) {
    return false;
  }
  if (!attention.device.q_seq_lens.defined() ||
      attention.device.q_seq_lens.numel() != num_sequences ||
      !attention.device.kv_seq_lens.defined() ||
      attention.device.kv_seq_lens.numel() != num_sequences ||
      !attention.device.q_cu_seq_lens.defined() ||
      attention.device.q_cu_seq_lens.numel() != num_sequences + 1 ||
      !attention.device.new_cache_slots.defined() ||
      attention.device.new_cache_slots.numel() != expected_tokens ||
      !attention.device.block_tables.defined() ||
      attention.device.block_tables.dim() != 2 ||
      attention.device.block_tables.size(/*dim=*/0) != num_sequences) {
    return false;
  }
  if (!params.embedding.linear_state_ids.empty() &&
      (!params.embedding.linear_state_indices.defined() ||
       params.embedding.linear_state_indices.numel() != num_sequences)) {
    return false;
  }
  if (!params.num_accepted_tokens_host.empty() &&
      (!params.num_accepted_tokens.defined() ||
       params.num_accepted_tokens.numel() != num_sequences)) {
    return false;
  }

  const bool uses_expanded_verify =
      graph.use_expanded_decode_for_spec_verify_attention;
  const bool has_any_expanded_tensor =
      graph.expanded_kv_seq_lens.defined() ||
      graph.expanded_block_tables.defined() ||
      graph.expanded_paged_kv_indptr.defined() ||
      graph.expanded_paged_kv_indices.defined() ||
      graph.expanded_paged_kv_last_page_len.defined() ||
      !graph.expanded_kv_seq_lens_vec.empty();
  if (uses_expanded_verify != has_any_expanded_tensor) {
    return false;
  }
  if (!uses_expanded_verify && graph.spec_verify_kv_seq_len_headroom != 0) {
    return false;
  }
  if (uses_expanded_verify) {
    if (!graph.expanded_kv_seq_lens.defined() ||
        graph.expanded_kv_seq_lens.numel() != expected_tokens ||
        !graph.expanded_block_tables.defined() ||
        graph.expanded_block_tables.dim() != 2 ||
        graph.expanded_block_tables.size(/*dim=*/0) != expected_tokens ||
        graph.expanded_block_tables.size(/*dim=*/1) !=
            attention.device.block_tables.size(/*dim=*/1) ||
        !graph.expanded_paged_kv_indptr.defined() ||
        graph.expanded_paged_kv_indptr.numel() != expected_tokens + 1 ||
        !graph.expanded_paged_kv_indices.defined() ||
        graph.expanded_paged_kv_indices.numel() <
            expected_tokens * graph.expanded_block_tables.size(/*dim=*/1) ||
        !graph.expanded_paged_kv_last_page_len.defined() ||
        graph.expanded_paged_kv_last_page_len.numel() != expected_tokens ||
        graph.expanded_kv_seq_lens_vec.size() !=
            static_cast<size_t>(expected_tokens) ||
        graph.spec_verify_kv_seq_len_headroom != spec_width - 1 ||
        params.parallel.query_start_loc.size() !=
            static_cast<size_t>(num_sequences + 1) ||
        params.parallel.query_start_loc.front() != 0 ||
        params.parallel.query_start_loc.back() != expected_tokens ||
        params.embedding.linear_state_ids.size() !=
            static_cast<size_t>(num_sequences) ||
        params.num_accepted_tokens_host.size() !=
            static_cast<size_t>(num_sequences)) {
      return false;
    }
    int64_t expanded_index = 0;
    for (int64_t sequence_index = 0; sequence_index < num_sequences;
         ++sequence_index) {
      const int32_t query_tokens =
          attention.host.q_seq_lens[static_cast<size_t>(sequence_index)];
      const int32_t final_kv_seq_len =
          attention.host.kv_seq_lens[static_cast<size_t>(sequence_index)];
      if (params.parallel
                  .query_start_loc[static_cast<size_t>(sequence_index)] !=
              expanded_index ||
          final_kv_seq_len < query_tokens) {
        return false;
      }
      for (int32_t token_index = 0; token_index < query_tokens; ++token_index) {
        const int32_t expected_kv_seq_len =
            final_kv_seq_len - query_tokens + token_index + 1;
        if (graph.expanded_kv_seq_lens_vec[static_cast<size_t>(
                expanded_index)] != expected_kv_seq_len) {
          return false;
        }
        ++expanded_index;
      }
    }
  }

  const std::array<const torch::Tensor*, 31> fixed_inputs = {
      &attention.device.q_seq_lens,
      &attention.device.kv_seq_lens,
      &attention.device.q_cu_seq_lens,
      &attention.device.new_cache_slots,
      &attention.device.block_tables,
      &attention.device.paged_kv_indptr,
      &attention.device.paged_kv_indices,
      &attention.device.paged_kv_last_page_len,
      &attention.device.new_cache_slot_offsets,
      &attention.device.kv_cache_start_offsets,
      &attention.device.kv_cache_tokens_nums,
      &attention.device.history_compressed_kv,
      &attention.device.history_k_rope,
      &attention.device.ring_cur_seqlen,
      &attention.device.ring_cache_seqlen,
      &attention.device.in_prefix_slots,
      &params.embedding.input_embedding,
      &params.embedding.linear_state_indices,
      &params.embedding.mtp_bootstrap_embeddings,
      &params.graph.attn_mask,
      &params.graph.tiling_data,
      &params.graph.expanded_kv_seq_lens,
      &params.graph.expanded_block_tables,
      &params.graph.expanded_paged_kv_indptr,
      &params.graph.expanded_paged_kv_indices,
      &params.graph.expanded_paged_kv_last_page_len,
      &params.graph.expanded_tiling_data,
      &params.num_accepted_tokens,
      &params.mtp_shifted_token_ids,
      &params.embedding.mtp_shifted_token_ids,
      &params.expert.eplb_decode_token_mask};
  return std::all_of(fixed_inputs.begin(),
                     fixed_inputs.end(),
                     [&input](const torch::Tensor* tensor) {
                       return !tensor->defined() || tensor->numel() == 0 ||
                              tensor_points_into_buffer(
                                  *tensor, input.device_input_buffer);
                     });
}

PreparedInputArena::PreparedInputArena(const torch::Device& device,
                                       uint64_t capacity_bytes)
    : device_(device), capacity_bytes_(capacity_bytes) {
  CHECK_GT(capacity_bytes_, 0);
  CHECK_LE(capacity_bytes_,
           static_cast<uint64_t>(std::numeric_limits<int64_t>::max()));
  const int64_t tensor_size = static_cast<int64_t>(capacity_bytes_);
  host_buffer_ = torch::empty({tensor_size},
                              torch::TensorOptions()
                                  .dtype(torch::kUInt8)
                                  .device(torch::kCPU)
                                  .pinned_memory(true));
  device_buffer_ =
      torch::empty({tensor_size},
                   torch::TensorOptions().dtype(torch::kUInt8).device(device_));
}

bool PreparedInputArena::stage(const ForwardInput& input,
                               ForwardInput& staged_input) {
  used_bytes_ = 0;
  partition_stride_bytes_ = 0;
  partition_count_ = 0;
  next_partition_ = 0;
  return stage_next(input, staged_input);
}

void PreparedInputArena::begin_partitioned_task(int32_t partition_count) {
  CHECK_GT(partition_count, 0);
  CHECK_LE(static_cast<uint64_t>(partition_count), capacity_bytes_);
  partition_count_ = partition_count;
  next_partition_ = 0;
  used_bytes_ = 0;
  partition_stride_bytes_ =
      (capacity_bytes_ / static_cast<uint64_t>(partition_count_)) /
      detail::kForwardInputBufferAlignment *
      detail::kForwardInputBufferAlignment;
  CHECK_GT(partition_stride_bytes_, 0);
}

bool PreparedInputArena::stage_next(const ForwardInput& input,
                                    ForwardInput& staged_input) {
  input.copy_metadata_to(staged_input);
  input.set_host_views(staged_input);
  staged_input.json_object_invalid_draft.clear();
  staged_input.json_object_errors.clear();
  staged_input.token_ids = torch::Tensor();
  staged_input.positions = torch::Tensor();
  staged_input.input_host_buffer = torch::Tensor();
  staged_input.device_input_buffer = torch::Tensor();

  const ModelInputParams& source_params = input.input_params;
  if (input.missing_required_host_views(staged_input) ||
      detail::has_contiguous_input_buffer_exclusions(source_params) ||
      !prepared_external_sources_supported(input)) {
    return false;
  }

  staged_input.input_params = source_params;
  detail::clear_contiguous_input_buffer_tensor_targets(
      staged_input.input_params);
  staged_input.sampling_params = input.sampling_params;
  staged_input.decoder_sampling_params = input.decoder_sampling_params;

  torch::Tensor positions_for_device =
      detail::normalize_positions_for_device(staged_input.positions_host);
  input_plan_.entries.clear();
  host_metadata_plan_.entries.clear();
  detail::ForwardInputBufferPlan& plan = input_plan_;
  detail::ForwardInputBufferPlan& host_metadata_plan = host_metadata_plan_;
  if (!plan.add(staged_input.token_ids_host, &staged_input.token_ids) ||
      !plan.add(positions_for_device, &staged_input.positions) ||
      !add_prepared_attention_to_plan(
          source_params.attention, staged_input.input_params.attention, plan) ||
      !add_prepared_model_tensors_to_plan(
          source_params, staged_input.input_params, plan) ||
      !detail::add_sampling_to_plan(
          input.sampling_params, staged_input.sampling_params, plan) ||
      !detail::add_sampling_to_plan(input.decoder_sampling_params,
                                    staged_input.decoder_sampling_params,
                                    plan) ||
      !add_prepared_host_model_tensors_to_plan(
          source_params, staged_input.input_params, host_metadata_plan)) {
    return false;
  }

  const uint64_t device_input_bytes = plan.prepare_layout();
  const uint64_t host_metadata_offset = detail::align_up(
      device_input_bytes, detail::kForwardInputBufferAlignment);
  const uint64_t host_metadata_bytes = host_metadata_plan.prepare_layout();
  const uint64_t total_bytes = host_metadata_offset + host_metadata_bytes;
  detail::ForwardInputHostCopyStats h2d_stats;
  staged_input.prepared_input_layout_signature =
      combine_layout_signatures(plan.layout_signature(),
                                host_metadata_plan.layout_signature(),
                                host_metadata_offset);
  uint64_t segment_offset =
      detail::align_up(used_bytes_, detail::kForwardInputBufferAlignment);
  uint64_t segment_capacity = capacity_bytes_ - segment_offset;
  if (partition_count_ > 0) {
    CHECK_LT(next_partition_, partition_count_)
        << "Prepared Task staged more invocation inputs than reserved";
    segment_offset =
        static_cast<uint64_t>(next_partition_) * partition_stride_bytes_;
    segment_capacity = partition_stride_bytes_;
    ++next_partition_;
  }
  CHECK_LE(segment_offset, capacity_bytes_);
  CHECK_LE(total_bytes, segment_capacity)
      << "Prepared Task invocation input requires " << total_bytes
      << " bytes, exceeding its fixed Arena partition of " << segment_capacity
      << " bytes; total Arena capacity is " << capacity_bytes_ << " bytes";
  if (total_bytes > 0) {
    const int64_t tensor_offset = static_cast<int64_t>(segment_offset);
    const int64_t total_tensor_size = static_cast<int64_t>(total_bytes);
    staged_input.input_host_buffer = host_buffer_.narrow(
        /*dim=*/0, tensor_offset, total_tensor_size);
    staged_input.device_input_buffer = device_buffer_.narrow(
        /*dim=*/0, tensor_offset, total_tensor_size);
    if (device_input_bytes > 0) {
      const int64_t device_tensor_size =
          static_cast<int64_t>(device_input_bytes);
      torch::Tensor host_device_input = staged_input.input_host_buffer.narrow(
          /*dim=*/0, /*start=*/0, device_tensor_size);
      torch::Tensor device_input = staged_input.device_input_buffer.narrow(
          /*dim=*/0, /*start=*/0, device_tensor_size);
      plan.pack_host_buffer(host_device_input);
      h2d_stats = plan.copy_host_sources(host_device_input, device_input);
      plan.bind_device_views(device_input, device_);
      plan.copy_device_sources();
    }
    if (host_metadata_bytes > 0) {
      const int64_t host_tensor_offset =
          static_cast<int64_t>(host_metadata_offset);
      const int64_t host_tensor_size =
          static_cast<int64_t>(host_metadata_bytes);
      torch::Tensor host_metadata = staged_input.input_host_buffer.narrow(
          /*dim=*/0, host_tensor_offset, host_tensor_size);
      host_metadata_plan.pack_host_buffer(host_metadata);
      host_metadata_plan.bind_device_views(host_metadata,
                                           torch::Device(torch::kCPU));
    }
  }
  GraphInput& staged_graph = staged_input.input_params.graph;
  if (source_params.is_spec_verify) {
    staged_graph.input_tokens_override = staged_input.token_ids;
    staged_graph.spec_verify_draft_token_sources.clear();
    staged_graph.spec_verify_source_addresses_stable = false;
    staged_graph.spec_verify_static_graph_tasks_prepared = false;
    staged_graph.prepared_spec_verify_direct_bind = true;
#if defined(USE_NPU)
    staged_graph.acl_graph_task_update_context.reset();
#endif
  } else if (source_params.graph.input_tokens_override.defined()) {
    staged_graph.input_tokens_override = staged_input.token_ids;
  }
  release_rebound_attention_buffers(staged_input.input_params.attention);
  record_transfer_stats(plan, h2d_stats, /*accumulate=*/false, staged_input);
  used_bytes_ = std::max(used_bytes_, segment_offset + total_bytes);

  // The packed pinned buffer is the only Host source retained beyond Prepare
  // Ack. Do not keep aliases into the caller-owned ForwardInput.
  staged_input.token_ids_host = torch::Tensor();
  staged_input.positions_host = torch::Tensor();
  staged_input.device_tensors_ready = true;
  staged_input.input_host_buffer_has_layout = false;
  return true;
}

bool PreparedInputArena::stage_generated_metadata(ForwardInput& staged_input) {
  CHECK_EQ(partition_count_, 0)
      << "Worker-generated metadata staging is only valid for the "
         "single-invocation Prepared LLM adapter";
  if (!prepared_external_sources_supported(staged_input)) {
    return false;
  }

  input_plan_.entries.clear();
  detail::ForwardInputBufferPlan& plan = input_plan_;
  if (!add_prepared_attention_to_plan(staged_input.input_params.attention,
                                      staged_input.input_params.attention,
                                      plan,
                                      &device_buffer_) ||
      !add_prepared_generated_model_tensors_to_plan(staged_input.input_params,
                                                    staged_input.input_params,
                                                    device_buffer_,
                                                    plan)) {
    return false;
  }
  if (plan.entries.empty()) {
    release_rebound_attention_buffers(staged_input.input_params.attention);
    return true;
  }

  const uint64_t total_bytes = plan.prepare_layout();
  detail::ForwardInputHostCopyStats h2d_stats;
  const uint64_t segment_offset =
      detail::align_up(used_bytes_, detail::kForwardInputBufferAlignment);
  CHECK_LE(segment_offset, capacity_bytes_);
  CHECK_LE(total_bytes, capacity_bytes_ - segment_offset)
      << "Prepared Worker-generated metadata requires " << total_bytes
      << " bytes after the primary input, exceeding the remaining Slot "
         "Arena capacity of "
      << capacity_bytes_ - segment_offset << " bytes";

  const int64_t tensor_size = static_cast<int64_t>(total_bytes);
  const int64_t tensor_offset = static_cast<int64_t>(segment_offset);
  torch::Tensor host_segment =
      host_buffer_.narrow(0, tensor_offset, tensor_size);
  torch::Tensor device_segment =
      device_buffer_.narrow(0, tensor_offset, tensor_size);
  plan.pack_host_buffer(host_segment);
  h2d_stats = plan.copy_host_sources(host_segment, device_segment);
  plan.bind_device_views(device_segment, device_);
  plan.copy_device_sources();
  release_rebound_attention_buffers(staged_input.input_params.attention);
  record_transfer_stats(plan, h2d_stats, /*accumulate=*/true, staged_input);

  used_bytes_ = segment_offset + total_bytes;
  staged_input.input_host_buffer = host_buffer_.narrow(
      /*dim=*/0, /*start=*/0, static_cast<int64_t>(used_bytes_));
  staged_input.device_input_buffer = device_buffer_.narrow(
      /*dim=*/0, /*start=*/0, static_cast<int64_t>(used_bytes_));
  staged_input.prepared_input_layout_signature =
      combine_layout_signatures(staged_input.prepared_input_layout_signature,
                                plan.layout_signature(),
                                segment_offset);
  return true;
}

}  // namespace xllm
