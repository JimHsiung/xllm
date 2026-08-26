/* Copyright 2025-2026 The xLLM Authors.

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

#include "acl_graph_executor_impl.h"

#include <c10/core/Device.h>
#include <c10/core/TensorOptions.h>
#include <glog/logging.h>
#include <torch/torch.h>
#include <torch_npu/csrc/core/npu/NPUGuard.h>
#include <torch_npu/csrc/libs/init_npu.h>
#include <torch_npu/torch_npu.h>

#include <algorithm>
#include <string_view>
#include <utility>

#include "core/common/global_flags.h"
#include "core/framework/config/execution_config.h"
#ifdef TORCH_HIGHER_THAN_PTA6
#include <torch_npu/csrc/framework/OpCommand.h>
#else
#include <torch_npu/csrc/aten/NPUNativeFunctions.h>
#include <torch_npu/csrc/framework/utils/OpPreparation.h>
#endif
#include "core/common/metrics.h"
#include "core/framework/speculative/mtp_async_state.h"
#include "core/kernels/npu/tilelang/tilelang_ops_api.h"
#include "core/kernels/ops_api.h"
#include "core/platform/device.h"
#include "core/platform/npu/acl_graph_task_update_context.h"
#include "core/util/utils.h"
#include "core/util/verbose_trace_logger.h"
#include "platform/npu/device_capture_lock.h"
#include "runtime/prepared_task/prepared_input_arena.h"

namespace xllm::npu {

namespace {
constexpr uint64_t kSpecVerifyGraphKeyMask = 1ull << 63;
constexpr uint64_t kSpecVerifyQMaxSeqLenShift = 32;
constexpr uint64_t kSpecVerifyBucketMask = (1ull << 16) - 1;
constexpr uint64_t kSpecVerifyFieldMask = (1ull << 16) - 1;
constexpr uint64_t kSpecVerifyExpandedBlockMask = (1ull << 15) - 1;
constexpr uint64_t kStaticGraphTaskHashSeed = 0x6a09e667f3bcc909ull;
constexpr size_t kMaxStaticMtpGraphVariantsPerSlot = 16;
constexpr uint64_t kMlaGraphKeyMask = 1ull << 62;
constexpr uint64_t kMlaGraphKeyPayloadMask = (1ull << 62) - 1;
constexpr uint64_t kPreparedGraphKeySeed = 0x510e527fade682d1ull;
constexpr uint64_t kPreparedExternalGraphInputSeed = 0x1f83d9abfb41bd6bull;
// Keep direct-bound speculative Target Verify dormant until a real-model
// capture/replay run validates its fixed Arena and numerical contracts.
constexpr bool kPreparedSpecVerifyGraphE2EValidated = false;
// Keep authoritative DSA Graph dormant until its capture/replay address and
// numerical contracts run on a real model. The implementation is compiled and
// its fixed-workspace builder is covered independently.
constexpr bool kPreparedDeviceDsaGraphE2EValidated = false;

std::string_view prepared_invocation_mode_name(PreparedInvocationMode mode) {
  switch (mode) {
    case PreparedInvocationMode::EAGER:
      return "eager";
    case PreparedInvocationMode::GRAPH_REPLAY:
      return "replay";
    case PreparedInvocationMode::GRAPH_CAPTURE:
      return "capture";
  }
  LOG(FATAL) << "Unknown Prepared invocation mode";
  return "unknown";
}

void trace_prepared_speculative_graph_execution(
    const PreparedSlotBinding& binding,
    const ForwardInput& input) {
  const ModelInputParams& params = input.input_params;
  if (!params.is_spec_verify ||
      !params.meta.batch_forward_type.is_chunked_prefill() ||
      !params.graph.use_expanded_decode_for_spec_verify_attention ||
      params.num_accepted_tokens_host.size() != 1) {
    return;
  }
  XLLM_VERBOSE_TRACE()
      << "event=prepared_speculative_graph_execution mode="
      << prepared_invocation_mode_name(binding.mode)
      << " slot_id=" << binding.slot_id << " graph_key=" << binding.graph_key
      << " graph_warmup=" << static_cast<int32_t>(params.meta.is_graph_warmup)
      << " accepted_length=" << params.num_accepted_tokens_host.front()
      << " verify_width=" << params.meta.q_max_seq_len
      << " static_graph_tasks_prepared="
      << static_cast<int32_t>(binding.static_graph_tasks_prepared);
}

bool tensor_is_within_buffer(const torch::Tensor& tensor,
                             const torch::Tensor& buffer) {
  if (!tensor.defined() || !buffer.defined() || tensor.numel() == 0) {
    return false;
  }
  const uintptr_t buffer_begin = reinterpret_cast<uintptr_t>(buffer.data_ptr());
  const uint64_t buffer_bytes =
      static_cast<uint64_t>(buffer.numel() * buffer.element_size());
  const uintptr_t buffer_end = buffer_begin + buffer_bytes;
  const uintptr_t tensor_begin = reinterpret_cast<uintptr_t>(tensor.data_ptr());
  const uint64_t tensor_bytes =
      static_cast<uint64_t>(tensor.numel() * tensor.element_size());
  const uintptr_t tensor_end = tensor_begin + tensor_bytes;
  return tensor_begin >= buffer_begin && tensor_end <= buffer_end;
}

bool tensor_is_undefined_or_within_buffer(const torch::Tensor& tensor,
                                          const torch::Tensor& buffer) {
  return !tensor.defined() || tensor.numel() == 0 ||
         tensor_is_within_buffer(tensor, buffer);
}

bool prepared_graph_inputs_are_arena_backed(const ForwardInput& input) {
  const torch::Tensor& buffer = input.device_input_buffer;
  const AttentionInput& attention = input.input_params.attention;
  const std::array<const torch::Tensor*, 28> graph_inputs = {
      &input.token_ids,
      &input.positions,
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
      &input.input_params.embedding.input_embedding,
      &input.input_params.embedding.linear_state_indices,
      &input.input_params.embedding.mtp_bootstrap_embeddings,
      &input.input_params.graph.attn_mask,
      &input.input_params.graph.tiling_data,
      &input.input_params.graph.expanded_kv_seq_lens,
      &input.input_params.graph.expanded_block_tables,
      &input.input_params.graph.expanded_paged_kv_indptr,
      &input.input_params.graph.expanded_paged_kv_indices,
      &input.input_params.graph.expanded_paged_kv_last_page_len,
      &input.input_params.graph.expanded_tiling_data};
  return std::all_of(graph_inputs.begin(),
                     graph_inputs.end(),
                     [&buffer](const torch::Tensor* tensor) {
                       return tensor_is_undefined_or_within_buffer(*tensor,
                                                                   buffer);
                     });
}

bool uses_static_mtp_graph_task_variant(const ModelInputParams& params,
                                        uint32_t bucket_num_tokens,
                                        int64_t block_size) {
  const int64_t batch_size = params.meta.num_sequences;
  const int64_t spec_width = params.meta.q_max_seq_len;
  return params.is_spec_verify &&
         params.meta.batch_forward_type.is_chunked_prefill() &&
         params.graph.use_expanded_decode_for_spec_verify_attention &&
         (params.graph.spec_verify_source_addresses_stable ||
          params.graph.prepared_spec_verify_direct_bind) &&
         kernel::npu::tilelang::has_spec_verify_graph_update_specialization(
             spec_width, block_size) &&
         batch_size == 1 && spec_width > 0 &&
         bucket_num_tokens == batch_size * spec_width &&
         params.parallel.query_start_loc.size() ==
             static_cast<size_t>(batch_size + 1) &&
         params.embedding.linear_state_ids.size() ==
             static_cast<size_t>(batch_size) &&
         params.num_accepted_tokens_host.size() ==
             static_cast<size_t>(batch_size);
}

uint64_t mix_graph_key(uint64_t hash, uint64_t value) {
  value += 0x9e3779b97f4a7c15ull;
  value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ull;
  value = (value ^ (value >> 27)) * 0x94d049bb133111ebull;
  value ^= value >> 31;
  return hash ^ (value + 0x9e3779b97f4a7c15ull + (hash << 6) + (hash >> 2));
}

struct PreparedExternalGraphInputIdentity {
  uint64_t layout_signature = kPreparedExternalGraphInputSeed;
  std::vector<const void*> addresses;
};

std::optional<PreparedExternalGraphInputIdentity>
prepared_external_graph_input_identity(const ForwardInput& input) {
  PreparedExternalGraphInputIdentity identity;
  const ModelInputParams& params = input.input_params;
  if (!params.dsa_device_geometry_authoritative) {
    return identity;
  }
  if (!input.positions.defined() ||
      params.dsa_device_geometry_workspace == nullptr ||
      params.device_multi_block_tables.empty()) {
    return std::nullopt;
  }

  const torch::Device expected_device = input.positions.device();
  identity.addresses.reserve(params.device_multi_block_tables.size() * 2 + 32);
  auto append_required_tensor = [&identity, &expected_device](
                                    const torch::Tensor& tensor) {
    if (!tensor.defined() || tensor.numel() <= 0 ||
        tensor.device() != expected_device) {
      return false;
    }
    identity.layout_signature = mix_graph_key(
        identity.layout_signature, static_cast<uint64_t>(tensor.scalar_type()));
    identity.layout_signature = mix_graph_key(
        identity.layout_signature, static_cast<uint64_t>(tensor.dim()));
    for (int64_t dim = 0; dim < tensor.dim(); ++dim) {
      identity.layout_signature = mix_graph_key(
          identity.layout_signature, static_cast<uint64_t>(tensor.size(dim)));
      identity.layout_signature = mix_graph_key(
          identity.layout_signature, static_cast<uint64_t>(tensor.stride(dim)));
    }
    identity.addresses.emplace_back(tensor.data_ptr());
    return true;
  };

  for (const torch::Tensor& block_table : params.device_multi_block_tables) {
    if (!append_required_tensor(block_table)) {
      return std::nullopt;
    }
  }

  const DSADeviceGeometryWorkspace& workspace =
      *params.dsa_device_geometry_workspace;
  const std::array<const torch::Tensor*, 30> fixed_tensors = {
      &workspace.actual_seq_lengths_query,
      &workspace.kv_cu_seq_lens,
      &workspace.max_seqlen_q,
      &workspace.max_seqlen_kv,
      &workspace.start_pos,
      &workspace.c4_compact_positions,
      &workspace.c128_compact_positions,
      &workspace.c4_compact_slots,
      &workspace.c128_compact_slots,
      &workspace.swa_block_table,
      &workspace.token_indices,
      &workspace.position_values,
      &workspace.position_remainders,
      &workspace.boundary_mask,
      &workspace.boundary_ranks,
      &workspace.sentinel_indices,
      &workspace.destination_indices,
      &workspace.token_offsets,
      &workspace.token_candidates,
      &workspace.mapping_valid,
      &workspace.mapping_valid_aux,
      &workspace.swa_logical_column_indices,
      &workspace.swa_physical_columns,
      &workspace.swa_gathered,
      &workspace.swa_valid,
      &workspace.swa_valid_aux,
      &workspace.manager_row_indices,
      &params.attention.device.q_seq_lens,
      &params.attention.device.kv_seq_lens,
      &params.attention.device.new_cache_slots};
  for (const torch::Tensor* tensor : fixed_tensors) {
    if (!append_required_tensor(*tensor)) {
      return std::nullopt;
    }
  }
  if (workspace.manager_expanded_block_tables.size() <
      params.device_multi_block_tables.size()) {
    return std::nullopt;
  }
  for (size_t manager_id = 0;
       manager_id < params.device_multi_block_tables.size();
       ++manager_id) {
    if (!append_required_tensor(
            workspace.manager_expanded_block_tables[manager_id])) {
      return std::nullopt;
    }
  }
  return identity;
}

static_assert(spec_verify_attention_plan_bucket_unchecked(127, 128) == 2);
static_assert(spec_verify_attention_plan_bucket_unchecked(128, 128) == 3);
static_assert(spec_verify_attention_plan_bucket_unchecked(129, 128) == 4);
static_assert(spec_verify_attention_plan_bucket_unchecked(255, 128) == 4);
static_assert(spec_verify_attention_plan_bucket_unchecked(256, 128) == 5);
static_assert(spec_verify_attention_plan_bucket_unchecked(257, 128) == 6);

uint64_t paged_attention_plan_bucket(int64_t max_kv, int64_t block_size) {
  CHECK_GT(max_kv, 0);
  CHECK_GT(block_size, 0);
  return spec_verify_attention_plan_bucket_unchecked(max_kv, block_size);
}

std::optional<int64_t> prepared_spec_verify_max_kv_with_headroom(
    const ModelInputParams& params,
    int64_t block_size) {
  if (params.graph.expanded_kv_seq_lens_vec.empty() ||
      params.graph.spec_verify_kv_seq_len_headroom < 0 || block_size <= 0) {
    return std::nullopt;
  }
  const int64_t template_max_kv =
      *std::max_element(params.graph.expanded_kv_seq_lens_vec.begin(),
                        params.graph.expanded_kv_seq_lens_vec.end());
  const int64_t headroom = params.graph.spec_verify_kv_seq_len_headroom;
  if (!spec_verify_attention_plan_headroom_is_safe(
          template_max_kv, headroom, block_size)) {
    return std::nullopt;
  }
  return template_max_kv + headroom;
}

uint64_t spec_verify_packed_graph_key(uint32_t bucket_num_tokens,
                                      uint64_t q_max_seq_len,
                                      uint64_t block_table_width,
                                      uint64_t expanded_block_table_width) {
  CHECK_LE(bucket_num_tokens, kSpecVerifyBucketMask);
  CHECK_LE(q_max_seq_len, kSpecVerifyFieldMask);
  CHECK_LE(block_table_width, kSpecVerifyFieldMask);
  CHECK_LE(expanded_block_table_width, kSpecVerifyExpandedBlockMask);
  return kSpecVerifyGraphKeyMask | (expanded_block_table_width << 48) |
         (block_table_width << 32) | (q_max_seq_len << 16) |
         static_cast<uint64_t>(bucket_num_tokens);
}

uint64_t spec_verify_attention_plan_lookup_key(
    uint64_t packed_graph_key,
    const std::vector<int32_t>& expanded_kv_seq_lens,
    int64_t kv_seq_len_headroom,
    int64_t block_size) {
  CHECK(!expanded_kv_seq_lens.empty());
  CHECK_GE(kv_seq_len_headroom, 0);
  const int64_t max_kv = *std::max_element(expanded_kv_seq_lens.begin(),
                                           expanded_kv_seq_lens.end()) +
                         kv_seq_len_headroom;
  return mix_graph_key(packed_graph_key,
                       paged_attention_plan_bucket(max_kv, block_size));
}

uint64_t spec_verify_attention_plan_lookup_key(uint32_t bucket_num_tokens,
                                               const ModelInputParams& params,
                                               int64_t block_size) {
  CHECK(params.attention.device.block_tables.defined());
  CHECK(params.graph.expanded_block_tables.defined());
  const uint64_t packed_key = spec_verify_packed_graph_key(
      bucket_num_tokens,
      static_cast<uint64_t>(std::max<int32_t>(params.meta.q_max_seq_len, 1)),
      static_cast<uint64_t>(params.attention.device.block_tables.size(1)),
      static_cast<uint64_t>(params.graph.expanded_block_tables.size(1)));
  return spec_verify_attention_plan_lookup_key(
      packed_key,
      params.graph.expanded_kv_seq_lens_vec,
      params.graph.spec_verify_kv_seq_len_headroom,
      block_size);
}

uint64_t static_mtp_graph_task_key(uint64_t base_key,
                                   const StaticGraphTaskSignature& signature) {
  uint64_t hash = mix_graph_key(kStaticGraphTaskHashSeed, base_key);
  hash = mix_graph_key(hash, static_cast<uint64_t>(signature.linear_state_id));
  hash =
      mix_graph_key(hash, static_cast<uint64_t>(signature.num_accepted_tokens));
  hash = mix_graph_key(hash,
                       static_cast<uint64_t>(signature.query_start_loc_begin));
  hash =
      mix_graph_key(hash, static_cast<uint64_t>(signature.query_start_loc_end));
  // Retain the spec-verify namespace bit. A signature comparison on replay
  // guards correctness even in the astronomically unlikely event of a hash
  // collision.
  return hash | kSpecVerifyGraphKeyMask;
}

std::pair<torch::Tensor, torch::Tensor> find_attention_plan_kv_cache(
    const std::vector<KVCache>& kv_caches) {
  for (const auto& cache : kv_caches) {
    auto k_cache = cache.get_k_cache();
    auto v_cache = cache.get_v_cache();
    if (k_cache.defined() && v_cache.defined() && k_cache.numel() > 0 &&
        v_cache.numel() > 0) {
      return {std::move(k_cache), std::move(v_cache)};
    }
  }
  return {torch::Tensor(), torch::Tensor()};
}

std::optional<std::array<const void*, 11>> spec_verify_input_addresses(
    const torch::Tensor& tokens,
    const torch::Tensor& positions,
    const ModelInputParams& params) {
  const torch::Tensor& graph_tokens =
      params.graph.input_tokens_override.defined()
          ? params.graph.input_tokens_override
          : tokens;
  // The graph key fixes tensor view shapes; this address list protects their
  // backing storage. Fixed-capacity packed buffers keep the corresponding
  // strides stable across replay generations.
  const std::array<const torch::Tensor*, 11> sources = {
      &graph_tokens,
      &positions,
      &params.attention.device.q_seq_lens,
      &params.attention.device.kv_seq_lens,
      &params.attention.device.new_cache_slots,
      &params.attention.device.block_tables,
      &params.embedding.linear_state_indices,
      &params.num_accepted_tokens,
      &params.attention.device.q_cu_seq_lens,
      &params.graph.expanded_kv_seq_lens,
      &params.graph.expanded_block_tables};
  std::array<const void*, 11> addresses;
  for (size_t i = 0; i < sources.size(); ++i) {
    if (!sources[i]->defined()) {
      return std::nullopt;
    }
    addresses[i] = sources[i]->data_ptr();
  }
  return addresses;
}

ModelOutput forward_eager(CausalLM* model,
                          const torch::Tensor& tokens,
                          const torch::Tensor& positions,
                          std::vector<KVCache>& kv_cache,
                          const ModelInputParams& params) {
  const torch::Tensor& verify_tokens =
      params.graph.input_tokens_override.defined()
          ? params.graph.input_tokens_override
          : tokens;
  torch::Tensor materialized_tokens =
      mtp_async::materialize_speculative_verify_tokens(
          verify_tokens, params.graph.spec_verify_draft_token_sources);
  return model->forward(materialized_tokens, positions, kv_cache, params);
}

void hash_graph_key_value(uint64_t& hash, uint64_t value) {
  constexpr uint64_t kFnvPrime = 1099511628211ull;
  for (int32_t i = 0; i < 8; ++i) {
    hash ^= (value >> (i * 8)) & 0xffull;
    hash *= kFnvPrime;
  }
}

uint64_t get_mla_graph_key(uint32_t bucket_num_tokens,
                           int32_t capture_kv_seq_len_bucket) {
  constexpr uint64_t kFnvOffsetBasis = 1469598103934665603ull;
  uint64_t hash = kFnvOffsetBasis;
  hash_graph_key_value(hash, bucket_num_tokens);
  hash_graph_key_value(hash, static_cast<uint64_t>(capture_kv_seq_len_bucket));

  return kMlaGraphKeyMask | (hash & kMlaGraphKeyPayloadMask);
}
}  // namespace

bool AclGraph::capture(CausalLM* model,
                       const runtime::Options& options,
                       const torch::Tensor& tokens,
                       const torch::Tensor& positions,
                       const ModelInputParams& params,
                       std::vector<KVCache>& kv_cache,
                       uint32_t bucket_num_tokens,
                       c10_npu::MempoolId_t graph_pool) {
  CHECK(!is_prepared_graph_);
  // Save bucket num_tokens for this graph instance
  num_tokens_ = bucket_num_tokens;

  // Get actual num_tokens from tokens tensor
  // const uint32_t actual_num_tokens = tokens.size(0);

  auto& tensor_options = model->options();

  torch::npu::synchronize();

  // Begin graph capture using NPUGraph mempool for temporary tensor management
  // Get current NPU stream from libtorch NPU API
  aclrtStream stream =
      c10_npu::getCurrentNPUStream(tensor_options.device().index()).stream();

  // For hybrid models (e.g., qwen3_next with mixed GDN/full_attention layers),
  // we need to find the first Full Attention layer to get the correct kv_cache.
  // GDN layers have empty key_cache_/value_cache_ while Full Attention layers
  // have valid kv caches. Using layer 0's cache directly would be incorrect
  // if layer 0 is a GDN layer.
  auto [k_cache, v_cache] = find_attention_plan_kv_cache(kv_cache);
  const uint32_t actual_num_tokens =
      static_cast<uint32_t>(tokens.size(/*dim=*/0));
  CHECK_GE(num_tokens_, actual_num_tokens)
      << "num_tokens_ >= actual_num_tokens";
  const bool update_spec_verify_tokens =
      params.graph.spec_verify_source_addresses_stable &&
      params.graph.input_tokens_override.defined() && params.is_spec_verify &&
      params.meta.batch_forward_type.is_chunked_prefill();
  auto graph_params = persistent_param_.update(tokens,
                                               k_cache,
                                               v_cache,
                                               positions,
                                               params,
                                               num_tokens_,
                                               /*return_capture_params=*/true,
                                               /*skip_token_update=*/
                                               update_spec_verify_tokens,
                                               /*for_capture=*/true);
  if (update_spec_verify_tokens) {
    persistent_param_.update_spec_verify_inputs(
        tokens,
        positions,
        params,
        num_tokens_,
        SpecVerifyInputUpdateScope::TOKENS_ONLY);
  }

  // Use the returned ModelInputParams for graph capture
  CHECK(graph_params.has_value())
      << "update() should return ModelInputParams when "
         "return_capture_params=true";
  const auto spec_verify_attention_plan =
      persistent_param_.paged_attention_plan_descriptor(
          actual_num_tokens, params.meta.q_max_seq_len);
  int64_t spec_verify_kv_split_core_count = 0;
  if (spec_verify_attention_plan.has_value()) {
    spec_verify_kv_split_core_count = static_cast<int64_t>(
        spec_verify_attention_plan->normalized_tiling.at(static_cast<size_t>(
            spec_verify_attention_plan->layout.kv_split_core_count_offset)));
  }
  const bool can_use_explicit_spec_verify_replay_update =
      model->is_hybrid_linear_attention() &&
      kernel::npu::tilelang::has_spec_verify_graph_update_specialization(
          params.meta.q_max_seq_len, options.block_size()) &&
      params.graph.spec_verify_source_addresses_stable &&
      params.is_spec_verify &&
      params.meta.batch_forward_type.is_chunked_prefill() &&
      spec_verify_attention_plan.has_value() &&
      spec_verify_kv_split_core_count > 0 &&
      actual_num_tokens == bucket_num_tokens && params.meta.num_sequences > 0 &&
      params.meta.q_max_seq_len > 0 &&
      actual_num_tokens == static_cast<uint32_t>(params.meta.num_sequences) *
                               params.meta.q_max_seq_len;
  if (can_use_explicit_spec_verify_replay_update) {
    spec_verify_block_size_ = options.block_size();
    spec_verify_kv_split_core_count_ = spec_verify_kv_split_core_count;
    spec_verify_paged_attention_tiling_layout_ =
        spec_verify_attention_plan->layout;
    const int64_t paged_attention_tiling_words = static_cast<int64_t>(
        spec_verify_attention_plan->normalized_tiling.size());
    CHECK_GT(paged_attention_tiling_words, 0);
    CHECK_LE(paged_attention_tiling_words,
             persistent_param_.tiling_data().numel());
    graph_paged_attention_tiling_data_ =
        persistent_param_.tiling_data().clone();
    // The paged-attention launch and the TileLang dynamic update must target
    // storage owned by this graph. A slot's persistent tiling tensor is shared
    // by every graph variant and can be overwritten while another variant is
    // still replaying.
    graph_params->graph.tiling_data = graph_paged_attention_tiling_data_;
    graph_params->graph.expanded_tiling_data =
        graph_paged_attention_tiling_data_;
  }
  prepare_model_graph_metadata(
      model,
      persistent_param_.persistent_positions(num_tokens_),
      graph_params.value());
  if (graph_paged_attention_tiling_data_.defined()) {
    spec_verify_input_addresses_at_capture_ =
        spec_verify_input_addresses(tokens, positions, params);
    // Raw TileLang launches are not replayed by the captured ACL graph. Run
    // the initial tiling update on the producer stream and let the existing
    // stream synchronization complete it before capture begins.
    update_spec_verify_attention_tiling(graph_params.value());
  }

  if (model->is_hybrid_linear_attention()) {
    graph_task_context_ = std::make_shared<AclGraphTaskUpdateContext>();
    graph_task_context_->begin_capture();
    graph_params->graph.acl_graph_task_update_context = graph_task_context_;
  }
  const bool capture_static_graph_tasks = uses_static_mtp_graph_task_variant(
      graph_params.value(), num_tokens_, options.block_size());
  // Synchronize stream to ensure all data is copied to graph persistent buffers
  aclrtSynchronizeStream(stream);

  // Acquire device-level lock to prevent prepare_work_before_execute from
  // executing simultaneously, which would trigger synchronous operations
  // that conflict with capture mode
  auto device_idx = tensor_options.device().index();
  Device::empty_cache(device_idx);

  bool need_restore_stream = false;
  graph_stream_ = stream;

  // capture lock scope
  {
    auto& capture_lock =
        ::xllm::npu::DeviceCaptureLock::get_instance().get_lock(device_idx);
    std::lock_guard<std::mutex> lock_guard(capture_lock);

    if (c10_npu::getCurrentNPUStream(device_idx) ==
        c10_npu::getDefaultNPUStream(device_idx)) {
      c10_npu::setCurrentNPUStream(capture_stream_.value());
      aclrtSynchronizeStream(capture_stream_.value().stream());
      graph_stream_ = capture_stream_.value().stream();
      need_restore_stream = true;
    }
    VLOG(kGraphExecutorLogVerboseLevel)
        << "ACL graph capture begin, bucket_num_tokens=" << bucket_num_tokens
        << ", actual_num_tokens=" << actual_num_tokens;

    // Reuse one pool per graph slot so bucket captures can share allocator
    // storage while the double-buffer slots remain independent.
    bool capture_started = false;
    try {
      graph_.capture_begin(
          graph_pool,
          aclmdlRICaptureMode::ACL_MODEL_RI_CAPTURE_MODE_THREAD_LOCAL);
      capture_started = true;
      // Execute forward pass - NPUGraph mempool manages temporary tensors
      auto forward_result =
          model->forward({persistent_param_.persistent_tokens(num_tokens_)},
                         {persistent_param_.persistent_positions(num_tokens_)},
                         kv_cache,
                         {graph_params.value()});

      persistent_param_.set_hidden_states(forward_result.hidden_states);
      if (options.enable_graph_aux_hidden_states() &&
          forward_result.aux_hidden_states.defined()) {
        persistent_param_.set_aux_hidden_states(
            forward_result.aux_hidden_states);
      }
      graph_.capture_end();
      capture_started = false;
    } catch (...) {
      if (capture_started) {
        try {
          graph_.capture_end();
        } catch (const std::exception& cleanup_error) {
          LOG(ERROR) << "ACL graph capture_end during cleanup failed: "
                     << cleanup_error.what();
        } catch (...) {
          LOG(ERROR) << "ACL graph capture_end during cleanup failed.";
        }
        graph_.reset();
      }
      if (need_restore_stream) {
        c10_npu::setCurrentNPUStream(
            c10_npu::getDefaultNPUStream(tensor_options.device().index()));
      }
      throw;
    }
    if (graph_task_context_ != nullptr) {
      graph_task_context_->end_capture();
    }
    // Lock is automatically released here when lock goes out of scope
    if (need_restore_stream) {
      c10_npu::setCurrentNPUStream(
          c10_npu::getDefaultNPUStream(tensor_options.device().index()));
    }
  }
  // Synchronize and test replay to verify graph capture
  aclrtSynchronizeStream(graph_stream_);
  aclrtSynchronizeStream(stream);
  graph_.replay();
  update_graph_tasks(graph_params.value());
  if (capture_static_graph_tasks) {
    capture_static_graph_task_signature(graph_params.value());
  }
  make_current_stream_wait_for_graph(stream);
  return true;
}

ModelOutput AclGraph::capture_prepared(CausalLM* model,
                                       const runtime::Options& options,
                                       const ForwardInput& input,
                                       std::vector<KVCache>& kv_cache,
                                       c10_npu::MempoolId_t graph_pool) {
  CHECK(model != nullptr);
  CHECK(input.token_ids.defined());
  CHECK(input.positions.defined());
  CHECK(input.device_input_buffer.defined());
  CHECK_NE(input.prepared_input_layout_signature, 0);

  const ModelInputParams* graph_params = &input.input_params;
  if (model->requires_graph_forward_metadata() ||
      model->is_hybrid_linear_attention()) {
    CHECK(prepared_model_graph_params_.has_value())
        << "Prepared ACL graph metadata must be populated during Prepare";
    if (model->requires_graph_forward_metadata()) {
      CHECK(prepared_model_graph_params_->attn_metadata != nullptr)
          << "Prepared model graph metadata is missing attention metadata";
    }
    graph_params = &prepared_model_graph_params_.value();
  }

  is_prepared_graph_ = true;
  num_tokens_ = static_cast<uint32_t>(input.token_ids.size(/*dim=*/0));
  prepared_layout_signature_ = input.prepared_input_layout_signature;
  prepared_input_buffer_address_ = input.device_input_buffer.data_ptr();
  prepared_tokens_address_ = input.token_ids.data_ptr();
  prepared_positions_address_ = input.positions.data_ptr();
  const std::optional<PreparedExternalGraphInputIdentity> external_identity =
      prepared_external_graph_input_identity(input);
  CHECK(external_identity.has_value())
      << "Prepared ACL graph external input contract is incomplete";
  prepared_external_layout_signature_ = external_identity->layout_signature;
  prepared_external_input_addresses_ = external_identity->addresses;

  if (graph_paged_attention_tiling_data_.defined()) {
    CHECK_EQ(graph_paged_attention_tiling_data_.data_ptr(),
             prepared_graph_tiling_address_)
        << "Prepared graph-local tiling storage moved before capture";
    update_spec_verify_attention_tiling(*graph_params);
  }
  if (model->is_hybrid_linear_attention()) {
    graph_task_context_ = std::make_shared<AclGraphTaskUpdateContext>();
    graph_task_context_->begin_capture();
    prepared_model_graph_params_->graph.acl_graph_task_update_context =
        graph_task_context_;
    graph_params = &prepared_model_graph_params_.value();
  }
  const bool capture_static_graph_tasks = uses_static_mtp_graph_task_variant(
      *graph_params, num_tokens_, options.block_size());

  torch::npu::synchronize();
  const c10::DeviceIndex device_index = model->options().device().index();
  aclrtStream current_stream =
      c10_npu::getCurrentNPUStream(device_index).stream();
  graph_stream_ = current_stream;

  bool restore_default_stream = false;
  {
    auto& capture_lock =
        ::xllm::npu::DeviceCaptureLock::get_instance().get_lock(device_index);
    std::lock_guard<std::mutex> lock_guard(capture_lock);
    if (c10_npu::getCurrentNPUStream(device_index) ==
        c10_npu::getDefaultNPUStream(device_index)) {
      c10_npu::setCurrentNPUStream(capture_stream_.value());
      CHECK_EQ(aclrtSynchronizeStream(capture_stream_.value().stream()),
               ACL_SUCCESS);
      graph_stream_ = capture_stream_.value().stream();
      restore_default_stream = true;
    }

    bool capture_started = false;
    try {
      graph_.capture_begin(
          graph_pool,
          aclmdlRICaptureMode::ACL_MODEL_RI_CAPTURE_MODE_THREAD_LOCAL);
      capture_started = true;
      ModelOutput output = model->forward(
          {input.token_ids}, {input.positions}, kv_cache, {*graph_params});
      prepared_hidden_states_ = output.hidden_states;
      if (options.enable_graph_aux_hidden_states()) {
        prepared_aux_hidden_states_ = output.aux_hidden_states;
      }
      graph_.capture_end();
      capture_started = false;
    } catch (...) {
      if (capture_started) {
        try {
          graph_.capture_end();
        } catch (const std::exception& cleanup_error) {
          LOG(ERROR) << "Prepared ACL graph capture_end cleanup failed: "
                     << cleanup_error.what();
        } catch (...) {
          LOG(ERROR) << "Prepared ACL graph capture_end cleanup failed";
        }
        graph_.reset();
      }
      if (restore_default_stream) {
        c10_npu::setCurrentNPUStream(
            c10_npu::getDefaultNPUStream(device_index));
      }
      throw;
    }

    if (graph_task_context_ != nullptr) {
      graph_task_context_->end_capture();
    }

    if (restore_default_stream) {
      c10_npu::setCurrentNPUStream(c10_npu::getDefaultNPUStream(device_index));
    }
  }

  CHECK_EQ(aclrtSynchronizeStream(graph_stream_), ACL_SUCCESS);
  CHECK_EQ(aclrtSynchronizeStream(current_stream), ACL_SUCCESS);
  graph_.replay();
  const bool graph_tasks_updated = update_graph_tasks(*graph_params);
  if (capture_static_graph_tasks) {
    CHECK(graph_tasks_updated)
        << "Prepared hybrid Target Verify captured no causal-conv graph task";
    capture_static_graph_task_signature(*graph_params);
  }
  make_current_stream_wait_for_graph(current_stream);
  if (prepared_aux_hidden_states_.defined() &&
      prepared_aux_hidden_states_.numel() > 0) {
    return ModelOutput(
        prepared_hidden_states_, torch::Tensor(), prepared_aux_hidden_states_);
  }
  return ModelOutput(prepared_hidden_states_);
}

bool AclGraph::matches_prepared_input(const ForwardInput& input) const {
  const std::optional<PreparedExternalGraphInputIdentity> external_identity =
      prepared_external_graph_input_identity(input);
  return external_identity.has_value() && is_prepared_graph_ &&
         input.device_input_buffer.defined() && input.token_ids.defined() &&
         input.positions.defined() &&
         input.prepared_input_layout_signature == prepared_layout_signature_ &&
         input.device_input_buffer.data_ptr() ==
             prepared_input_buffer_address_ &&
         input.token_ids.data_ptr() == prepared_tokens_address_ &&
         input.positions.data_ptr() == prepared_positions_address_ &&
         external_identity->layout_signature ==
             prepared_external_layout_signature_ &&
         external_identity->addresses == prepared_external_input_addresses_;
}

void AclGraph::prepare_prepared_model_graph_metadata(
    CausalLM* model,
    const ForwardInput& input,
    const PagedAttentionPlanDescriptor* attention_plan,
    int64_t block_size) {
  CHECK(model != nullptr);
  const bool needs_prepared_params = model->requires_graph_forward_metadata() ||
                                     model->is_hybrid_linear_attention() ||
                                     attention_plan != nullptr;
  if (!needs_prepared_params) {
    prepared_model_graph_params_.reset();
    return;
  }

  ModelInputParams graph_params = input.input_params;
  if (attention_plan != nullptr) {
    CHECK(input.input_params.is_spec_verify);
    CHECK(input.input_params.meta.batch_forward_type.is_chunked_prefill());
    CHECK(input.input_params.graph.expanded_kv_seq_lens.defined());
    CHECK_GT(block_size, 0);
    if (prepared_attention_plan_descriptor_.has_value()) {
      CHECK(prepared_attention_plan_descriptor_.value() == *attention_plan)
          << "Prepared speculative Graph attention plan changed for one key";
    } else {
      prepared_attention_plan_descriptor_ = *attention_plan;
      CHECK(persistent_param_.tiling_data().defined());
      CHECK_GT(persistent_param_.tiling_data().numel(), 0);
      CHECK_LE(static_cast<int64_t>(attention_plan->normalized_tiling.size()),
               persistent_param_.tiling_data().numel());
      graph_paged_attention_tiling_data_ =
          torch::zeros_like(persistent_param_.tiling_data());
      std::vector<int32_t> normalized_tiling;
      normalized_tiling.reserve(attention_plan->normalized_tiling.size());
      for (uint32_t value : attention_plan->normalized_tiling) {
        normalized_tiling.emplace_back(static_cast<int32_t>(value));
      }
      torch::Tensor normalized_tiling_host = torch::tensor(
          normalized_tiling,
          torch::TensorOptions().dtype(torch::kInt).device(torch::kCPU));
      graph_paged_attention_tiling_data_
          .narrow(/*dim=*/0,
                  /*start=*/0,
                  /*length=*/normalized_tiling_host.numel())
          .copy_(normalized_tiling_host, /*non_blocking=*/true);
      prepared_graph_tiling_address_ =
          graph_paged_attention_tiling_data_.data_ptr();
      spec_verify_block_size_ = block_size;
      spec_verify_paged_attention_tiling_layout_ = attention_plan->layout;
      const int64_t split_core_offset = static_cast<int64_t>(
          attention_plan->layout.kv_split_core_count_offset);
      CHECK_LT(split_core_offset,
               static_cast<int64_t>(attention_plan->normalized_tiling.size()));
      spec_verify_kv_split_core_count_ = static_cast<int64_t>(
          attention_plan
              ->normalized_tiling[static_cast<size_t>(split_core_offset)]);
      CHECK_GT(spec_verify_kv_split_core_count_, 0);
    }
    CHECK_EQ(graph_paged_attention_tiling_data_.data_ptr(),
             prepared_graph_tiling_address_);
    graph_params.graph.tiling_data = graph_paged_attention_tiling_data_;
    graph_params.graph.expanded_tiling_data =
        graph_paged_attention_tiling_data_;
  }
  prepare_model_graph_metadata(model, input.positions, graph_params);
  prepared_model_graph_params_ = std::move(graph_params);
}

bool AclGraph::prepare_static_prepared_graph_tasks(
    const ModelInputParams& params,
    const c10_npu::NPUStream& signal_stream) {
  if (!static_graph_task_signature_matches(params)) {
    return false;
  }
  signal_static_graph_tasks(signal_stream);
  return true;
}

ModelOutput AclGraph::replay_prepared(const ForwardInput& input,
                                      bool static_graph_tasks_prepared) {
  CHECK(matches_prepared_input(input))
      << "Prepared ACL graph input address or layout changed after capture";
  CHECK_EQ(static_cast<uint32_t>(input.token_ids.size(/*dim=*/0)), num_tokens_);
  CHECK(!static_graph_task_signature_.has_value() ||
        static_graph_tasks_prepared)
      << "Prepared static causal-conv tasks were not signaled before replay";
  aclrtStream current_stream = c10_npu::getCurrentNPUStream().stream();
  if (graph_paged_attention_tiling_data_.defined()) {
    CHECK_EQ(graph_paged_attention_tiling_data_.data_ptr(),
             prepared_graph_tiling_address_)
        << "Prepared graph-local tiling storage moved after capture";
    update_spec_verify_attention_tiling(input.input_params);
  }
  make_graph_wait_for_current_stream(current_stream);
  graph_.replay();
  make_current_stream_wait_for_graph(current_stream);
  if (prepared_aux_hidden_states_.defined() &&
      prepared_aux_hidden_states_.numel() > 0) {
    return ModelOutput(
        prepared_hidden_states_, torch::Tensor(), prepared_aux_hidden_states_);
  }
  return ModelOutput(prepared_hidden_states_);
}

bool AclGraph::update_graph_tasks(const ModelInputParams& params) {
  if (graph_task_context_ == nullptr ||
      graph_task_context_->causal_conv1d_tasks.empty()) {
    return false;
  }

  const std::vector<int64_t> empty_host_args;
  CHECK(!params.parallel.query_start_loc.empty())
      << "causal_conv1d graph update requires padded query_start_loc";
  CHECK(!params.embedding.linear_state_ids.empty())
      << "causal_conv1d graph update requires padded cache indices";

  std::vector<int64_t> linear_state_indices_host(
      params.embedding.linear_state_ids.begin(),
      params.embedding.linear_state_ids.end());

  c10_npu::NPUStream update_stream = update_stream_.value();
  c10_npu::NPUStreamGuard stream_guard(update_stream);

  for (auto& task : graph_task_context_->causal_conv1d_tasks) {
    CHECK_EQ(params.parallel.query_start_loc.back(), task.x.size(0))
        << "causal_conv1d graph update host args must be padded to the "
           "capture x.shape[0]";
    CHECK_EQ(linear_state_indices_host.size() + 1,
             params.parallel.query_start_loc.size())
        << "cache_indices must be sequence-scoped";

    const std::vector<int64_t>& num_accepted_tokens =
        task.branch == CausalConv1dGraphBranch::kSpecVerify
            ? params.num_accepted_tokens_host
            : empty_host_args;
    if (task.branch == CausalConv1dGraphBranch::kSpecVerify) {
      CHECK_EQ(num_accepted_tokens.size(), linear_state_indices_host.size())
          << "spec causal_conv1d graph update requires accepted-token counts";
    }

    c10_npu::graph_task_update_begin(update_stream, task.handle);
    xllm::kernel::causal_conv1d_out(
        task.output,
        task.x,
        task.weight,
        task.conv_state,
        task.bias,
        torch::IntArrayRef(params.parallel.query_start_loc),
        torch::IntArrayRef(linear_state_indices_host),
        torch::IntArrayRef(empty_host_args),
        torch::IntArrayRef(num_accepted_tokens),
        task.activation_mode,
        task.pad_slot_id,
        task.run_mode);
    c10_npu::graph_task_update_end(update_stream);
    if (task.event != nullptr) {
      task.event->record(update_stream);
    }
  }
  return true;
}

void AclGraph::signal_static_graph_tasks(
    const c10_npu::NPUStream& signal_stream) {
  CHECK(graph_task_context_ != nullptr);
  for (auto& task : graph_task_context_->causal_conv1d_tasks) {
    CHECK(task.event != nullptr)
        << "static graph-task replay requires a captured ready event";
    task.event->record(signal_stream);
  }
}

bool AclGraph::static_graph_task_signature_matches(
    const ModelInputParams& params) const {
  const auto current_signature = make_static_graph_task_signature(params);
  return current_signature.has_value() &&
         static_graph_task_signature_ == current_signature;
}

void AclGraph::capture_static_graph_task_signature(
    const ModelInputParams& params) {
  static_graph_task_signature_ = make_static_graph_task_signature(params);
  CHECK(static_graph_task_signature_.has_value());
  LOG(INFO) << "Captured static MTP graph-task signature: linear_state_id="
            << static_graph_task_signature_->linear_state_id
            << ", accepted_tokens="
            << static_graph_task_signature_->num_accepted_tokens;
}

AclGraph::~AclGraph() {
  if (graph_stream_ != nullptr) {
    aclrtSynchronizeStream(graph_stream_);
  } else if (capture_stream_.has_value()) {
    aclrtSynchronizeStream(capture_stream_.value().stream());
  }
  if (replay_done_event_ != nullptr) {
    aclrtDestroyEvent(replay_done_event_);
    replay_done_event_ = nullptr;
  }
  if (replay_input_ready_event_ != nullptr) {
    aclrtDestroyEvent(replay_input_ready_event_);
    replay_input_ready_event_ = nullptr;
  }
}

void AclGraph::initialize_streams(c10::DeviceIndex device_index,
                                  const c10_npu::NPUStream& capture_stream) {
  capture_stream_ = capture_stream;
  update_stream_ = c10_npu::getStreamFromPool(true, device_index);
  device_index_ = device_index;
  CHECK_EQ(aclrtCreateEventWithFlag(&replay_input_ready_event_, ACL_EVENT_SYNC),
           ACL_SUCCESS)
      << "Failed to create ACL graph replay input-ready event";
  CHECK_EQ(aclrtCreateEventWithFlag(&replay_done_event_, ACL_EVENT_SYNC),
           ACL_SUCCESS)
      << "Failed to create ACL graph replay completion event";
  VLOG(kGraphExecutorLogVerboseLevel)
      << "Initialized capture_stream"
      << ", id: " << capture_stream_.value().id()
      << ", device_index: " << static_cast<int32_t>(device_index);
}

void AclGraph::make_graph_wait_for_current_stream(aclrtStream current_stream) {
  CHECK_NE(graph_stream_, nullptr) << "graph_stream is not initialized";
  CHECK_NE(replay_input_ready_event_, nullptr)
      << "replay_input_ready_event is not initialized";
  if (current_stream == graph_stream_) {
    return;
  }
  CHECK_EQ(aclrtRecordEvent(replay_input_ready_event_, current_stream),
           ACL_SUCCESS)
      << "aclrtRecordEvent(replay_input_ready_event) failed";
  CHECK_EQ(aclrtStreamWaitEvent(graph_stream_, replay_input_ready_event_),
           ACL_SUCCESS)
      << "aclrtStreamWaitEvent(graph_stream, replay_input_ready_event) failed";
}

void AclGraph::make_current_stream_wait_for_graph(aclrtStream current_stream) {
  CHECK_NE(graph_stream_, nullptr) << "graph_stream is not initialized";
  CHECK_NE(replay_done_event_, nullptr)
      << "replay_done_event is not initialized";
  CHECK_EQ(aclrtRecordEvent(replay_done_event_, graph_stream_), ACL_SUCCESS)
      << "aclrtRecordEvent(replay_done_event) failed";
  if (current_stream != graph_stream_) {
    CHECK_EQ(aclrtStreamWaitEvent(current_stream, replay_done_event_),
             ACL_SUCCESS)
        << "aclrtStreamWaitEvent(current_stream, replay_done_event) failed";
  }
}

void AclGraph::prepare_model_graph_metadata(CausalLM* model,
                                            const torch::Tensor& positions,
                                            ModelInputParams& params) {
  CHECK(model != nullptr) << "ACL graph model must not be null";
  if (!model->requires_graph_forward_metadata()) {
    return;
  }
  if (!model_graph_metadata_state_) {
    model_graph_metadata_state_ = model->create_graph_forward_metadata_state();
    CHECK(model_graph_metadata_state_)
        << "ACL graph metadata state must be initialized during capture";
  }
  model->prepare_graph_forward_metadata(
      model_graph_metadata_state_.get(), positions, params);
  CHECK(params.attn_metadata)
      << "model graph metadata preparation did not populate attn_metadata";
}

void AclGraph::update_spec_verify_attention_tiling(
    const ModelInputParams& params) {
  CHECK(graph_paged_attention_tiling_data_.defined());
  CHECK(spec_verify_paged_attention_tiling_layout_.has_value());
  const std::optional<int64_t> max_kv_with_headroom =
      prepared_spec_verify_max_kv_with_headroom(params,
                                                spec_verify_block_size_);
  CHECK(max_kv_with_headroom.has_value())
      << "speculative Graph KV headroom crosses an attention-plan bucket";
  kernel::npu::tilelang::spec_verify_attention_tiling_update(
      params.graph.expanded_kv_seq_lens,
      graph_paged_attention_tiling_data_,
      spec_verify_paged_attention_tiling_layout_.value(),
      params.meta.q_max_seq_len,
      spec_verify_block_size_,
      max_kv_with_headroom.value(),
      spec_verify_kv_split_core_count_);
}

ModelOutput AclGraph::replay(CausalLM* model,
                             const torch::Tensor& tokens,
                             const torch::Tensor& positions,
                             std::vector<KVCache>& kv_cache,
                             const ModelInputParams& params) {
  const uint32_t actual_num_tokens =
      static_cast<uint32_t>(tokens.size(/*dim=*/0));
  CHECK_LE(actual_num_tokens, num_tokens_)
      << "num_tokens mismatch: expected <= " << num_tokens_ << ", got "
      << actual_num_tokens;

  // Update persistent parameters with new input data
  // Note: tiling_data is updated in update() if needed - for hybrid models
  // (e.g., qwen3_next with mixed GDN/attention layers), tiling should only
  // be updated when Full Attention layers are involved, which is determined
  // by k_cache being valid and non-empty
  const bool needs_graph_metadata = model->requires_graph_forward_metadata() ||
                                    model->is_hybrid_linear_attention();
  const bool replay_inputs_prepared =
      replay_inputs_prepared_.exchange(false, std::memory_order_acq_rel);
  const bool can_use_prepared_inputs =
      replay_inputs_prepared && params.graph.input_tokens_override.defined() &&
      !needs_graph_metadata;
  std::optional<ModelInputParams> graph_params;
  if (graph_paged_attention_tiling_data_.defined()) {
    const auto current_addresses =
        spec_verify_input_addresses(tokens, positions, params);
    if (!spec_verify_input_addresses_at_capture_.has_value() ||
        current_addresses != spec_verify_input_addresses_at_capture_) {
      LOG_FIRST_N(ERROR, 1)
          << "Falling back to eager speculative verification because graph "
             "input source storage moved after capture.";
      COUNTER_INC(num_model_execution_total_eager);
      return forward_eager(model, tokens, positions, kv_cache, params);
    }
    // Raw TileLang launches require an explicit metadata refresh.
    persistent_param_.update_spec_verify_inputs(
        tokens,
        positions,
        params,
        num_tokens_,
        SpecVerifyInputUpdateScope::ALL_INPUTS);
    update_spec_verify_attention_tiling(params);
    // Explicit producer-stream updates have populated the persistent graph
    // inputs. Host-only task parameters remain current and are consumed by
    // update_graph_tasks() below.
    graph_params = params;
  } else if (can_use_prepared_inputs) {
    persistent_param_.update_tokens(
        tokens, params, actual_num_tokens, num_tokens_);
  } else {
    auto [k_cache, v_cache] = find_attention_plan_kv_cache(kv_cache);
    graph_params = persistent_param_.update(tokens,
                                            k_cache,
                                            v_cache,
                                            positions,
                                            params,
                                            num_tokens_,
                                            needs_graph_metadata);
    if (needs_graph_metadata) {
      CHECK(graph_params.has_value())
          << "ACL graph replay requires persistent params for graph metadata";
      prepare_model_graph_metadata(
          model,
          persistent_param_.persistent_positions(num_tokens_),
          graph_params.value());
    }
  }

  aclrtStream stream = c10_npu::getCurrentNPUStream().stream();

  if (graph_paged_attention_tiling_data_.defined()) {
    make_graph_wait_for_current_stream(stream);
  }
  const bool use_static_graph_tasks =
      graph_params.has_value() &&
      static_graph_task_signature_matches(graph_params.value());
  const bool static_graph_tasks_prepared =
      params.graph.spec_verify_static_graph_tasks_prepared;
  CHECK(!static_graph_tasks_prepared || use_static_graph_tasks)
      << "prepared static graph tasks do not match the replay signature";
  if (use_static_graph_tasks && !static_graph_tasks_prepared) {
    // Cold/fallback path: the final-draft pre-submit could not find this graph
    // variant. Signal its task-ready events immediately before replay; steady
    // supported-width cycles use the compute-stream pre-submit path instead.
    CHECK(update_stream_.has_value());
    signal_static_graph_tasks(update_stream_.value());
  }
  graph_.replay();
  if (model->is_hybrid_linear_attention()) {
    CHECK(graph_params.has_value())
        << "update() should return ModelInputParams for graph task update";
    if (use_static_graph_tasks) {
      // This graph variant's task-ready event was recorded before replay.
    } else {
      update_graph_tasks(graph_params.value());
    }
  }
  make_current_stream_wait_for_graph(stream);

  // Return the actual num_tokens portion of ModelOutput
  // Note: aux_hidden_states handling is done in AclGraphExecutorImpl::run()
  // since replay() doesn't have access to options
  return ModelOutput(get_hidden_states(actual_num_tokens));
}

void AclGraph::prepare_replay_inputs(const torch::Tensor& tokens,
                                     const torch::Tensor& positions,
                                     std::vector<KVCache>& kv_cache,
                                     const ModelInputParams& params) {
  if (graph_paged_attention_tiling_data_.defined()) {
    return;
  }
  const uint32_t actual_num_tokens =
      static_cast<uint32_t>(tokens.size(/*dim=*/0));
  CHECK_LE(actual_num_tokens, num_tokens_)
      << "num_tokens mismatch: expected <= " << num_tokens_ << ", got "
      << actual_num_tokens;
  auto [k_cache, v_cache] = find_attention_plan_kv_cache(kv_cache);
  persistent_param_.update(tokens,
                           k_cache,
                           v_cache,
                           positions,
                           params,
                           num_tokens_,
                           /*return_capture_params=*/false,
                           /*skip_token_update=*/true);
  replay_inputs_prepared_.store(true, std::memory_order_release);
}

bool AclGraph::prepare_static_mtp_graph_tasks(
    const SpecVerifyGraphTaskSignal& signal,
    const c10_npu::NPUStream& signal_stream) {
  if (static_graph_task_signature_ !=
      make_static_graph_task_signature(signal)) {
    return false;
  }
  signal_static_graph_tasks(signal_stream);
  return true;
}

AclGraphExecutorImpl::AclGraphExecutorImpl(CausalLM* model,
                                           const ModelArgs& args,
                                           const torch::Device& device,
                                           const runtime::Options& options)
    : model_(model), args_(args), device_(device), options_(options) {
  const bool need_update_attn_mask = model->is_hybrid_linear_attention();
  const bool is_hybrid_linear_attn = model->is_hybrid_linear_attention();
  graph_slot_count_ =
      ::xllm::ExecutionConfig::get_instance().enable_graph_double_buffer() ? 2
                                                                           : 1;
  for (int32_t slot_idx = 0; slot_idx < graph_slot_count_; ++slot_idx) {
    GraphSlot& slot = graph_slots_[slot_idx];
    slot.persistent_param = std::make_unique<GraphPersistentParam>(
        args_,
        device_,
        options_,
        need_update_attn_mask,
        is_hybrid_linear_attn,
        model_->supports_mla_graph_kv_bucketing());
    slot.graph_pool = c10_npu::graph_pool_handle();
  }
}

size_t AclGraphExecutorImpl::get_graph_count() const {
  size_t graph_count = 0;
  for (int32_t slot_idx = 0; slot_idx < graph_slot_count_; ++slot_idx) {
    graph_count += graph_slots_[slot_idx].graphs.size();
  }
  return graph_count;
}

size_t AclGraphExecutorImpl::prepared_graph_count_for_test(
    int32_t slot_id) const {
  CHECK_GE(slot_id, 0);
  CHECK_LT(slot_id, graph_slot_count_);
  return graph_slots_[slot_id].prepared_graphs.size();
}

size_t AclGraphExecutorImpl::get_graph_memory_pool_count() {
  std::vector<c10_npu::MempoolId_t> memory_pools;
  memory_pools.reserve(graph_slot_count_);
  for (int32_t slot_idx = 0; slot_idx < graph_slot_count_; ++slot_idx) {
    for (auto& [graph_key, graph] : graph_slots_[slot_idx].graphs) {
      (void)graph_key;
      const c10_npu::MempoolId_t memory_pool = graph->memory_pool();
      if (std::find(memory_pools.begin(), memory_pools.end(), memory_pool) ==
          memory_pools.end()) {
        memory_pools.emplace_back(memory_pool);
      }
    }
  }
  return memory_pools.size();
}

size_t AclGraphExecutorImpl::get_graph_capture_stream_count() const {
  std::vector<int64_t> stream_ids;
  stream_ids.reserve(graph_slot_count_);
  for (int32_t slot_idx = 0; slot_idx < graph_slot_count_; ++slot_idx) {
    for (const auto& [graph_key, graph] : graph_slots_[slot_idx].graphs) {
      (void)graph_key;
      const int64_t stream_id = graph->capture_stream_id();
      if (std::find(stream_ids.begin(), stream_ids.end(), stream_id) ==
          stream_ids.end()) {
        stream_ids.emplace_back(stream_id);
      }
    }
  }
  return stream_ids.size();
}

ForwardInput AclGraphExecutorImpl::prepare_inputs(Batch& batch) {
  // Prepare inputs for workers
  return batch.prepare_forward_input(
      options_.num_decoding_tokens(), 0, args_, options_.cp_size());
}

void AclGraphExecutorImpl::prepare_prepared_graph_input(
    int32_t slot_id,
    ForwardInput& input,
    std::vector<KVCache>& kv_caches) {
  CHECK_GE(slot_id, 0);
  CHECK_LT(slot_id, graph_slot_count_)
      << "Prepared logical Slot does not have a matching ACL Graph Slot";
  if (!input.input_params.meta.batch_forward_type.is_decode() ||
      args_.n_layers() == 1 || model_->requires_graph_forward_metadata() ||
      model_->is_hybrid_linear_attention()) {
    return;
  }
  auto [k_cache, v_cache] = find_attention_plan_kv_cache(kv_caches);
  torch::Tensor host_tiling =
      graph_slots_[slot_id]
          .persistent_param->prepare_paged_attention_tiling_host(
              input.token_ids,
              k_cache,
              v_cache,
              input.input_params.attention.device.block_tables,
              input.input_params);
  if (host_tiling.defined()) {
    input.input_params.graph.tiling_data = std::move(host_tiling);
  }
}

bool AclGraphExecutorImpl::supports_prepared_graph(
    const ForwardInput& input) const {
  const ModelInputParams& params = input.input_params;
  const bool is_decode = params.meta.batch_forward_type.is_decode();
  const bool is_spec_verify =
      params.is_spec_verify &&
      params.meta.batch_forward_type.is_chunked_prefill();
  if ((!is_decode && !is_spec_verify) ||
      (params.is_spec_verify && !is_spec_verify) || args_.n_layers() == 1 ||
      options_.dp_size() != 1 || options_.cp_size() != 1 ||
      options_.num_decoding_tokens() <= 0 ||
      (params.graph.input_tokens_override.defined() &&
       (!is_spec_verify ||
        !params.graph.input_tokens_override.is_same(input.token_ids))) ||
      params.meta.kv_max_seq_len > args_.max_position_embeddings() ||
      !input.device_input_buffer.defined() ||
      input.prepared_input_layout_signature == 0 ||
      !prepared_graph_inputs_are_arena_backed(input)) {
    return false;
  }
  if (is_spec_verify &&
      params.graph.use_expanded_decode_for_spec_verify_attention) {
    const std::optional<int64_t> max_kv_with_headroom =
        prepared_spec_verify_max_kv_with_headroom(params,
                                                  options_.block_size());
    if (!max_kv_with_headroom.has_value() ||
        max_kv_with_headroom.value() > args_.max_position_embeddings()) {
      return false;
    }
  }
  if (is_spec_verify &&
      (!kPreparedSpecVerifyGraphE2EValidated ||
       !prepared_spec_verify_graph_contract_is_complete(input))) {
    return false;
  }
  if (model_->is_hybrid_linear_attention() &&
      (!is_spec_verify ||
       !params.graph.use_expanded_decode_for_spec_verify_attention ||
       !make_static_graph_task_signature(params).has_value())) {
    return false;
  }
  if (params.dsa_device_geometry_authoritative &&
      (!kPreparedDeviceDsaGraphE2EValidated ||
       !prepared_external_graph_input_identity(input).has_value())) {
    return false;
  }

  const uint32_t num_tokens =
      static_cast<uint32_t>(input.token_ids.size(/*dim=*/0));
  if (num_tokens == 0) {
    return false;
  }
  const uint32_t invocation_width =
      static_cast<uint32_t>(is_spec_verify ? params.meta.q_max_seq_len
                                           : options_.num_decoding_tokens());
  if (invocation_width == 0 || num_tokens % invocation_width != 0) {
    return false;
  }
  const uint32_t batch_size = num_tokens / invocation_width;
  const uint32_t batch_size_limit = static_cast<uint32_t>(
      std::max<int32_t>(1,
                        ::xllm::ExecutionConfig::get_instance()
                            .acl_graph_decode_batch_size_limit()));
  return batch_size > 0 && batch_size <= batch_size_limit;
}

uint64_t AclGraphExecutorImpl::get_prepared_graph_key(
    const ForwardInput& input,
    uint64_t attention_plan_class) const {
  const uint32_t num_tokens =
      static_cast<uint32_t>(input.token_ids.size(/*dim=*/0));
  uint64_t graph_key = mix_graph_key(
      kPreparedGraphKeySeed,
      get_graph_key(num_tokens, input.input_params, attention_plan_class));
  graph_key = mix_graph_key(graph_key, input.prepared_input_layout_signature);
  graph_key = mix_graph_key(
      graph_key, static_cast<uint64_t>(input.input_params.meta.num_sequences));
  const std::optional<PreparedExternalGraphInputIdentity> external_identity =
      prepared_external_graph_input_identity(input);
  CHECK(external_identity.has_value())
      << "Prepared ACL graph external input contract is incomplete";
  graph_key = mix_graph_key(graph_key, external_identity->layout_signature);
  return graph_key;
}

PreparedSlotBinding AclGraphExecutorImpl::bind_prepared(
    int32_t slot_id,
    const ForwardInput& input,
    std::vector<KVCache>& kv_caches) {
  CHECK_GE(slot_id, 0);
  CHECK_LT(slot_id, graph_slot_count_)
      << "Prepared logical Slot does not have a matching ACL Graph Slot";

  PreparedSlotBinding binding;
  binding.slot_id = slot_id;
  if (!supports_prepared_graph(input)) {
    return binding;
  }

  uint64_t attention_plan_class = 0;
  std::optional<PagedAttentionPlanDescriptor> attention_plan;
  const bool needs_attention_plan =
      input.input_params.graph.use_expanded_decode_for_spec_verify_attention;
  if (needs_attention_plan) {
    const uint32_t num_tokens =
        static_cast<uint32_t>(input.token_ids.size(/*dim=*/0));
    const uint64_t lookup_key = spec_verify_attention_plan_lookup_key(
        num_tokens, input.input_params, options_.block_size());
    std::optional<uint64_t> cached_plan_class =
        find_spec_verify_attention_plan_class(lookup_key);
    if (cached_plan_class.has_value()) {
      attention_plan_class = cached_plan_class.value();
    } else {
      if (!input.input_params.meta.is_graph_warmup) {
        COUNTER_INC(prepared_task_graph_runtime_misses_total);
        return binding;
      }
      auto [k_cache, v_cache] = find_attention_plan_kv_cache(kv_caches);
      std::optional<PagedAttentionPlanDescriptor> descriptor =
          graph_slots_[slot_id]
              .persistent_param->classify_spec_verify_paged_attention_plan(
                  input.token_ids, k_cache, v_cache, input.input_params);
      if (!descriptor.has_value()) {
        return binding;
      }
      std::lock_guard<std::mutex> lock(graph_slots_mutex_);
      auto descriptor_it =
          std::find(spec_verify_attention_plan_descriptors_.begin(),
                    spec_verify_attention_plan_descriptors_.end(),
                    descriptor.value());
      if (descriptor_it == spec_verify_attention_plan_descriptors_.end()) {
        spec_verify_attention_plan_descriptors_.emplace_back(
            std::move(descriptor.value()));
        attention_plan_class = spec_verify_attention_plan_descriptors_.size();
      } else {
        attention_plan_class =
            static_cast<uint64_t>(
                std::distance(spec_verify_attention_plan_descriptors_.begin(),
                              descriptor_it)) +
            1;
      }
      auto [plan_it, inserted] = spec_verify_attention_plan_classes_.emplace(
          lookup_key, attention_plan_class);
      CHECK(inserted || plan_it->second == attention_plan_class)
          << "Prepared paged-attention plan class changed for one KV bucket";
      attention_plan_class = plan_it->second;
    }
    {
      std::lock_guard<std::mutex> lock(graph_slots_mutex_);
      CHECK_GT(attention_plan_class, 0);
      CHECK_LE(attention_plan_class,
               spec_verify_attention_plan_descriptors_.size());
      attention_plan =
          spec_verify_attention_plan_descriptors_[static_cast<size_t>(
              attention_plan_class - 1)];
    }
  }

  binding.graph_key = get_prepared_graph_key(input, attention_plan_class);
  std::shared_ptr<AclGraph> graph;
  {
    std::lock_guard<std::mutex> lock(graph_slots_mutex_);
    GraphSlot& slot = graph_slots_[slot_id];
    auto graph_it = slot.prepared_graphs.find(binding.graph_key);
    if (graph_it != slot.prepared_graphs.end()) {
      if (!graph_it->second->matches_prepared_input(input)) {
        COUNTER_INC(prepared_task_graph_address_mismatches_total);
        LOG_FIRST_N(ERROR, 1)
            << "Prepared ACL graph input arena address changed; using "
               "Prepared Eager for this invocation";
        return binding;
      }
      binding.mode = PreparedInvocationMode::GRAPH_REPLAY;
      graph = graph_it->second;
    }
  }
  if (graph != nullptr) {
    if (model_->requires_graph_forward_metadata() ||
        model_->is_hybrid_linear_attention() || needs_attention_plan) {
      graph->prepare_prepared_model_graph_metadata(
          model_,
          input,
          attention_plan.has_value() ? &attention_plan.value() : nullptr,
          options_.block_size());
    }
    if (model_->is_hybrid_linear_attention()) {
      binding.static_graph_tasks_prepared =
          graph->prepare_static_prepared_graph_tasks(
              input.input_params,
              c10_npu::getCurrentNPUStream(device_.index()));
      CHECK(binding.static_graph_tasks_prepared)
          << "Prepared hybrid Graph key has no matching static task signature";
    }
    binding.native_handle = graph;
    return binding;
  }

  if (input.input_params.meta.is_graph_warmup) {
    binding.mode = PreparedInvocationMode::GRAPH_CAPTURE;
    if (model_->requires_graph_forward_metadata() ||
        model_->is_hybrid_linear_attention() || needs_attention_plan) {
      GraphSlot& slot = graph_slots_[slot_id];
      std::optional<c10_npu::NPUStream> capture_stream;
      {
        std::lock_guard<std::mutex> lock(graph_slots_mutex_);
        if (!slot.graph_capture_stream.has_value()) {
          slot.graph_capture_stream = c10_npu::getStreamFromPool(
              /*isHighPriority=*/true, device_.index());
        }
        capture_stream = slot.graph_capture_stream;
      }
      graph = std::make_shared<AclGraph>(
          *slot.persistent_param, device_.index(), capture_stream.value());
      graph->prepare_prepared_model_graph_metadata(
          model_,
          input,
          attention_plan.has_value() ? &attention_plan.value() : nullptr,
          options_.block_size());
      binding.native_handle = graph;
    }
  } else {
    COUNTER_INC(prepared_task_graph_runtime_misses_total);
    LOG_FIRST_N(WARNING, 1)
        << "Prepared ACL graph miss; using Prepared Eager without runtime "
           "capture";
  }
  return binding;
}

ModelOutput AclGraphExecutorImpl::launch_prepared(
    const PreparedSlotBinding& binding,
    const ForwardInput& input,
    std::vector<KVCache>& kv_caches) {
  CHECK_GE(binding.slot_id, 0);
  CHECK_LT(binding.slot_id, graph_slot_count_);
  if (binding.mode == PreparedInvocationMode::EAGER) {
    COUNTER_INC(prepared_task_execution_total_eager);
    ModelInputParams eager_params = input.input_params;
    eager_params.enable_graph = false;
    eager_params.attn_metadata.reset();
    COUNTER_INC(num_model_execution_total_eager);
    ModelOutput output = forward_eager(
        model_, input.token_ids, input.positions, kv_caches, eager_params);
    if (model_->is_hybrid_linear_attention()) {
      trace_prepared_speculative_graph_execution(binding, input);
    }
    return output;
  }

  if (binding.mode == PreparedInvocationMode::GRAPH_REPLAY) {
    COUNTER_INC(prepared_task_execution_total_graph_replay);
    CHECK(binding.native_handle != nullptr);
    std::shared_ptr<AclGraph> graph =
        std::static_pointer_cast<AclGraph>(binding.native_handle);
    CHECK(graph->matches_prepared_input(input));
    ModelOutput output =
        graph->replay_prepared(input, binding.static_graph_tasks_prepared);
    if (model_->is_hybrid_linear_attention()) {
      trace_prepared_speculative_graph_execution(binding, input);
    }
    return output;
  }

  CHECK(binding.mode == PreparedInvocationMode::GRAPH_CAPTURE);
  COUNTER_INC(prepared_task_execution_total_graph_capture);
  CHECK(input.input_params.meta.is_graph_warmup)
      << "Prepared ACL graph capture is only allowed during startup warmup";
  GraphSlot& slot = graph_slots_[binding.slot_id];
  std::optional<c10_npu::NPUStream> capture_stream;
  {
    std::lock_guard<std::mutex> lock(graph_slots_mutex_);
    CHECK(slot.prepared_graphs.find(binding.graph_key) ==
          slot.prepared_graphs.end());
    if (!slot.graph_capture_stream.has_value()) {
      slot.graph_capture_stream =
          c10_npu::getStreamFromPool(/*isHighPriority=*/true, device_.index());
    }
    capture_stream = slot.graph_capture_stream;
  }
  std::shared_ptr<AclGraph> graph;
  if (binding.native_handle != nullptr) {
    graph = std::static_pointer_cast<AclGraph>(binding.native_handle);
  } else {
    graph = std::make_shared<AclGraph>(
        *slot.persistent_param, device_.index(), capture_stream.value());
  }
  ModelOutput output = graph->capture_prepared(
      model_, options_, input, kv_caches, slot.graph_pool);
  {
    std::lock_guard<std::mutex> lock(graph_slots_mutex_);
    auto [graph_it, inserted] =
        slot.prepared_graphs.emplace(binding.graph_key, graph);
    CHECK(inserted);
    (void)graph_it;
  }
  LOG(INFO) << "Captured startup Prepared ACL graph: slot=" << binding.slot_id
            << ", key=" << binding.graph_key
            << ", num_tokens=" << input.token_ids.size(/*dim=*/0);
  if (model_->is_hybrid_linear_attention()) {
    trace_prepared_speculative_graph_execution(binding, input);
  }
  return output;
}

// Main execution method with graph optimization for decode phase
// tokens: [num_decode_tokens]
// positions: [num_decode_tokens] token pos in the sequence
// returns: [num_decode_tokens, hidden_size]
ModelOutput AclGraphExecutorImpl::run(const torch::Tensor& tokens,
                                      const torch::Tensor& positions,
                                      std::vector<KVCache>& kv_caches,
                                      const ModelInputParams& params) {
  auto run_eager = [&]() {
    ModelInputParams eager_params = params;
    eager_params.enable_graph = false;
    eager_params.attn_metadata.reset();
    return forward_eager(model_, tokens, positions, kv_caches, eager_params);
  };
  // no mirco batch in decode phase
  const torch::Tensor& tokens_tensor = tokens;
  const torch::Tensor& positions_tensor = positions;
  const ModelInputParams& params_single = params;
  const bool in_decoding_phase =
      params_single.meta.batch_forward_type.is_decode();
  const bool in_spec_verify_phase =
      params_single.is_spec_verify &&
      params_single.meta.batch_forward_type.is_chunked_prefill();
  VLOG(50) << "in_decoding_phase: " << in_decoding_phase
           << " in_spec_verify_phase: " << in_spec_verify_phase
           << " q_max_seq_len: " << params_single.meta.q_max_seq_len
           << " n_layers: " << args_.n_layers();
  if ((!in_decoding_phase && !in_spec_verify_phase) || args_.n_layers() == 1) {
    VLOG(kGraphExecutorLogVerboseLevel)
        << "AclGraphExecutorImpl::run() in eager mode";
    COUNTER_INC(num_model_execution_total_eager);
    return run_eager();
  }
  if (in_spec_verify_phase && !model_->is_hybrid_linear_attention()) {
    LOG_FIRST_N(WARNING, 1)
        << "Falling back to eager mode for spec verify because the "
           "chunked-prefill validate graph path is currently only adapted for "
           "hybrid linear attention models.";
    COUNTER_INC(num_model_execution_total_eager);
    return run_eager();
  }
  // CP shards the query rows of a prefill batch and gathers them per layer, so
  // token counts and collectives differ from the captured decode shape. Decode
  // itself runs with CP inactive (both CP paths return early on decode), which
  // is why graph mode and CP can coexist -- but spec-verify chunked prefill is
  // a non-decode batch that reaches capture, so it must stay eager under CP.
  if (in_spec_verify_phase && options_.cp_size() > 1) {
    LOG_FIRST_N(WARNING, 1)
        << "Falling back to eager mode for spec verify because context "
           "parallel (cp_size="
        << options_.cp_size()
        << ") shards prefill rows, which the captured graph shape does not "
           "describe.";
    COUNTER_INC(num_model_execution_total_eager);
    return run_eager();
  }
  if (in_decoding_phase &&
      params_single.parallel.dp_global_token_nums.size() > 1) {
    if (params_single.parallel.dp_is_decode.size() !=
        params_single.parallel.dp_global_token_nums.size()) {
      LOG_FIRST_N(WARNING, 1)
          << "Falling back to eager mode because dp_is_decode size ("
          << params_single.parallel.dp_is_decode.size()
          << ") does not match dp_global_token_nums size ("
          << params_single.parallel.dp_global_token_nums.size()
          << "); ACL graph decode requires valid DP forward metadata. "
          << "dp_global_token_nums="
          << params_single.parallel.dp_global_token_nums
          << ", dp_is_decode=" << params_single.parallel.dp_is_decode;
      COUNTER_INC(num_model_execution_total_eager);
      return run_eager();
    }

    if (std::find(params_single.parallel.dp_is_decode.begin(),
                  params_single.parallel.dp_is_decode.end(),
                  0) != params_single.parallel.dp_is_decode.end()) {
      LOG_FIRST_N(WARNING, 1)
          << "Falling back to eager mode because not all DP ranks are in "
             "decode phase; ACL graph decode requires all DP ranks to be "
             "decode to avoid using prefill or chunked-prefill token counts "
             "as graph bucket size. dp_global_token_nums="
          << params_single.parallel.dp_global_token_nums
          << ", dp_is_decode=" << params_single.parallel.dp_is_decode;
      COUNTER_INC(num_model_execution_total_eager);
      return run_eager();
    }
  }

  // Only use acl graph in decode phase for performance optimization
  // For DP, all ranks use the largest rank-local token count as the graph key;
  // a local shard can be empty on some ranks.
  uint32_t max_local_num_tokens = tokens_tensor.size(/*dim=*/0);
  if (params_single.parallel.dp_global_token_nums.size() > 1) {
    max_local_num_tokens =
        util::max(params_single.parallel.dp_global_token_nums);
  }
  // Keep actual n_tokens for replay output slicing.
  const uint32_t n_tokens = tokens_tensor.size(/*dim=*/0);
  const uint32_t local_batch_size = n_tokens / options_.num_decoding_tokens();
  const uint32_t max_local_batch_size =
      max_local_num_tokens / options_.num_decoding_tokens();

  // Large decode batches create too many/too large ACL graphs and may OOM.
  // Fall back to eager mode when batch size exceeds the safety threshold.
  // Use max_local_batch_size so all DP ranks make the same decision and stay
  // in sync on HCCL collectives.
  const uint32_t decode_batch_size_limit = static_cast<uint32_t>(
      std::max<int32_t>(1,
                        ::xllm::ExecutionConfig::get_instance()
                            .acl_graph_decode_batch_size_limit()));
  if (max_local_batch_size > decode_batch_size_limit) {
    LOG_FIRST_N(WARNING, 1)
        << "Falling back to eager mode because decode batch_size (global="
        << max_local_batch_size << ", local=" << local_batch_size << ") > "
        << decode_batch_size_limit
        << "; ACL graph is disabled for this request size to avoid OOM. "
        << "This message is logged only once. "
        << "Monitor counter 'num_model_execution_total_eager' for frequency.";
    COUNTER_INC(num_model_execution_total_eager);
    return run_eager();
  }

  const uint32_t bucket_num_tokens =
      get_bucket_num_tokens(max_local_num_tokens);

  // Check if conditions are suitable for graph execution (replay or capture)
  const auto max_seq_len = args_.max_position_embeddings();
  const bool seq_len_supported =
      params_single.meta.kv_max_seq_len <= max_seq_len;

  // Combined condition for graph capture support
  // ACL graph executor only supports single tensor inputs (no micro-batching)
  const bool capture_supported = seq_len_supported;

  // Early return if conditions are not suitable for graph operations
  if (!capture_supported) {
    LOG_FIRST_N(WARNING, 1)
        << "Falling back to eager mode because kv_max_seq_len ("
        << params_single.meta.kv_max_seq_len << ") > max_seq_len ("
        << max_seq_len << "). This message is logged only once. "
        << "Monitor counter 'num_model_execution_total_eager' for frequency.";
    COUNTER_INC(num_model_execution_total_eager);
    return run_eager();
  }

  int32_t slot_idx = 0;
  {
    std::lock_guard<std::mutex> lock(graph_slots_mutex_);
    slot_idx = next_replay_slot_;
    next_replay_slot_ = (next_replay_slot_ + 1) % graph_slot_count_;
    last_started_replay_slot_ = slot_idx;
    auto& slot = graph_slots_[slot_idx];
    slot.is_prepared = false;
  }
  auto& active_slot = graph_slots_[slot_idx];
  auto& active_persistent_param = *active_slot.persistent_param;

  uint64_t attention_plan_class = 0;
  const bool needs_attention_plan_class =
      params_single.is_spec_verify &&
      params_single.meta.batch_forward_type.is_chunked_prefill() &&
      params_single.graph.spec_verify_source_addresses_stable;
  if (needs_attention_plan_class) {
    const uint64_t lookup_key = spec_verify_attention_plan_lookup_key(
        bucket_num_tokens, params_single, options_.block_size());
    if (auto plan_class = find_spec_verify_attention_plan_class(lookup_key)) {
      attention_plan_class = plan_class.value();
    }
    if (attention_plan_class == 0) {
      // Cold path for a previously unseen KV block bucket. Run ATB Setup to
      // classify its immutable tiling plan, then reuse any graph already
      // captured for that plan class.
      auto [k_cache, v_cache] = find_attention_plan_kv_cache(kv_caches);
      auto descriptor =
          active_persistent_param.classify_spec_verify_paged_attention_plan(
              tokens_tensor, k_cache, v_cache, params_single);
      if (!descriptor.has_value()) {
        LOG_FIRST_N(ERROR, 1)
            << "Falling back to eager speculative verification because the "
               "paged-attention tiling layout cannot be classified safely.";
        COUNTER_INC(num_model_execution_total_eager);
        return forward_eager(
            model_, tokens, positions, kv_caches, params_single);
      }
      std::lock_guard<std::mutex> lock(graph_slots_mutex_);
      auto descriptor_it =
          std::find(spec_verify_attention_plan_descriptors_.begin(),
                    spec_verify_attention_plan_descriptors_.end(),
                    descriptor.value());
      if (descriptor_it == spec_verify_attention_plan_descriptors_.end()) {
        spec_verify_attention_plan_descriptors_.push_back(
            std::move(descriptor.value()));
        attention_plan_class = spec_verify_attention_plan_descriptors_.size();
      } else {
        attention_plan_class =
            static_cast<uint64_t>(
                std::distance(spec_verify_attention_plan_descriptors_.begin(),
                              descriptor_it)) +
            1;
      }
      auto [it, inserted] = spec_verify_attention_plan_classes_.emplace(
          lookup_key, attention_plan_class);
      CHECK(inserted || it->second == attention_plan_class)
          << "paged-attention plan class changed for one KV block bucket";
      attention_plan_class = it->second;
    }
  }

  const uint64_t graph_key =
      get_graph_key(bucket_num_tokens, params_single, attention_plan_class);
  std::shared_ptr<AclGraph> replay_graph;
  {
    std::lock_guard<std::mutex> lock(graph_slots_mutex_);
    auto it = active_slot.graphs.find(graph_key);
    if (it != active_slot.graphs.end()) {
      replay_graph = it->second;
    }
  }

  if (replay_graph != nullptr) {
    // Replay the existing graph
    VLOG(kGraphExecutorLogVerboseLevel)
        << "AclGraphExecutorImpl::run() in replay mode";
    ModelOutput result = replay_graph->replay(
        model_, tokens_tensor, positions_tensor, kv_caches, params_single);
    // Handle aux_hidden_states based on options
    if (options_.enable_graph_aux_hidden_states()) {
      torch::Tensor aux_hidden_states =
          active_persistent_param.aux_hidden_states(n_tokens);
      if (aux_hidden_states.defined() && aux_hidden_states.numel() > 0) {
        return ModelOutput(
            result.hidden_states, torch::Tensor(), aux_hidden_states);
      }
    }
    return result;
  }

  if (in_decoding_phase && ::xllm::ExecutionConfig::get_instance()
                               .enable_graph_mode_decode_no_padding()) {
    const uint32_t max_graph_batch_size =
        std::min(static_cast<uint32_t>(
                     std::max<int32_t>(1, options_.max_seqs_per_batch())),
                 decode_batch_size_limit);
    const uint32_t dp_size =
        static_cast<uint32_t>(std::max<int32_t>(1, options_.dp_size()));
    if (!is_acl_graph_decode_capture_allowed(
            max_local_batch_size,
            max_graph_batch_size,
            dp_size,
            params_single.meta.is_graph_warmup)) {
      LOG_FIRST_N(WARNING, 1)
          << "Falling back to eager mode because no ACL graph was prewarmed "
             "for no-padding decode batch_size="
          << max_local_batch_size << " (local=" << local_batch_size
          << ", max_graph_batch_size=" << max_graph_batch_size
          << ", dp_size=" << dp_size
          << "). Runtime capture is limited to graph warmup buckets to "
             "prevent unbounded graph memory growth. This message is logged "
             "only once. Monitor counter 'num_model_execution_total_eager' "
             "for frequency.";
      COUNTER_INC(num_model_execution_total_eager);
      return run_eager();
    }
  }

  // Graph doesn't exist for this bucket num_tokens, try to create it lazily
  if (!active_slot.graph_capture_stream.has_value()) {
    active_slot.graph_capture_stream =
        c10_npu::getStreamFromPool(/*isHighPriority=*/true, device_.index());
  }
  auto graph =
      std::make_shared<AclGraph>(active_persistent_param,
                                 device_.index(),
                                 active_slot.graph_capture_stream.value());
  VLOG(kGraphExecutorLogVerboseLevel)
      << "AclGraphExecutorImpl::run() in capture mode";
  const bool capture_success = graph->capture(model_,
                                              options_,
                                              tokens_tensor,
                                              positions_tensor,
                                              params_single,
                                              kv_caches,
                                              bucket_num_tokens,
                                              active_slot.graph_pool);

  CHECK(capture_success)
      << "Failed to capture ACL graph for bucket num_tokens: "
      << bucket_num_tokens;
  LOG(INFO) << "Lazy capturing ACL graph for bucket num_tokens: "
            << bucket_num_tokens << " (actual num_tokens: " << n_tokens
            << ") done";

  const bool static_mtp_variant = uses_static_mtp_graph_task_variant(
      params_single, bucket_num_tokens, options_.block_size());
  {
    std::lock_guard<std::mutex> lock(graph_slots_mutex_);
    if (static_mtp_variant) {
      while (active_slot.static_mtp_graph_keys.size() >=
             kMaxStaticMtpGraphVariantsPerSlot) {
        const uint64_t evicted_key = active_slot.static_mtp_graph_keys.front();
        active_slot.static_mtp_graph_keys.pop_front();
        active_slot.graphs.erase(evicted_key);
      }
      active_slot.static_mtp_graph_keys.push_back(graph_key);
    }
    // shared_ptr keeps a replay/prepare that already left the map alive if a
    // later capture evicts this static variant.
    active_slot.graphs[graph_key] = graph;
  }

  // Return the output from capture (no need to replay since capture
  // already executed)
  torch::Tensor hidden_states = graph->get_hidden_states(n_tokens);
  if (options_.enable_graph_aux_hidden_states()) {
    torch::Tensor aux_hidden_states =
        active_persistent_param.aux_hidden_states(n_tokens);
    if (aux_hidden_states.defined() && aux_hidden_states.numel() > 0) {
      return ModelOutput(hidden_states, torch::Tensor(), aux_hidden_states);
    }
  }
  return ModelOutput(hidden_states);
}

void AclGraphExecutorImpl::prepare_graph_input(const torch::Tensor& tokens,
                                               const torch::Tensor& positions,
                                               std::vector<KVCache>& kv_caches,
                                               const ModelInputParams& params) {
  const bool in_decoding_phase = params.meta.batch_forward_type.is_decode();
  const bool in_spec_verify_phase =
      params.is_spec_verify &&
      params.meta.batch_forward_type.is_chunked_prefill();
  if ((!in_decoding_phase && !in_spec_verify_phase) || args_.n_layers() == 1) {
    return;
  }
  if (model_->requires_graph_forward_metadata()) {
    return;
  }
  if (in_spec_verify_phase && !model_->is_hybrid_linear_attention()) {
    return;
  }
  if (in_decoding_phase && params.parallel.dp_global_token_nums.size() > 1) {
    if (params.parallel.dp_is_decode.size() !=
        params.parallel.dp_global_token_nums.size()) {
      return;
    }
    if (std::find(params.parallel.dp_is_decode.begin(),
                  params.parallel.dp_is_decode.end(),
                  0) != params.parallel.dp_is_decode.end()) {
      return;
    }
  }
  if (params.meta.kv_max_seq_len > args_.max_position_embeddings()) {
    return;
  }
  if (graph_slot_count_ <= 1) {
    return;
  }

  uint32_t graph_num_tokens = tokens.size(/*dim=*/0);
  if (params.parallel.dp_global_token_nums.size() > 1) {
    graph_num_tokens = util::max(params.parallel.dp_global_token_nums);
  }
  if (graph_num_tokens == 0) {
    return;
  }
  const uint32_t bucket_num_tokens = get_bucket_num_tokens(graph_num_tokens);
  uint64_t attention_plan_class = 0;
  if (params.is_spec_verify &&
      params.meta.batch_forward_type.is_chunked_prefill() &&
      params.graph.spec_verify_source_addresses_stable) {
    const uint64_t lookup_key = spec_verify_attention_plan_lookup_key(
        bucket_num_tokens, params, options_.block_size());
    auto plan_class = find_spec_verify_attention_plan_class(lookup_key);
    if (!plan_class.has_value()) {
      return;
    }
    attention_plan_class = plan_class.value();
  }
  const uint64_t graph_key =
      get_graph_key(bucket_num_tokens, params, attention_plan_class);

  std::shared_ptr<AclGraph> graph;
  {
    std::lock_guard<std::mutex> lock(graph_slots_mutex_);
    if (last_started_replay_slot_ < 0) {
      return;
    }
    const int32_t prepare_slot =
        (last_started_replay_slot_ + 1) % graph_slot_count_;
    auto& slot = graph_slots_[prepare_slot];
    if (slot.is_prepared) {
      return;
    }
    auto it = slot.graphs.find(graph_key);
    if (it == slot.graphs.end()) {
      return;
    }
    graph = it->second;
    slot.is_prepared = true;
  }
  graph->prepare_replay_inputs(tokens, positions, kv_caches, params);
}

bool AclGraphExecutorImpl::prepare_static_mtp_graph_tasks(
    const SpecVerifyGraphTaskSignal& signal,
    const Stream& signal_stream) {
  if (!model_->is_hybrid_linear_attention() || graph_slot_count_ != 1 ||
      !kernel::npu::tilelang::has_spec_verify_graph_update_specialization(
          signal.spec_width, options_.block_size()) ||
      signal.block_table_width < 1 ||
      signal.block_table_width > kSpecVerifyExpandedBlockMask ||
      signal.max_kv_seq_len < 1) {
    return false;
  }
  const uint64_t bucket_num_tokens = static_cast<uint64_t>(signal.spec_width);
  const uint64_t q_max_seq_len = static_cast<uint64_t>(signal.spec_width);
  const uint64_t width = static_cast<uint64_t>(signal.block_table_width);
  const uint64_t packed_key = spec_verify_packed_graph_key(
      static_cast<uint32_t>(bucket_num_tokens), q_max_seq_len, width, width);
  const uint64_t lookup_key =
      mix_graph_key(packed_key,
                    paged_attention_plan_bucket(signal.max_kv_seq_len,
                                                options_.block_size()));
  auto attention_plan_class = find_spec_verify_attention_plan_class(lookup_key);
  if (!attention_plan_class.has_value()) {
    return false;
  }
  const uint64_t base_key =
      mix_graph_key(packed_key, attention_plan_class.value());
  const uint64_t graph_key = static_mtp_graph_task_key(
      base_key, make_static_graph_task_signature(signal));
  std::shared_ptr<AclGraph> graph;
  {
    std::lock_guard<std::mutex> lock(graph_slots_mutex_);
    auto& graphs = graph_slots_[0].graphs;
    auto it = graphs.find(graph_key);
    if (it == graphs.end()) {
      return false;
    }
    graph = it->second;
  }
  return graph->prepare_static_mtp_graph_tasks(signal,
                                               *signal_stream.get_stream());
}

void AclGraph::print_graph_tensors() const {
  VLOG(kGraphExecutorLogVerboseLevel)
      << "graph persistent_tokens_: " << persistent_param_.persistent_tokens();
  VLOG(kGraphExecutorLogVerboseLevel)
      << "graph persistent_positions_: "
      << persistent_param_.persistent_positions();
  VLOG(kGraphExecutorLogVerboseLevel)
      << "graph persistent_new_cache_slots_: "
      << persistent_param_.persistent_new_cache_slots();
  VLOG(kGraphExecutorLogVerboseLevel)
      << "graph q_seq_lens_: " << persistent_param_.q_seq_lens();
  VLOG(kGraphExecutorLogVerboseLevel)
      << "graph kv_seq_lens_: " << persistent_param_.kv_seq_lens();
  VLOG(kGraphExecutorLogVerboseLevel)
      << "graph persistent_block_tables_: "
      << persistent_param_.persistent_block_tables();
  VLOG(kGraphExecutorLogVerboseLevel)
      << "graph hidden_states_: " << persistent_param_.hidden_states();
}

// bucket will be [1, 2, 4, 8, 16, 32, 48, 64, ..., max_seqs_per_batch]
uint32_t AclGraphExecutorImpl::get_bucket_num_tokens(
    uint32_t num_tokens) const {
  if (::xllm::ExecutionConfig::get_instance()
          .enable_graph_mode_decode_no_padding()) {
    return num_tokens;
  }
  if (num_tokens <= 1) {
    return 1;
  } else if (num_tokens <= 2) {
    return 2;
  } else if (num_tokens <= 4) {
    return 4;
  } else if (num_tokens <= 8) {
    return 8;
  } else {
    // For num_tokens > 8, use multiples of 16.
    return ((num_tokens + 15) / 16) * 16;
  }
}

std::optional<uint64_t>
AclGraphExecutorImpl::find_spec_verify_attention_plan_class(
    uint64_t lookup_key) {
  std::lock_guard<std::mutex> lock(graph_slots_mutex_);
  auto it = spec_verify_attention_plan_classes_.find(lookup_key);
  if (it == spec_verify_attention_plan_classes_.end()) {
    return std::nullopt;
  }
  return it->second;
}

uint64_t AclGraphExecutorImpl::get_graph_key(
    uint32_t bucket_num_tokens,
    const ModelInputParams& params,
    uint64_t attention_plan_class) const {
  if (params.is_spec_verify &&
      params.meta.batch_forward_type.is_chunked_prefill()) {
    const uint64_t q_max_seq_len =
        static_cast<uint64_t>(std::max<int32_t>(params.meta.q_max_seq_len, 1));
    if ((params.graph.spec_verify_source_addresses_stable ||
         params.graph.prepared_spec_verify_direct_bind) &&
        params.graph.use_expanded_decode_for_spec_verify_attention) {
      CHECK(params.attention.device.block_tables.defined());
      CHECK(params.graph.expanded_block_tables.defined());
      const uint64_t block_table_width =
          static_cast<uint64_t>(params.attention.device.block_tables.size(1));
      const uint64_t expanded_block_table_width =
          static_cast<uint64_t>(params.graph.expanded_block_tables.size(1));
      // Persistent graph inputs encode tensor view shapes. Specialize the MTP
      // target graph by both block-table widths so a synthetic warmup graph
      // cannot be replayed with a real request's wider table view.
      const uint64_t packed_key =
          spec_verify_packed_graph_key(bucket_num_tokens,
                                       q_max_seq_len,
                                       block_table_width,
                                       expanded_block_table_width);
      CHECK_NE(attention_plan_class, 0)
          << "stable speculative-verify graph requires an attention plan "
             "class";
      const uint64_t base_key = mix_graph_key(packed_key, attention_plan_class);
      if (uses_static_mtp_graph_task_variant(
              params, bucket_num_tokens, options_.block_size())) {
        const auto signature = make_static_graph_task_signature(params);
        CHECK(signature.has_value());
        return static_mtp_graph_task_key(base_key, signature.value());
      }
      return base_key;
    }
    return static_cast<uint64_t>(bucket_num_tokens) | kSpecVerifyGraphKeyMask |
           (q_max_seq_len << kSpecVerifyQMaxSeqLenShift);
  }
  if (model_->supports_mla_graph_kv_bucketing()) {
    const int32_t capture_kv_seq_len_bucket =
        get_mla_capture_kv_seq_len_bucket(params, options_);
    return get_mla_graph_key(bucket_num_tokens, capture_kv_seq_len_bucket);
  }
  return static_cast<uint64_t>(bucket_num_tokens);
}

}  // namespace xllm::npu
