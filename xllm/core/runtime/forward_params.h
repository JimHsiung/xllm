/* Copyright 2025-2026 The xLLM Authors.
Copyright 2024 The ScaleLLM Authors. All Rights Reserved.

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

#pragma once

#include <torch/torch.h>

#include <algorithm>
#include <cstring>
#include <memory>
#include <nlohmann/json.hpp>
#include <numeric>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "common/types.h"
#include "core/framework/multimodal/mm_batch_data.h"
#include "core/framework/multimodal/mm_data.h"
#include "framework/config/execution_config.h"
#include "framework/model/model_input_params.h"
#include "framework/sampling/beam_searcher.h"
#include "framework/sampling/json_object_grammar.h"
#include "framework/sampling/sampling_params.h"
#include "platform/device.h"
#include "platform/platform.h"
#include "runtime/dit_forward_params.h"
#include "runtime/json_object_output_rows.h"

namespace xllm {

struct ForwardInput;

namespace detail {

constexpr uint64_t kForwardInputBufferAlignment = 16;

inline uint64_t align_up(uint64_t value, uint64_t alignment) {
  if (alignment == 0) {
    return value;
  }
  return ((value + alignment - 1) / alignment) * alignment;
}

inline bool supports_contiguous_forward_input_buffer(
    const torch::Device& device) {
#if defined(USE_CUDA)
  return device.type() == torch::kCUDA;
#elif defined(USE_MLU) || defined(USE_MUSA)
  return device.type() == torch::kPrivateUse1;
#elif defined(USE_NPU)
  (void)device;
  return true;
#else
  (void)device;
  return false;
#endif
}

bool try_to_device_from_input_host_buffer(const ForwardInput& input,
                                          const torch::Device& device,
                                          torch::ScalarType dtype,
                                          ForwardInput& output);

bool unpack_from_input_host_buffer(const ForwardInput& input,
                                   const torch::Device& device,
                                   torch::ScalarType dtype,
                                   ForwardInput& output,
                                   bool materialize_device_buffer);

bool unpack_from_input_host_buffer(const ForwardInput& input,
                                   const torch::Device& device,
                                   ForwardInput& output);

struct ForwardInputBufferEntry {
  torch::Tensor host_tensor;
  torch::Tensor* target = nullptr;
  uint64_t offset = 0;
  uint64_t aligned_bytes = 0;
};

struct ForwardInputHostCopyStats {
  uint64_t bytes = 0;
  int32_t copies = 0;
};

struct ForwardInputBufferPlan {
  std::vector<ForwardInputBufferEntry> entries;

  bool add(const torch::Tensor& tensor, torch::Tensor* target) {
    return add_impl(tensor, target, /*allow_device_source=*/false);
  }

  // Device sources must be declared per field. This keeps newly added input
  // metadata Host-only by default instead of silently introducing a D2D copy
  // into PreparedInputArena staging.
  bool add_external_device_source(const torch::Tensor& tensor,
                                  torch::Tensor* target) {
    return add_impl(tensor, target, /*allow_device_source=*/true);
  }

  bool add_impl(const torch::Tensor& tensor,
                torch::Tensor* target,
                bool allow_device_source) {
    if (!tensor.defined()) {
      return true;
    }
    if (!tensor.device().is_cpu() && !allow_device_source) {
      return false;
    }
    entries.push_back({tensor.contiguous(), target, 0, 0});
    return true;
  }

  uint64_t prepare_layout() {
    uint64_t total = 0;
    for (auto& entry : entries) {
      total = align_up(total, kForwardInputBufferAlignment);
      entry.offset = total;
      const uint64_t bytes = static_cast<uint64_t>(
          entry.host_tensor.numel() * entry.host_tensor.element_size());
      entry.aligned_bytes = align_up(bytes, kForwardInputBufferAlignment);
      total += entry.aligned_bytes;
    }
    return total;
  }

  uint64_t layout_signature() const {
    constexpr uint64_t kFnvOffsetBasis = 1469598103934665603ULL;
    uint64_t signature = kFnvOffsetBasis;
    auto mix = [&signature](uint64_t value) {
      constexpr uint64_t kLocalFnvPrime = 1099511628211ULL;
      signature ^= value;
      signature *= kLocalFnvPrime;
    };
    mix(static_cast<uint64_t>(entries.size()));
    for (const ForwardInputBufferEntry& entry : entries) {
      mix(entry.offset);
      mix(entry.aligned_bytes);
      mix(static_cast<uint64_t>(entry.host_tensor.scalar_type()));
      mix(static_cast<uint64_t>(entry.host_tensor.dim()));
      for (int64_t dimension : entry.host_tensor.sizes()) {
        mix(static_cast<uint64_t>(dimension));
      }
    }
    return signature;
  }

  torch::Tensor build_host_buffer(uint64_t total_bytes) const {
    auto buffer = torch::empty({static_cast<int64_t>(total_bytes)},
                               torch::TensorOptions()
                                   .dtype(torch::kUInt8)
                                   .device(torch::kCPU)
                                   .pinned_memory(true));
    pack_host_buffer(buffer);
    return buffer;
  }

  void pack_host_buffer(const torch::Tensor& buffer) const {
    CHECK(buffer.defined());
    CHECK(buffer.device().is_cpu());
    CHECK_EQ(buffer.scalar_type(), torch::kUInt8);
    char* base = static_cast<char*>(buffer.data_ptr());
    for (const auto& entry : entries) {
      const uint64_t bytes = static_cast<uint64_t>(
          entry.host_tensor.numel() * entry.host_tensor.element_size());
      if (bytes == 0) {
        continue;
      }
      if (entry.host_tensor.device().is_cpu()) {
        std::memcpy(base + entry.offset, entry.host_tensor.data_ptr(), bytes);
      } else {
        std::memset(base + entry.offset, 0, static_cast<size_t>(bytes));
      }
      if (entry.aligned_bytes > bytes) {
        std::memset(base + entry.offset + bytes,
                    0,
                    static_cast<size_t>(entry.aligned_bytes - bytes));
      }
    }
  }

  ForwardInputHostCopyStats copy_host_sources(
      const torch::Tensor& host_buffer,
      const torch::Tensor& device_buffer) const {
    CHECK(host_buffer.defined());
    CHECK(device_buffer.defined());
    CHECK(host_buffer.device().is_cpu());
    CHECK_EQ(host_buffer.scalar_type(), torch::kUInt8);
    CHECK_EQ(device_buffer.scalar_type(), torch::kUInt8);
    CHECK_EQ(host_buffer.numel(), device_buffer.numel());

    ForwardInputHostCopyStats stats;
    uint64_t range_start = 0;
    uint64_t range_end = 0;
    bool has_range = false;
    auto flush_range = [&]() {
      if (!has_range || range_end <= range_start) {
        has_range = false;
        return;
      }
      const int64_t start = static_cast<int64_t>(range_start);
      const int64_t length = static_cast<int64_t>(range_end - range_start);
      device_buffer.narrow(/*dim=*/0, start, length)
          .copy_(host_buffer.narrow(/*dim=*/0, start, length),
                 /*non_blocking=*/true);
      stats.bytes += range_end - range_start;
      ++stats.copies;
      has_range = false;
    };

    for (const ForwardInputBufferEntry& entry : entries) {
      if (entry.aligned_bytes == 0) {
        continue;
      }
      if (!entry.host_tensor.device().is_cpu()) {
        flush_range();
        continue;
      }
      if (!has_range) {
        range_start = entry.offset;
        has_range = true;
      } else {
        CHECK_EQ(range_end, entry.offset)
            << "Host input ranges must remain contiguous";
      }
      range_end = entry.offset + entry.aligned_bytes;
    }
    flush_range();
    return stats;
  }

  void bind_device_views(const torch::Tensor& device_buffer,
                         const torch::Device& device) const {
    const char* base = static_cast<const char*>(device_buffer.data_ptr());
    for (const auto& entry : entries) {
      if (entry.target == nullptr || !entry.host_tensor.defined()) {
        continue;
      }
      const void* ptr = base + entry.offset;
#if defined(USE_CUDA) || defined(USE_DCU)
      if (device.type() == torch::kCUDA) {
        *entry.target = get_tensor_from_blob(entry.host_tensor.sizes().vec(),
                                             entry.host_tensor.scalar_type(),
                                             ptr,
                                             device_buffer);
        continue;
      }
#endif
#if defined(USE_MLU) || defined(USE_MUSA)
      if (device.type() == torch::kPrivateUse1) {
        *entry.target = get_tensor_from_blob(entry.host_tensor.sizes().vec(),
                                             entry.host_tensor.scalar_type(),
                                             ptr,
                                             device_buffer);
        continue;
      }
#endif
#if defined(USE_NPU)
      if (device.type() == torch::kPrivateUse1) {
        *entry.target = get_tensor_from_blob(entry.host_tensor.sizes().vec(),
                                             entry.host_tensor.scalar_type(),
                                             ptr);
        continue;
      }
#endif
      CHECK(device.is_cpu())
          << "Unsupported contiguous input buffer device: " << device;
      *entry.target =
          torch::from_blob(const_cast<void*>(ptr),
                           entry.host_tensor.sizes().vec(),
                           torch::TensorOptions()
                               .dtype(entry.host_tensor.scalar_type())
                               .device(torch::kCPU));
    }
  }

  void copy_device_sources() const {
    for (const ForwardInputBufferEntry& entry : entries) {
      if (entry.target == nullptr || !entry.host_tensor.defined() ||
          entry.host_tensor.device().is_cpu()) {
        continue;
      }
      CHECK(entry.target->defined());
      entry.target->copy_(entry.host_tensor, /*non_blocking=*/true);
    }
  }

  uint64_t device_source_bytes() const {
    uint64_t bytes = 0;
    for (const ForwardInputBufferEntry& entry : entries) {
      if (entry.host_tensor.defined() && !entry.host_tensor.device().is_cpu()) {
        bytes += static_cast<uint64_t>(entry.host_tensor.numel() *
                                       entry.host_tensor.element_size());
      }
    }
    return bytes;
  }

  int32_t device_source_count() const {
    int32_t count = 0;
    for (const ForwardInputBufferEntry& entry : entries) {
      if (entry.host_tensor.defined() && entry.host_tensor.numel() > 0 &&
          !entry.host_tensor.device().is_cpu()) {
        ++count;
      }
    }
    return count;
  }
};

inline bool add_sampling_to_plan(const SamplingParameters& source,
                                 SamplingParameters& target,
                                 ForwardInputBufferPlan& plan) {
  return plan.add(source.selected_token_idxes, &target.selected_token_idxes) &&
         plan.add(source.frequency_penalties, &target.frequency_penalties) &&
         plan.add(source.presence_penalties, &target.presence_penalties) &&
         plan.add(source.repetition_penalties, &target.repetition_penalties) &&
         plan.add(source.temperatures, &target.temperatures) &&
         plan.add(source.top_p, &target.top_p) &&
         plan.add(source.top_k, &target.top_k) &&
         plan.add(source.unique_token_ids, &target.unique_token_ids) &&
         plan.add(source.unique_token_counts, &target.unique_token_counts) &&
         plan.add(source.unique_token_ids_lens,
                  &target.unique_token_ids_lens) &&
         plan.add(source.sample_idxes, &target.sample_idxes) &&
         plan.add(source.do_sample, &target.do_sample) &&
         plan.add(source.filter_mask, &target.filter_mask) &&
         plan.add(source.filter_bitmask, &target.filter_bitmask) &&
         plan.add(source.acc_logprob, &target.acc_logprob);
}

inline torch::Tensor normalize_positions_for_device(
    const torch::Tensor& positions) {
  if ((Platform::is_cuda() || Platform::is_ilu() || Platform::is_musa()) &&
      positions.defined() && positions.scalar_type() != torch::kInt64) {
    return positions.to(torch::kInt64);
  }
  return positions;
}

inline bool has_contiguous_input_buffer_exclusions(
    const ModelInputParams& params) {
  return params.multimodal.mm_data.valid() || params.has_onerec_params() ||
         params.has_llmrec_params() || params.dit_forward_input.valid() ||
         !params.multimodal.deep_stacks.empty();
}

inline void clear_contiguous_input_buffer_tensor_targets(
    ModelInputParams& params) {
  params.attention.device.in_prefix_slots = torch::Tensor();
  params.embedding.input_embedding = torch::Tensor();
  params.embedding.linear_state_indices = torch::Tensor();
  params.embedding.mtp_bootstrap_embeddings = torch::Tensor();
  params.embedding.predecessor_rows = torch::Tensor();
  params.embedding.mtp_shifted_token_ids = torch::Tensor();
  params.block_copy.src_block_indices = torch::Tensor();
  params.block_copy.dst_block_indices = torch::Tensor();
  params.block_copy.cum_sum = torch::Tensor();
  params.parallel.dp_ep_padding_data.attn_padding_idx() = torch::Tensor();
  params.parallel.dp_ep_padding_data.attn_unpadding_idx() = torch::Tensor();
  params.parallel.dp_ep_padding_data.ffn_padding_idx() = torch::Tensor();
  params.parallel.dp_ep_padding_data.ffn_unpadding_idx() = torch::Tensor();
  params.parallel.dp_ep_padding_data.lm_head_skip_padding_token_indices() =
      torch::Tensor();
  params.parallel.dp_ep_padding_data.gather_prenorm_idx() = torch::Tensor();
  params.parallel.dp_ep_padding_data.padding_idx() = torch::Tensor();
  params.parallel.dp_ep_padding_data.un_padding_idx() = torch::Tensor();
  params.parallel.dp_ep_padding_data.dynamic_ep_idx() = torch::Tensor();
  params.parallel.dp_ep_padding_data.moe_idx() = torch::Tensor();
  params.parallel.dp_ep_padding_data.expert_array() = torch::Tensor();
  params.parallel.dp_ep_padding_data.post_lmhead_gather_indices() =
      torch::Tensor();
  params.expert.expert_array = torch::Tensor();
  params.expert.eplb_decode_token_mask = torch::Tensor();
  params.graph.attn_mask = torch::Tensor();
  params.graph.tiling_data = torch::Tensor();
  params.graph.expanded_kv_seq_lens = torch::Tensor();
  params.graph.expanded_block_tables = torch::Tensor();
  params.graph.expanded_paged_kv_indptr = torch::Tensor();
  params.graph.expanded_paged_kv_indices = torch::Tensor();
  params.graph.expanded_paged_kv_last_page_len = torch::Tensor();
  params.graph.expanded_tiling_data = torch::Tensor();
  params.num_accepted_tokens = torch::Tensor();
  params.mtp_shifted_token_ids = torch::Tensor();
}

inline bool add_attention_to_plan(const AttentionInput& source,
                                  AttentionInput& target,
                                  ForwardInputBufferPlan& plan) {
  return plan.add(source.device.q_seq_lens, &target.device.q_seq_lens) &&
         plan.add(source.device.kv_seq_lens, &target.device.kv_seq_lens) &&
         plan.add(source.device.q_cu_seq_lens, &target.device.q_cu_seq_lens) &&
         plan.add(source.device.new_cache_slots,
                  &target.device.new_cache_slots) &&
         plan.add(source.device.block_tables, &target.device.block_tables) &&
         plan.add(source.device.paged_kv_indptr,
                  &target.device.paged_kv_indptr) &&
         plan.add(source.device.paged_kv_indices,
                  &target.device.paged_kv_indices) &&
         plan.add(source.device.paged_kv_last_page_len,
                  &target.device.paged_kv_last_page_len) &&
         plan.add(source.device.new_cache_slot_offsets,
                  &target.device.new_cache_slot_offsets) &&
         plan.add(source.device.kv_cache_start_offsets,
                  &target.device.kv_cache_start_offsets) &&
         plan.add(source.device.kv_cache_tokens_nums,
                  &target.device.kv_cache_tokens_nums) &&
         plan.add(source.device.ring_cur_seqlen,
                  &target.device.ring_cur_seqlen) &&
         plan.add(source.device.ring_cache_seqlen,
                  &target.device.ring_cache_seqlen) &&
         plan.add(source.device.in_prefix_slots,
                  &target.device.in_prefix_slots);
}

struct ModelTensorPlanSourceOverrides {
  const torch::Tensor* input_embedding = nullptr;
  const torch::Tensor* linear_state_indices = nullptr;
  const torch::Tensor* mtp_bootstrap_embeddings = nullptr;
  const torch::Tensor* expanded_kv_seq_lens = nullptr;
  const torch::Tensor* num_accepted_tokens = nullptr;
};

inline bool add_model_tensors_to_plan(
    const ModelInputParams& source,
    ModelInputParams& target,
    ForwardInputBufferPlan& plan,
    const ModelTensorPlanSourceOverrides* overrides = nullptr) {
  const torch::Tensor& input_embedding =
      overrides != nullptr && overrides->input_embedding != nullptr
          ? *overrides->input_embedding
          : source.embedding.input_embedding;
  const torch::Tensor& linear_state_indices =
      overrides != nullptr && overrides->linear_state_indices != nullptr
          ? *overrides->linear_state_indices
          : source.embedding.linear_state_indices;
  const torch::Tensor& mtp_bootstrap_embeddings =
      overrides != nullptr && overrides->mtp_bootstrap_embeddings != nullptr
          ? *overrides->mtp_bootstrap_embeddings
          : source.embedding.mtp_bootstrap_embeddings;
  const torch::Tensor& expanded_kv_seq_lens =
      overrides != nullptr && overrides->expanded_kv_seq_lens != nullptr
          ? *overrides->expanded_kv_seq_lens
          : source.graph.expanded_kv_seq_lens;
  const torch::Tensor& num_accepted_tokens =
      overrides != nullptr && overrides->num_accepted_tokens != nullptr
          ? *overrides->num_accepted_tokens
          : source.num_accepted_tokens;
  return plan.add(input_embedding, &target.embedding.input_embedding) &&
         plan.add(linear_state_indices,
                  &target.embedding.linear_state_indices) &&
         plan.add(mtp_bootstrap_embeddings,
                  &target.embedding.mtp_bootstrap_embeddings) &&
         plan.add(source.embedding.predecessor_rows,
                  &target.embedding.predecessor_rows) &&
         plan.add(source.embedding.mtp_shifted_token_ids,
                  &target.embedding.mtp_shifted_token_ids) &&
         plan.add(source.block_copy.src_block_indices,
                  &target.block_copy.src_block_indices) &&
         plan.add(source.block_copy.dst_block_indices,
                  &target.block_copy.dst_block_indices) &&
         plan.add(source.block_copy.cum_sum, &target.block_copy.cum_sum) &&
         plan.add(source.parallel.dp_ep_padding_data.attn_padding_idx(),
                  &target.parallel.dp_ep_padding_data.attn_padding_idx()) &&
         plan.add(source.parallel.dp_ep_padding_data.attn_unpadding_idx(),
                  &target.parallel.dp_ep_padding_data.attn_unpadding_idx()) &&
         plan.add(source.parallel.dp_ep_padding_data.ffn_padding_idx(),
                  &target.parallel.dp_ep_padding_data.ffn_padding_idx()) &&
         plan.add(source.parallel.dp_ep_padding_data.ffn_unpadding_idx(),
                  &target.parallel.dp_ep_padding_data.ffn_unpadding_idx()) &&
         plan.add(source.parallel.dp_ep_padding_data
                      .lm_head_skip_padding_token_indices(),
                  &target.parallel.dp_ep_padding_data
                       .lm_head_skip_padding_token_indices()) &&
         plan.add(source.parallel.dp_ep_padding_data.gather_prenorm_idx(),
                  &target.parallel.dp_ep_padding_data.gather_prenorm_idx()) &&
         plan.add(source.parallel.dp_ep_padding_data.padding_idx(),
                  &target.parallel.dp_ep_padding_data.padding_idx()) &&
         plan.add(source.parallel.dp_ep_padding_data.un_padding_idx(),
                  &target.parallel.dp_ep_padding_data.un_padding_idx()) &&
         plan.add(source.parallel.dp_ep_padding_data.dynamic_ep_idx(),
                  &target.parallel.dp_ep_padding_data.dynamic_ep_idx()) &&
         plan.add(source.parallel.dp_ep_padding_data.moe_idx(),
                  &target.parallel.dp_ep_padding_data.moe_idx()) &&
         plan.add(source.parallel.dp_ep_padding_data.expert_array(),
                  &target.parallel.dp_ep_padding_data.expert_array()) &&
         plan.add(
             source.parallel.dp_ep_padding_data.post_lmhead_gather_indices(),
             &target.parallel.dp_ep_padding_data
                  .post_lmhead_gather_indices()) &&
         plan.add(source.expert.expert_array, &target.expert.expert_array) &&
         plan.add(source.expert.eplb_decode_token_mask,
                  &target.expert.eplb_decode_token_mask) &&
         plan.add(source.graph.attn_mask, &target.graph.attn_mask) &&
         plan.add(source.graph.tiling_data, &target.graph.tiling_data) &&
         plan.add(expanded_kv_seq_lens, &target.graph.expanded_kv_seq_lens) &&
         plan.add(source.graph.expanded_block_tables,
                  &target.graph.expanded_block_tables) &&
         plan.add(source.graph.expanded_paged_kv_indptr,
                  &target.graph.expanded_paged_kv_indptr) &&
         plan.add(source.graph.expanded_paged_kv_indices,
                  &target.graph.expanded_paged_kv_indices) &&
         plan.add(source.graph.expanded_paged_kv_last_page_len,
                  &target.graph.expanded_paged_kv_last_page_len) &&
         plan.add(source.graph.expanded_tiling_data,
                  &target.graph.expanded_tiling_data) &&
         plan.add(num_accepted_tokens, &target.num_accepted_tokens) &&
         plan.add(source.mtp_shifted_token_ids, &target.mtp_shifted_token_ids);
}

inline torch::Tensor gather_tensor_by_indices(
    const torch::Tensor& tensor,
    const std::vector<int64_t>& indices) {
  if (!tensor.defined()) {
    return tensor;
  }
  torch::Tensor cpu_tensor = tensor.device().is_cpu() ? tensor : tensor.cpu();
  cpu_tensor = cpu_tensor.contiguous();
  torch::Tensor gather_indices = torch::tensor(
      indices, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
  if (cpu_tensor.dim() <= 1) {
    return cpu_tensor.index_select(0, gather_indices);
  }
  CHECK_EQ(cpu_tensor.dim(), 2) << "Expected 1-D or 2-D tensor for CP shard";
  return cpu_tensor.index_select(1, gather_indices);
}

inline torch::Tensor gather_tensor_by_indices_on_dim(
    const torch::Tensor& tensor,
    const std::vector<int64_t>& indices,
    int64_t dim) {
  if (!tensor.defined()) {
    return tensor;
  }
  torch::Tensor cpu_tensor = tensor.device().is_cpu() ? tensor : tensor.cpu();
  cpu_tensor = cpu_tensor.contiguous();
  torch::Tensor gather_indices = torch::tensor(
      indices, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
  return cpu_tensor.index_select(dim, gather_indices);
}

inline torch::Tensor int_vector_to_cpu_tensor(
    const std::vector<int32_t>& values) {
  if (values.empty()) {
    return torch::Tensor();
  }
  return torch::tensor(values,
                       torch::TensorOptions()
                           .dtype(torch::kInt)
                           .device(torch::kCPU)
                           .pinned_memory(true));
}

template <typename T>
inline std::vector<T> tensor_to_vector(const torch::Tensor& tensor) {
  if (!tensor.defined() || tensor.numel() == 0) {
    return {};
  }
  torch::Tensor cpu_tensor = tensor.device().is_cpu() ? tensor : tensor.cpu();
  cpu_tensor = cpu_tensor.contiguous();
  if (cpu_tensor.scalar_type() != get_scalar_type<T>()) {
    cpu_tensor = cpu_tensor.to(get_scalar_type<T>());
  }
  const T* data_ptr = cpu_tensor.data_ptr<T>();
  const size_t size = static_cast<size_t>(cpu_tensor.numel());
  return std::vector<T>(data_ptr, data_ptr + size);
}

}  // namespace detail

class WorkerType {
 public:
  enum Value : int8_t {
    INVALID = 0,
    LLM,     // LLM
    VLM,     // VLM
    DIT,     // DIT
    ELM,     // Embedding LM
    EVLM,    // Embedding VLM
    REC,     // Rec
    MMEVLM,  // Encoder Embedding VLM
  };

  constexpr WorkerType(Value v) : value_(v) {}
  WorkerType(const std::string& str) {
    if (str == "LLM") {
      value_ = LLM;
    } else if (str == "VLM") {
      value_ = VLM;
    } else if (str == "DIT") {
      value_ = DIT;
    } else if (str == "ELM") {
      value_ = ELM;
    } else if (str == "EVLM") {
      value_ = EVLM;
    } else if (str == "REC") {
      value_ = REC;
    } else if (str == "MMEVLM") {
      value_ = MMEVLM;
    } else {
      value_ = INVALID;
    }
  }

  WorkerType() = delete;

  constexpr operator Value() const { return value_; }
  explicit operator bool() = delete;

  bool operator==(WorkerType rhs) const { return value_ == rhs.value_; }
  bool operator!=(WorkerType rhs) const { return value_ != rhs.value_; }
  bool operator==(Value rhs) const { return value_ == rhs; }
  bool operator!=(Value rhs) const { return value_ != rhs; }

  constexpr const char* to_string() const {
    if (this->value_ == LLM) {
      return "LLM";
    } else if (this->value_ == VLM) {
      return "VLM";
    } else if (this->value_ == DIT) {
      return "DIT";
    } else if (this->value_ == ELM) {
      return "ELM";
    } else if (this->value_ == EVLM) {
      return "EVLM";
    } else if (this->value_ == REC) {
      return "REC";
    } else if (this->value_ == MMEVLM) {
      return "MMEVLM";
    } else {
      return "INVALID";
    }
  }

 private:
  Value value_;
};

// Worker-local KV slot layout for NPU CP (not transported).
enum class KvSlotLayout : int8_t {
  LOGICAL_REAL = 0,  // Builder slots; input to prepare_cache_slots.
  NPU_CP_RECOVERED_PHYSICAL = 1,  // Already CP-expanded; skip re-prepare.
};

// Step-level decode metadata for Rec multi-round (device loop).
struct StepDecodeMeta {
  int32_t batch_size = 0;
  int32_t beam_width = 1;
  int32_t current_round = 0;
  int32_t total_round = 0;
  // Planned decode kv cache shape: [batch_size * beam_width, n_kv_heads,
  // step_rounds, head_dim]
  std::vector<int64_t> full_kv_shape;
  // Flattened decode positions for each sequence.
  std::vector<int32_t> decode_positions_vec;
};

// Inputs for forward execution
struct ForwardInput {
  ForwardInput to(const torch::Device& device, torch::ScalarType dtype) const {
    if (device_tensors_ready) {
      return *this;
    }

    if (input_host_buffer_has_layout) {
      ForwardInput buffer_inputs;
      const bool materialize_device_buffer =
          ::xllm::ExecutionConfig::get_instance()
              .use_contiguous_input_buffer() &&
          detail::supports_contiguous_forward_input_buffer(device);
      if (detail::unpack_from_input_host_buffer(
              *this, device, dtype, buffer_inputs, materialize_device_buffer)) {
        if (buffer_inputs.device_tensors_ready) {
          return buffer_inputs;
        }
        return buffer_inputs.to(device, dtype);
      }
    }

    if (::xllm::ExecutionConfig::get_instance().use_contiguous_input_buffer() &&
        detail::supports_contiguous_forward_input_buffer(device)) {
      ForwardInput contiguous_inputs;
      if (to_contiguous_input_buffer(device, contiguous_inputs)) {
        return contiguous_inputs;
      }
    }

    ForwardInput inputs;
    set_host_views(inputs);
    const torch::Tensor& source_token_ids =
        inputs.token_ids_host.defined() ? inputs.token_ids_host : token_ids;
    const torch::Tensor& source_positions =
        inputs.positions_host.defined() ? inputs.positions_host : positions;
    inputs.token_ids = safe_to(source_token_ids, device, true);
    inputs.positions = detail::normalize_positions_for_device(
        safe_to(source_positions, device, true));
    inputs.input_params = input_params.to(device);
    inputs.sampling_params = sampling_params.to(device, dtype);
    inputs.decoder_sampling_params = decoder_sampling_params.to(device, dtype);
    copy_metadata_to(inputs);
    inputs.input_host_buffer = input_host_buffer;
    inputs.device_input_buffer = device_input_buffer;
    inputs.input_host_buffer_has_layout = input_host_buffer_has_layout;
    inputs.device_tensors_ready = true;
    inputs.kv_slot_layout = kv_slot_layout;
    return inputs;
  }

  bool to_contiguous_input_buffer(const torch::Device& device,
                                  ForwardInput& inputs) const {
    copy_metadata_to(inputs);
    set_host_views(inputs);

    const ModelInputParams& source_params = input_params;
    if (missing_required_host_views(inputs) ||
        detail::has_contiguous_input_buffer_exclusions(source_params)) {
      return false;
    }

    inputs.input_params = source_params;
    detail::clear_contiguous_input_buffer_tensor_targets(inputs.input_params);

    inputs.sampling_params = sampling_params;
    inputs.decoder_sampling_params = decoder_sampling_params;

    torch::Tensor positions_for_device =
        detail::normalize_positions_for_device(inputs.positions_host);

    detail::ForwardInputBufferPlan plan;
    if (!plan.add(inputs.token_ids_host, &inputs.token_ids) ||
        !plan.add(positions_for_device, &inputs.positions)) {
      return false;
    }

    if (!detail::add_attention_to_plan(
            source_params.attention, inputs.input_params.attention, plan) ||
        !detail::add_model_tensors_to_plan(
            source_params, inputs.input_params, plan)) {
      return false;
    }

    if (!detail::add_sampling_to_plan(
            sampling_params, inputs.sampling_params, plan) ||
        !detail::add_sampling_to_plan(
            decoder_sampling_params, inputs.decoder_sampling_params, plan)) {
      return false;
    }

    const uint64_t total_bytes = plan.prepare_layout();
    inputs.prepared_input_layout_signature = plan.layout_signature();
    if (total_bytes > 0) {
      inputs.input_host_buffer = plan.build_host_buffer(total_bytes);
      inputs.device_input_buffer =
          safe_to(inputs.input_host_buffer,
                  torch::TensorOptions().dtype(torch::kUInt8).device(device),
                  true);
      plan.bind_device_views(inputs.device_input_buffer, device);
    }

    inputs.device_tensors_ready = true;
    inputs.input_host_buffer_has_layout = false;
    return true;
  }

  void copy_metadata_to(ForwardInput& inputs) const {
    inputs.transfer_kv_infos = transfer_kv_infos;
    inputs.step_decode = step_decode;
    inputs.skip_sampling_for_logits_only = skip_sampling_for_logits_only;
    inputs.return_selected_hidden = return_selected_hidden;
    inputs.kv_slot_layout = kv_slot_layout;
    inputs.metadata_ready_event = metadata_ready_event;
    inputs.retained_device_tensors = retained_device_tensors;
    inputs.sample_sequence_ids = sample_sequence_ids;
    inputs.sample_prior_output_rows = sample_prior_output_rows;
    inputs.json_object_states = json_object_states;
    inputs.json_object_state_snapshots = json_object_state_snapshots;
    inputs.prepared_input_layout_signature = prepared_input_layout_signature;
    inputs.prepared_arena_h2d_bytes = prepared_arena_h2d_bytes;
    inputs.prepared_arena_h2d_copies = prepared_arena_h2d_copies;
    inputs.prepared_arena_d2d_bytes = prepared_arena_d2d_bytes;
    inputs.prepared_arena_d2d_copies = prepared_arena_d2d_copies;
  }

  void set_host_views(ForwardInput& inputs) const {
    inputs.token_ids_host =
        token_ids_host.defined() ? token_ids_host : cpu_view(token_ids);
    inputs.positions_host =
        positions_host.defined() ? positions_host : cpu_view(positions);
  }

  bool missing_required_host_views(const ForwardInput& inputs) const {
    return (token_ids.defined() && !inputs.token_ids_host.defined()) ||
           (positions.defined() && !inputs.positions_host.defined());
  }

  const torch::Tensor& host_token_ids() const {
    return token_ids_host.defined() ? token_ids_host : token_ids;
  }

  const torch::Tensor& host_positions() const {
    return positions_host.defined() ? positions_host : positions;
  }

  static torch::Tensor cpu_view(const torch::Tensor& tensor) {
    if (tensor.defined() && tensor.device().is_cpu()) {
      return tensor;
    }
    return torch::Tensor();
  }

  void print() const {
    LOG(INFO) << "  token_ids: " << token_ids << std::endl;
    LOG(INFO) << "  positions: " << positions << std::endl;
    input_params.print();
    LOG(INFO) << " params.selected_token_idxes "
              << sampling_params.selected_token_idxes;
    LOG(INFO) << " params.sample_idxes " << sampling_params.sample_idxes;
    LOG(INFO) << " params.do_sample " << sampling_params.do_sample;
  }

  const StepDecodeMeta* step_meta() const {
    return step_decode ? &(*step_decode) : nullptr;
  }

  bool has_step_meta() const { return step_decode.has_value(); }

  // flatten token ids
  torch::Tensor token_ids;
  // flatten positions
  torch::Tensor positions;
  torch::Tensor token_ids_host;
  torch::Tensor positions_host;
  ModelInputParams input_params;
  SamplingParameters sampling_params;
  SamplingParameters decoder_sampling_params;
  std::vector<std::string> sample_sequence_ids;
  std::vector<int32_t> sample_prior_output_rows;
  std::vector<JsonObjectGrammarState> json_object_states;
  std::vector<JsonObjectGrammarSnapshot> json_object_state_snapshots;
  // Flattened [sequence][draft position] flags produced during MTP
  // validation. This is execution-local metadata and is not transported.
  std::vector<uint8_t> json_object_invalid_draft;
  // Errors detected while aligning prior overlap output with grammar rows.
  std::vector<JsonObjectOutputError> json_object_errors;

  // step-level decode metadata
  std::optional<StepDecodeMeta> step_decode;
  // If true, skip sampler forward and only keep logits.
  bool skip_sampling_for_logits_only = false;
  // If true, populate ForwardOutput.selected_hidden with hidden states matching
  // the `logits` selection layout. Used by DSpark ConfidenceHead which needs
  // pre-lm_head hidden states of the draft tokens.
  bool return_selected_hidden = false;

  // kv info for disaggregated prefill/decode
  std::vector<TransferKVInfo> transfer_kv_infos;

  // A tensor used to store all device-side input data, with other input tensors
  // constructed based on the address and offset of this tensor.
  torch::Tensor input_host_buffer;
  torch::Tensor device_input_buffer;
  bool input_host_buffer_has_layout = false;

  // True when token_ids, positions, model input tensors and sampling tensors
  // already point to the device-side views for execution. Worker prepare can
  // then skip rebuilding/H2D in ForwardInput::to().
  bool device_tensors_ready = false;

  // Stable fingerprint of the field offsets, dtypes, and shapes inside a
  // PreparedInputArena. It lets Prepared Graph bind only captures whose tensor
  // addresses have the same fixed layout.
  uint64_t prepared_input_layout_signature = 0;

  // Transfer volume submitted while binding this input to its fixed Prepared
  // Arena. H2D includes aligned padding in Host-only ranges and skips explicit
  // Device-source ranges. D2D covers only sources that cannot be reconstructed
  // from Host.
  uint64_t prepared_arena_h2d_bytes = 0;
  int32_t prepared_arena_h2d_copies = 0;
  uint64_t prepared_arena_d2d_bytes = 0;
  int32_t prepared_arena_d2d_copies = 0;

  // new_cache_slots layout; flip after one-shot CP remap.
  KvSlotLayout kv_slot_layout = KvSlotLayout::LOGICAL_REAL;

  // Device-side readiness dependencies for inputs prepared on a different
  // stream. These are local runtime handles and are intentionally not included
  // in proto or shared-memory transport.
  StreamEventPtr metadata_ready_event;

  // Keep cross-stream metadata sources alive through no-sync execution. These
  // handles are local runtime state and are not serialized.
  std::vector<torch::Tensor> retained_device_tensors;
};

// output after forward execution
struct ForwardOutput {
  // sample parameters for speculative decoding
  torch::Tensor do_sample;
  // whether to return logprobs
  bool logprobs = false;
  // max number of top logprobs in the batch
  int64_t max_top_logprobs = 0;
  SampleOutput sample_output;
  // The target sampler applies packed token masks in-place before returning
  // sampled tokens. MTP validation uses this local contract to avoid applying
  // the same mask to target logits a second time.
  bool filter_bitmask_applied_to_logits = false;
  std::vector<JsonObjectOutputError> json_object_errors;
  // Keep no-sync input tensor handles alive until downstream consumers finish
  // using outputs on the same compute stream. Composite workers append child
  // outputs' retained inputs here. Local runtime handles; not in proto/shm.
  std::vector<std::shared_ptr<ForwardInput>> retained_inputs;
  // Device-side readiness dependency for no-sync outputs. This local runtime
  // handle is intentionally not included in proto or shared-memory transport.
  StreamEventPtr ready_event;
  torch::Tensor logits;
  torch::Tensor embedding;
  // Selected hidden states matching `logits` layout: [num_selected,
  // hidden_dim]. Populated when a speculative worker asks for the pre-lm_head
  // hidden (e.g. DSpark's ConfidenceHead needs the draft-step hidden). Only
  // computed when `input.return_selected_hidden` is true.
  torch::Tensor selected_hidden;
  // Backend-neutral state for the next MTP draft step.
  MtpTopkStatePtr mtp_topk_state;

  // for eplb, collect the tokens load of experts on each worker.
  torch::Tensor expert_load_data;
  // EPLB prepare-attempt token completed by this worker.
  int64_t prepared_token = -1;

  BeamSearchOutput beam_search_output;
  torch::Tensor beam_sequence_group;

  // dit output data
  DiTForwardOutput dit_forward_output;
};

inline void copy_retained_inputs(ForwardOutput& destination,
                                 const ForwardOutput& source) {
  destination.retained_inputs.insert(destination.retained_inputs.end(),
                                     source.retained_inputs.begin(),
                                     source.retained_inputs.end());
}

inline void transfer_retained_inputs(ForwardOutput& destination,
                                     ForwardOutput& source) {
  CHECK_NE(&destination, &source)
      << "transfer_retained_inputs cannot alias source and destination";
  destination.retained_inputs.insert(
      destination.retained_inputs.end(),
      std::make_move_iterator(source.retained_inputs.begin()),
      std::make_move_iterator(source.retained_inputs.end()));
  source.retained_inputs.clear();
}

inline std::vector<std::shared_ptr<ForwardInput>> take_retained_inputs(
    ForwardOutput& source) {
  return std::exchange(source.retained_inputs, {});
}

struct RawSampleOutput {
  std::vector<RawToken> tokens;  // num tokens
  // multimodal embedding output for this sequence
  std::vector<torch::Tensor> mm_embeddings;
};

struct RawForwardOutput {
  std::vector<RawSampleOutput> outputs;  // num seqs
  std::vector<JsonObjectOutputError> json_object_errors;
  std::vector<int64_t> expert_load_data;
  int64_t prepared_token = -1;
  // beam search kernel output
  std::vector<int32_t> src_seq_idxes;
  std::vector<int32_t> out_tokens;
  std::vector<float> out_logprobs;

  // batch-level beam output for Rec multi-round mode
  std::vector<int32_t> beam_sequence_group;  // flattened 2D
  // dit output data
  DiTForwardOutput dit_forward_output;
};

struct BatchedForwardInputs {
  std::vector<ForwardInput> micro_inputs;
  SamplingParameters concated_sampling_params;
};

}  // namespace xllm
