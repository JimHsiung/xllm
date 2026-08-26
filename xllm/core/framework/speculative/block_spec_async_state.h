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

#pragma once

#include <torch/torch.h>

#include <cstdint>

namespace xllm::block_spec_async {

enum class CacheSlotMappingMode : int8_t {
  LINEAR = 0,
  CIRCULAR,
};

// Fixed Slot-local output of DFlash/DSpark rejection sampling. Context hidden,
// position, and cache-slot rows keep the verifier's static [batch, width]
// layout. Invalid suffix rows are masked and redirected to padding slot 0 so
// Context-KV publication does not require a Host-side compaction pass.
struct BlockSpecDeviceStepState {
  torch::Tensor accepted_tokens;
  torch::Tensor accepted_lengths;
  torch::Tensor context_hidden;
  torch::Tensor context_positions;
  torch::Tensor context_cache_slots;
  torch::Tensor valid_tokens;
  torch::Tensor tail_tokens;
  torch::Tensor tail_context_hidden;
  torch::Tensor next_positions;
  torch::Tensor valid_rows;
};

// Fixed scratch for deriving Context-KV write metadata entirely on Device.
// position_offsets is initialized once; every other Tensor is overwritten by
// out/in-place operations for each Task.
struct BlockSpecContextKvPatchWorkspace {
  torch::Tensor position_offsets;
  torch::Tensor position_block_indices;
  torch::Tensor block_indices;
  torch::Tensor block_ids;
  torch::Tensor cache_offsets;
  torch::Tensor invalid_tokens;
  torch::Tensor row_indices;
  torch::Tensor last_sequence_indices;
  torch::Tensor last_flat_indices;
};

// Fixed destination views in the next Task's block-draft input. Rows without
// a predecessor keep their Worker-prepared initial values.
struct BlockSpecContinuationState {
  torch::Tensor anchor_tokens;
  torch::Tensor anchor_context_hidden;
  torch::Tensor base_positions;
};

struct BlockSpecPredecessorPatchWorkspace {
  torch::Tensor gather_rows;
  torch::Tensor gathered_valid_rows;
  torch::Tensor continuation_mask;
  torch::Tensor gathered_tokens;
  torch::Tensor patched_tokens;
  torch::Tensor gathered_context_hidden;
  torch::Tensor patched_context_hidden;
  torch::Tensor gathered_positions;
  torch::Tensor patched_positions;
};

// Fixed DFlash/DSpark Decode model-input views. Query/Target token and
// metadata tensors point into PreparedInputArena partitions; source block
// tables retain one row per logical sequence.
struct BlockSpecDecodeInputPatchTarget {
  torch::Tensor query_token_ids;
  torch::Tensor query_positions;
  torch::Tensor query_kv_seq_lens;
  torch::Tensor query_new_cache_slots;
  torch::Tensor target_token_ids;
  torch::Tensor target_positions;
  torch::Tensor target_kv_seq_lens;
  torch::Tensor target_new_cache_slots;
  torch::Tensor source_block_tables;
  CacheSlotMappingMode cache_slot_mapping_mode = CacheSlotMappingMode::LINEAR;
};

struct BlockSpecDecodeInputPatchWorkspace {
  torch::Tensor query_position_offsets;
  torch::Tensor query_position_block_indices;
  torch::Tensor query_block_indices;
  torch::Tensor query_block_ids;
  torch::Tensor query_cache_offsets;
  torch::Tensor target_position_offsets;
  torch::Tensor target_position_block_indices;
  torch::Tensor target_block_indices;
  torch::Tensor target_block_ids;
  torch::Tensor target_cache_offsets;
};

BlockSpecDeviceStepState allocate_block_spec_device_step_state(
    int64_t max_rows,
    int64_t accepted_token_capacity,
    int64_t context_hidden_size,
    const torch::TensorOptions& token_options,
    const torch::TensorOptions& hidden_options,
    const torch::TensorOptions& position_options,
    const torch::TensorOptions& cache_slot_options);

BlockSpecContextKvPatchWorkspace allocate_block_spec_context_kv_patch_workspace(
    int64_t max_rows,
    int64_t accepted_token_capacity,
    const torch::TensorOptions& position_options,
    const torch::TensorOptions& cache_slot_options);

BlockSpecPredecessorPatchWorkspace
allocate_block_spec_predecessor_patch_workspace(
    int64_t max_rows,
    int64_t context_hidden_size,
    const torch::TensorOptions& token_options,
    const torch::TensorOptions& hidden_options,
    const torch::TensorOptions& position_options);

BlockSpecDecodeInputPatchWorkspace
allocate_block_spec_decode_input_patch_workspace(
    int64_t max_rows,
    int64_t query_width,
    int64_t target_width,
    const torch::TensorOptions& position_options,
    const torch::TensorOptions& cache_slot_options);

// Publishes fixed-width rejection outputs and derives the Context-KV scatter
// metadata without D2H, Host scans, dynamic output allocation, or destination
// Tensor replacement. accepted_tokens retains the full Target Verify width,
// keeps its first token valid, and uses -1 only for a contiguous invalid
// suffix. Thus continuation advances beyond the one-token Host template by at
// most accepted_token_capacity - 1.
void publish_block_spec_context_kv_state(
    const torch::Tensor& accepted_tokens,
    const torch::Tensor& accepted_context_hidden,
    const torch::Tensor& base_positions,
    const torch::Tensor& block_tables,
    int32_t block_size,
    CacheSlotMappingMode cache_slot_mapping_mode,
    BlockSpecDeviceStepState& destination,
    BlockSpecContextKvPatchWorkspace& workspace);

void patch_block_spec_continuation_rows(
    const BlockSpecDeviceStepState& predecessor,
    const torch::Tensor& predecessor_rows,
    BlockSpecContinuationState& continuation,
    BlockSpecPredecessorPatchWorkspace& workspace);

// Rebuilds all position-dependent fixed Query/Target metadata after the
// predecessor continuation is applied. target_kv_seq_lens may use either the
// tokenwise [batch * width] layout or the chunked [batch] layout.
void patch_block_spec_decode_input_geometry(
    const BlockSpecContinuationState& continuation,
    int32_t block_size,
    BlockSpecDecodeInputPatchTarget& target,
    BlockSpecDecodeInputPatchWorkspace& workspace);

void patch_block_spec_target_token_ids(const torch::Tensor& draft_token_ids,
                                       torch::Tensor target_token_ids);

}  // namespace xllm::block_spec_async
