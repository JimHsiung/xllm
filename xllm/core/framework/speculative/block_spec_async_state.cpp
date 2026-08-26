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

#include "core/framework/speculative/block_spec_async_state.h"

#include <glog/logging.h>

namespace xllm::block_spec_async {
namespace {

void check_block_spec_tensor_contract(const torch::Tensor& tensor,
                                      const torch::Device& expected_device,
                                      torch::ScalarType expected_dtype,
                                      const char* contract_name,
                                      const char* tensor_name) {
  CHECK(tensor.defined()) << contract_name << " " << tensor_name
                          << " must be defined";
  CHECK_EQ(tensor.device(), expected_device)
      << contract_name << " " << tensor_name
      << " must share the contract device";
  CHECK_EQ(tensor.scalar_type(), expected_dtype)
      << contract_name << " " << tensor_name << " has an incompatible dtype";
}

void map_cache_slots_2d_out(const torch::Tensor& positions,
                            const torch::Tensor& block_tables,
                            int32_t block_size,
                            CacheSlotMappingMode cache_slot_mapping_mode,
                            torch::Tensor destination,
                            torch::Tensor position_block_indices,
                            torch::Tensor block_indices,
                            torch::Tensor block_ids,
                            torch::Tensor cache_offsets) {
  CHECK_GT(block_size, 0);
  CHECK_EQ(positions.dim(), 2);
  CHECK_EQ(block_tables.dim(), 2);
  CHECK_EQ(block_tables.size(0), positions.size(0));
  CHECK(destination.sizes() == positions.sizes());
  CHECK(position_block_indices.sizes() == positions.sizes());
  CHECK(block_indices.sizes() == positions.sizes());
  CHECK(block_ids.sizes() == positions.sizes());
  CHECK(cache_offsets.sizes() == positions.sizes());
  CHECK_EQ(block_indices.scalar_type(), torch::kLong);
  CHECK_EQ(block_tables.scalar_type(), destination.scalar_type());

  torch::floor_divide_out(position_block_indices, positions, block_size);
  if (cache_slot_mapping_mode == CacheSlotMappingMode::CIRCULAR) {
    CHECK_GT(block_tables.size(1), 0)
        << "Circular cache-slot mapping requires a non-empty block table";
    torch::remainder_out(
        position_block_indices, position_block_indices, block_tables.size(1));
  }
  block_indices.copy_(position_block_indices, /*non_blocking=*/true);
  const torch::Tensor expanded_block_tables =
      block_tables.unsqueeze(/*dim=*/1).expand(
          {positions.size(0), positions.size(1), block_tables.size(1)});
  torch::Tensor block_ids_rows = block_ids.unsqueeze(/*dim=*/2);
  torch::gather_out(block_ids_rows,
                    expanded_block_tables,
                    /*dim=*/2,
                    block_indices.unsqueeze(/*dim=*/2));
  torch::remainder_out(cache_offsets, positions, block_size);
  torch::mul_out(destination, block_ids, block_size);
  destination.add_(cache_offsets);
}

}  // namespace

BlockSpecDeviceStepState allocate_block_spec_device_step_state(
    int64_t max_rows,
    int64_t accepted_token_capacity,
    int64_t context_hidden_size,
    const torch::TensorOptions& token_options,
    const torch::TensorOptions& hidden_options,
    const torch::TensorOptions& position_options,
    const torch::TensorOptions& cache_slot_options) {
  CHECK_GT(max_rows, 0);
  CHECK_GT(accepted_token_capacity, 0);
  CHECK_GT(context_hidden_size, 0);
  CHECK_EQ(token_options.device(), hidden_options.device());
  CHECK_EQ(token_options.device(), position_options.device());
  CHECK_EQ(token_options.device(), cache_slot_options.device());

  BlockSpecDeviceStepState state;
  state.accepted_tokens =
      torch::full({max_rows, accepted_token_capacity}, -1, token_options);
  state.accepted_lengths =
      torch::zeros({max_rows}, token_options.dtype(torch::kLong));
  state.context_hidden = torch::zeros(
      {max_rows, accepted_token_capacity, context_hidden_size}, hidden_options);
  state.context_positions =
      torch::zeros({max_rows, accepted_token_capacity}, position_options);
  state.context_cache_slots =
      torch::zeros({max_rows, accepted_token_capacity}, cache_slot_options);
  state.valid_tokens = torch::zeros({max_rows, accepted_token_capacity},
                                    token_options.dtype(torch::kBool));
  state.tail_tokens = torch::empty({max_rows}, token_options);
  state.tail_context_hidden =
      torch::empty({max_rows, context_hidden_size}, hidden_options);
  state.next_positions = torch::zeros({max_rows}, position_options);
  state.valid_rows =
      torch::zeros({max_rows}, token_options.dtype(torch::kBool));
  return state;
}

BlockSpecContextKvPatchWorkspace allocate_block_spec_context_kv_patch_workspace(
    int64_t max_rows,
    int64_t accepted_token_capacity,
    const torch::TensorOptions& position_options,
    const torch::TensorOptions& cache_slot_options) {
  CHECK_GT(max_rows, 0);
  CHECK_GT(accepted_token_capacity, 0);
  CHECK_EQ(position_options.device(), cache_slot_options.device());

  BlockSpecContextKvPatchWorkspace workspace;
  workspace.position_offsets =
      torch::arange(accepted_token_capacity, position_options);
  workspace.position_block_indices =
      torch::empty({max_rows, accepted_token_capacity}, position_options);
  workspace.block_indices = torch::empty({max_rows, accepted_token_capacity},
                                         position_options.dtype(torch::kLong));
  workspace.block_ids =
      torch::empty({max_rows, accepted_token_capacity}, cache_slot_options);
  workspace.cache_offsets =
      torch::empty({max_rows, accepted_token_capacity}, cache_slot_options);
  workspace.invalid_tokens = torch::empty({max_rows, accepted_token_capacity},
                                          position_options.dtype(torch::kBool));
  workspace.row_indices =
      torch::arange(max_rows, position_options.dtype(torch::kLong));
  workspace.last_sequence_indices =
      torch::empty({max_rows}, position_options.dtype(torch::kLong));
  workspace.last_flat_indices =
      torch::empty({max_rows}, position_options.dtype(torch::kLong));
  return workspace;
}

BlockSpecPredecessorPatchWorkspace
allocate_block_spec_predecessor_patch_workspace(
    int64_t max_rows,
    int64_t context_hidden_size,
    const torch::TensorOptions& token_options,
    const torch::TensorOptions& hidden_options,
    const torch::TensorOptions& position_options) {
  CHECK_GT(max_rows, 0);
  CHECK_GT(context_hidden_size, 0);
  CHECK_EQ(token_options.device(), hidden_options.device());
  CHECK_EQ(token_options.device(), position_options.device());

  BlockSpecPredecessorPatchWorkspace workspace;
  workspace.gather_rows =
      torch::empty({max_rows}, token_options.dtype(torch::kLong));
  workspace.gathered_valid_rows =
      torch::empty({max_rows}, token_options.dtype(torch::kBool));
  workspace.continuation_mask =
      torch::empty({max_rows}, token_options.dtype(torch::kBool));
  workspace.gathered_tokens = torch::empty({max_rows}, token_options);
  workspace.patched_tokens = torch::empty({max_rows}, token_options);
  workspace.gathered_context_hidden =
      torch::empty({max_rows, context_hidden_size}, hidden_options);
  workspace.patched_context_hidden =
      torch::empty({max_rows, context_hidden_size}, hidden_options);
  workspace.gathered_positions = torch::empty({max_rows}, position_options);
  workspace.patched_positions = torch::empty({max_rows}, position_options);
  return workspace;
}

BlockSpecDecodeInputPatchWorkspace
allocate_block_spec_decode_input_patch_workspace(
    int64_t max_rows,
    int64_t query_width,
    int64_t target_width,
    const torch::TensorOptions& position_options,
    const torch::TensorOptions& cache_slot_options) {
  CHECK_GT(max_rows, 0);
  CHECK_GT(query_width, 0);
  CHECK_GT(target_width, 0);
  CHECK_EQ(position_options.device(), cache_slot_options.device());

  BlockSpecDecodeInputPatchWorkspace workspace;
  workspace.query_position_offsets =
      torch::arange(query_width, position_options);
  workspace.query_position_block_indices =
      torch::empty({max_rows, query_width}, position_options);
  workspace.query_block_indices = torch::empty(
      {max_rows, query_width}, position_options.dtype(torch::kLong));
  workspace.query_block_ids =
      torch::empty({max_rows, query_width}, cache_slot_options);
  workspace.query_cache_offsets =
      torch::empty({max_rows, query_width}, cache_slot_options);
  workspace.target_position_offsets =
      torch::arange(target_width, position_options);
  workspace.target_position_block_indices =
      torch::empty({max_rows, target_width}, position_options);
  workspace.target_block_indices = torch::empty(
      {max_rows, target_width}, position_options.dtype(torch::kLong));
  workspace.target_block_ids =
      torch::empty({max_rows, target_width}, cache_slot_options);
  workspace.target_cache_offsets =
      torch::empty({max_rows, target_width}, cache_slot_options);
  return workspace;
}

void publish_block_spec_context_kv_state(
    const torch::Tensor& accepted_tokens,
    const torch::Tensor& accepted_context_hidden,
    const torch::Tensor& base_positions,
    const torch::Tensor& block_tables,
    int32_t block_size,
    CacheSlotMappingMode cache_slot_mapping_mode,
    BlockSpecDeviceStepState& destination,
    BlockSpecContextKvPatchWorkspace& workspace) {
  CHECK_GT(block_size, 0);
  CHECK(accepted_tokens.defined());
  CHECK(accepted_context_hidden.defined());
  CHECK(base_positions.defined());
  CHECK(block_tables.defined());
  CHECK_EQ(accepted_tokens.dim(), 2);
  CHECK_EQ(accepted_tokens.scalar_type(), torch::kLong);
  CHECK_EQ(accepted_context_hidden.dim(), 3);
  CHECK_EQ(base_positions.dim(), 1);
  CHECK_EQ(block_tables.dim(), 2);
  CHECK(accepted_tokens.is_contiguous());
  CHECK(accepted_context_hidden.is_contiguous());

  const torch::Device publish_device = accepted_tokens.device();
  const torch::ScalarType hidden_dtype = accepted_context_hidden.scalar_type();
  const torch::ScalarType position_dtype = base_positions.scalar_type();
  const torch::ScalarType cache_slot_dtype = block_tables.scalar_type();
  check_block_spec_tensor_contract(accepted_context_hidden,
                                   publish_device,
                                   hidden_dtype,
                                   "Block-Spec publish",
                                   "accepted_context_hidden");
  check_block_spec_tensor_contract(base_positions,
                                   publish_device,
                                   position_dtype,
                                   "Block-Spec publish",
                                   "base_positions");
  check_block_spec_tensor_contract(block_tables,
                                   publish_device,
                                   cache_slot_dtype,
                                   "Block-Spec publish",
                                   "block_tables");
  check_block_spec_tensor_contract(destination.accepted_tokens,
                                   publish_device,
                                   torch::kLong,
                                   "Block-Spec publish",
                                   "destination.accepted_tokens");
  check_block_spec_tensor_contract(destination.accepted_lengths,
                                   publish_device,
                                   torch::kLong,
                                   "Block-Spec publish",
                                   "destination.accepted_lengths");
  check_block_spec_tensor_contract(destination.context_hidden,
                                   publish_device,
                                   hidden_dtype,
                                   "Block-Spec publish",
                                   "destination.context_hidden");
  check_block_spec_tensor_contract(destination.context_positions,
                                   publish_device,
                                   position_dtype,
                                   "Block-Spec publish",
                                   "destination.context_positions");
  check_block_spec_tensor_contract(destination.context_cache_slots,
                                   publish_device,
                                   cache_slot_dtype,
                                   "Block-Spec publish",
                                   "destination.context_cache_slots");
  check_block_spec_tensor_contract(destination.valid_tokens,
                                   publish_device,
                                   torch::kBool,
                                   "Block-Spec publish",
                                   "destination.valid_tokens");
  check_block_spec_tensor_contract(destination.tail_tokens,
                                   publish_device,
                                   torch::kLong,
                                   "Block-Spec publish",
                                   "destination.tail_tokens");
  check_block_spec_tensor_contract(destination.tail_context_hidden,
                                   publish_device,
                                   hidden_dtype,
                                   "Block-Spec publish",
                                   "destination.tail_context_hidden");
  check_block_spec_tensor_contract(destination.next_positions,
                                   publish_device,
                                   position_dtype,
                                   "Block-Spec publish",
                                   "destination.next_positions");
  check_block_spec_tensor_contract(destination.valid_rows,
                                   publish_device,
                                   torch::kBool,
                                   "Block-Spec publish",
                                   "destination.valid_rows");
  check_block_spec_tensor_contract(workspace.position_offsets,
                                   publish_device,
                                   position_dtype,
                                   "Block-Spec publish",
                                   "workspace.position_offsets");
  check_block_spec_tensor_contract(workspace.position_block_indices,
                                   publish_device,
                                   position_dtype,
                                   "Block-Spec publish",
                                   "workspace.position_block_indices");
  check_block_spec_tensor_contract(workspace.block_indices,
                                   publish_device,
                                   torch::kLong,
                                   "Block-Spec publish",
                                   "workspace.block_indices");
  check_block_spec_tensor_contract(workspace.block_ids,
                                   publish_device,
                                   cache_slot_dtype,
                                   "Block-Spec publish",
                                   "workspace.block_ids");
  check_block_spec_tensor_contract(workspace.cache_offsets,
                                   publish_device,
                                   cache_slot_dtype,
                                   "Block-Spec publish",
                                   "workspace.cache_offsets");
  check_block_spec_tensor_contract(workspace.invalid_tokens,
                                   publish_device,
                                   torch::kBool,
                                   "Block-Spec publish",
                                   "workspace.invalid_tokens");
  check_block_spec_tensor_contract(workspace.row_indices,
                                   publish_device,
                                   torch::kLong,
                                   "Block-Spec publish",
                                   "workspace.row_indices");
  check_block_spec_tensor_contract(workspace.last_sequence_indices,
                                   publish_device,
                                   torch::kLong,
                                   "Block-Spec publish",
                                   "workspace.last_sequence_indices");
  check_block_spec_tensor_contract(workspace.last_flat_indices,
                                   publish_device,
                                   torch::kLong,
                                   "Block-Spec publish",
                                   "workspace.last_flat_indices");

  const int64_t row_count = accepted_tokens.size(0);
  const int64_t accepted_width = accepted_tokens.size(1);
  const int64_t hidden_size = accepted_context_hidden.size(2);
  CHECK_GT(row_count, 0);
  CHECK_GT(accepted_width, 0);
  CHECK_EQ(accepted_context_hidden.size(0), row_count);
  CHECK_EQ(accepted_context_hidden.size(1), accepted_width);
  CHECK_EQ(base_positions.numel(), row_count);
  CHECK_EQ(block_tables.size(0), row_count);
  CHECK_LE(row_count, destination.accepted_tokens.size(0));
  CHECK_EQ(accepted_width, destination.accepted_tokens.size(1))
      << "Block-Spec accepted-token width must match the configured Target "
         "Verify capacity";
  CHECK_EQ(destination.context_hidden.size(2), hidden_size);
  CHECK_LE(row_count, workspace.position_block_indices.size(0));
  CHECK_EQ(accepted_width, workspace.position_block_indices.size(1));
  CHECK_GE(workspace.position_offsets.numel(), accepted_width);
  destination.accepted_tokens.fill_(-1);
  destination.accepted_lengths.zero_();
  destination.context_hidden.zero_();
  destination.context_positions.zero_();
  destination.context_cache_slots.zero_();
  destination.valid_tokens.zero_();
  destination.tail_tokens.fill_(-1);
  destination.tail_context_hidden.zero_();
  destination.next_positions.zero_();
  destination.valid_rows.zero_();

  torch::Tensor published_tokens =
      destination.accepted_tokens.narrow(0, 0, row_count)
          .narrow(1, 0, accepted_width);
  published_tokens.copy_(accepted_tokens, /*non_blocking=*/true);
  torch::Tensor valid_tokens = destination.valid_tokens.narrow(0, 0, row_count)
                                   .narrow(1, 0, accepted_width);
  torch::ge_out(valid_tokens, accepted_tokens, 0);
  torch::Tensor accepted_lengths =
      destination.accepted_lengths.narrow(0, 0, row_count);
  torch::sum_out(accepted_lengths,
                 valid_tokens,
                 /*dim=*/{1},
                 /*keepdim=*/false,
                 /*dtype=*/torch::kLong);
  torch::Tensor valid_rows = destination.valid_rows.narrow(0, 0, row_count);
  torch::gt_out(valid_rows, accepted_lengths, 0);

  torch::Tensor last_sequence_indices =
      workspace.last_sequence_indices.narrow(0, 0, row_count);
  torch::sub_out(last_sequence_indices, accepted_lengths, 1);
  last_sequence_indices.clamp_min_(0);
  torch::Tensor row_indices = workspace.row_indices.narrow(0, 0, row_count);
  torch::Tensor last_flat_indices =
      workspace.last_flat_indices.narrow(0, 0, row_count);
  torch::mul_out(last_flat_indices, row_indices, accepted_width);
  last_flat_indices.add_(last_sequence_indices);
  torch::Tensor tail_tokens = destination.tail_tokens.narrow(0, 0, row_count);
  torch::index_select_out(
      tail_tokens, accepted_tokens.flatten(), /*dim=*/0, last_flat_indices);

  torch::Tensor context_hidden =
      destination.context_hidden.narrow(0, 0, row_count)
          .narrow(1, 0, accepted_width);
  context_hidden.copy_(accepted_context_hidden, /*non_blocking=*/true);
  torch::Tensor tail_context_hidden =
      destination.tail_context_hidden.narrow(0, 0, row_count);
  torch::index_select_out(tail_context_hidden,
                          accepted_context_hidden.flatten(
                              /*start_dim=*/0, /*end_dim=*/1),
                          /*dim=*/0,
                          last_flat_indices);

  torch::Tensor context_positions =
      destination.context_positions.narrow(0, 0, row_count)
          .narrow(1, 0, accepted_width);
  context_positions.copy_(base_positions.view({row_count, 1}));
  context_positions.add_(workspace.position_offsets.narrow(0, 0, accepted_width)
                             .view({1, accepted_width}));
  torch::Tensor next_positions =
      destination.next_positions.narrow(0, 0, row_count);
  next_positions.copy_(base_positions, /*non_blocking=*/true);
  next_positions.add_(accepted_lengths);

  torch::Tensor context_cache_slots =
      destination.context_cache_slots.narrow(0, 0, row_count)
          .narrow(1, 0, accepted_width);
  map_cache_slots_2d_out(
      context_positions,
      block_tables,
      block_size,
      cache_slot_mapping_mode,
      context_cache_slots,
      workspace.position_block_indices.narrow(0, 0, row_count)
          .narrow(1, 0, accepted_width),
      workspace.block_indices.narrow(0, 0, row_count)
          .narrow(1, 0, accepted_width),
      workspace.block_ids.narrow(0, 0, row_count).narrow(1, 0, accepted_width),
      workspace.cache_offsets.narrow(0, 0, row_count)
          .narrow(1, 0, accepted_width));

  torch::Tensor invalid_tokens =
      workspace.invalid_tokens.narrow(0, 0, row_count)
          .narrow(1, 0, accepted_width);
  torch::logical_not_out(invalid_tokens, valid_tokens);
  context_positions.masked_fill_(invalid_tokens, 0);
  context_cache_slots.masked_fill_(invalid_tokens, 0);
}

void patch_block_spec_continuation_rows(
    const BlockSpecDeviceStepState& predecessor,
    const torch::Tensor& predecessor_rows,
    BlockSpecContinuationState& continuation,
    BlockSpecPredecessorPatchWorkspace& workspace) {
  CHECK(predecessor_rows.defined());
  CHECK_EQ(predecessor_rows.dim(), 1);
  CHECK_EQ(predecessor_rows.scalar_type(), torch::kLong);
  const int64_t row_count = predecessor_rows.numel();
  CHECK_GT(row_count, 0);
  CHECK(continuation.anchor_context_hidden.defined());
  CHECK(continuation.base_positions.defined());
  const torch::Device continuation_device = predecessor_rows.device();
  const torch::ScalarType hidden_dtype =
      continuation.anchor_context_hidden.scalar_type();
  const torch::ScalarType position_dtype =
      continuation.base_positions.scalar_type();
  check_block_spec_tensor_contract(predecessor.valid_rows,
                                   continuation_device,
                                   torch::kBool,
                                   "Block-Spec continuation",
                                   "predecessor.valid_rows");
  check_block_spec_tensor_contract(predecessor.tail_tokens,
                                   continuation_device,
                                   torch::kLong,
                                   "Block-Spec continuation",
                                   "predecessor.tail_tokens");
  check_block_spec_tensor_contract(predecessor.tail_context_hidden,
                                   continuation_device,
                                   hidden_dtype,
                                   "Block-Spec continuation",
                                   "predecessor.tail_context_hidden");
  check_block_spec_tensor_contract(predecessor.next_positions,
                                   continuation_device,
                                   position_dtype,
                                   "Block-Spec continuation",
                                   "predecessor.next_positions");
  check_block_spec_tensor_contract(continuation.anchor_tokens,
                                   continuation_device,
                                   torch::kLong,
                                   "Block-Spec continuation",
                                   "continuation.anchor_tokens");
  check_block_spec_tensor_contract(continuation.anchor_context_hidden,
                                   continuation_device,
                                   hidden_dtype,
                                   "Block-Spec continuation",
                                   "continuation.anchor_context_hidden");
  check_block_spec_tensor_contract(continuation.base_positions,
                                   continuation_device,
                                   position_dtype,
                                   "Block-Spec continuation",
                                   "continuation.base_positions");
  check_block_spec_tensor_contract(workspace.gather_rows,
                                   continuation_device,
                                   torch::kLong,
                                   "Block-Spec continuation",
                                   "workspace.gather_rows");
  check_block_spec_tensor_contract(workspace.gathered_valid_rows,
                                   continuation_device,
                                   torch::kBool,
                                   "Block-Spec continuation",
                                   "workspace.gathered_valid_rows");
  check_block_spec_tensor_contract(workspace.continuation_mask,
                                   continuation_device,
                                   torch::kBool,
                                   "Block-Spec continuation",
                                   "workspace.continuation_mask");
  check_block_spec_tensor_contract(workspace.gathered_tokens,
                                   continuation_device,
                                   torch::kLong,
                                   "Block-Spec continuation",
                                   "workspace.gathered_tokens");
  check_block_spec_tensor_contract(workspace.patched_tokens,
                                   continuation_device,
                                   torch::kLong,
                                   "Block-Spec continuation",
                                   "workspace.patched_tokens");
  check_block_spec_tensor_contract(workspace.gathered_context_hidden,
                                   continuation_device,
                                   hidden_dtype,
                                   "Block-Spec continuation",
                                   "workspace.gathered_context_hidden");
  check_block_spec_tensor_contract(workspace.patched_context_hidden,
                                   continuation_device,
                                   hidden_dtype,
                                   "Block-Spec continuation",
                                   "workspace.patched_context_hidden");
  check_block_spec_tensor_contract(workspace.gathered_positions,
                                   continuation_device,
                                   position_dtype,
                                   "Block-Spec continuation",
                                   "workspace.gathered_positions");
  check_block_spec_tensor_contract(workspace.patched_positions,
                                   continuation_device,
                                   position_dtype,
                                   "Block-Spec continuation",
                                   "workspace.patched_positions");
  CHECK_LE(row_count, predecessor.valid_rows.numel());
  CHECK_EQ(continuation.anchor_tokens.numel(), row_count);
  CHECK_EQ(continuation.anchor_context_hidden.dim(), 2);
  CHECK_EQ(continuation.anchor_context_hidden.size(0), row_count);
  CHECK_EQ(continuation.base_positions.numel(), row_count);
  CHECK_EQ(continuation.anchor_context_hidden.size(1),
           predecessor.tail_context_hidden.size(1));
  CHECK_LE(row_count, workspace.gather_rows.numel());
  CHECK_EQ(workspace.gathered_context_hidden.size(1),
           continuation.anchor_context_hidden.size(1));

  torch::Tensor gather_rows = workspace.gather_rows.narrow(0, 0, row_count);
  gather_rows.copy_(predecessor_rows, /*non_blocking=*/true);
  gather_rows.clamp_min_(0);
  torch::Tensor continuation_mask =
      workspace.continuation_mask.narrow(0, 0, row_count);
  torch::ge_out(continuation_mask, predecessor_rows, 0);
  torch::Tensor gathered_valid_rows =
      workspace.gathered_valid_rows.narrow(0, 0, row_count);
  torch::index_select_out(gathered_valid_rows,
                          predecessor.valid_rows,
                          /*dim=*/0,
                          gather_rows);
  torch::logical_and_out(
      continuation_mask, continuation_mask, gathered_valid_rows);

  torch::Tensor gathered_tokens =
      workspace.gathered_tokens.narrow(0, 0, row_count);
  torch::index_select_out(gathered_tokens,
                          predecessor.tail_tokens,
                          /*dim=*/0,
                          gather_rows);
  torch::Tensor patched_tokens =
      workspace.patched_tokens.narrow(0, 0, row_count);
  torch::where_out(patched_tokens,
                   continuation_mask,
                   gathered_tokens,
                   continuation.anchor_tokens);
  continuation.anchor_tokens.copy_(patched_tokens, /*non_blocking=*/true);

  torch::Tensor gathered_context_hidden =
      workspace.gathered_context_hidden.narrow(0, 0, row_count);
  torch::index_select_out(gathered_context_hidden,
                          predecessor.tail_context_hidden,
                          /*dim=*/0,
                          gather_rows);
  torch::Tensor patched_context_hidden =
      workspace.patched_context_hidden.narrow(0, 0, row_count);
  torch::where_out(patched_context_hidden,
                   continuation_mask.unsqueeze(/*dim=*/1),
                   gathered_context_hidden,
                   continuation.anchor_context_hidden);
  continuation.anchor_context_hidden.copy_(patched_context_hidden,
                                           /*non_blocking=*/true);

  torch::Tensor gathered_positions =
      workspace.gathered_positions.narrow(0, 0, row_count);
  torch::index_select_out(gathered_positions,
                          predecessor.next_positions,
                          /*dim=*/0,
                          gather_rows);
  torch::Tensor patched_positions =
      workspace.patched_positions.narrow(0, 0, row_count);
  torch::where_out(patched_positions,
                   continuation_mask,
                   gathered_positions,
                   continuation.base_positions);
  continuation.base_positions.copy_(patched_positions,
                                    /*non_blocking=*/true);
}

void patch_block_spec_decode_input_geometry(
    const BlockSpecContinuationState& continuation,
    int32_t block_size,
    BlockSpecDecodeInputPatchTarget& target,
    BlockSpecDecodeInputPatchWorkspace& workspace) {
  CHECK_GT(block_size, 0);
  CHECK(continuation.anchor_tokens.defined());
  CHECK(continuation.base_positions.defined());
  CHECK_EQ(continuation.anchor_tokens.dim(), 1);
  CHECK_EQ(continuation.base_positions.dim(), 1);
  CHECK(target.source_block_tables.defined());
  const torch::Device geometry_device = continuation.anchor_tokens.device();
  const torch::ScalarType position_dtype =
      continuation.base_positions.scalar_type();
  const torch::ScalarType cache_slot_dtype =
      target.source_block_tables.scalar_type();
  check_block_spec_tensor_contract(continuation.anchor_tokens,
                                   geometry_device,
                                   torch::kLong,
                                   "Block-Spec decode geometry",
                                   "continuation.anchor_tokens");
  check_block_spec_tensor_contract(continuation.base_positions,
                                   geometry_device,
                                   position_dtype,
                                   "Block-Spec decode geometry",
                                   "continuation.base_positions");
  check_block_spec_tensor_contract(target.query_token_ids,
                                   geometry_device,
                                   torch::kLong,
                                   "Block-Spec decode geometry",
                                   "target.query_token_ids");
  check_block_spec_tensor_contract(target.query_positions,
                                   geometry_device,
                                   position_dtype,
                                   "Block-Spec decode geometry",
                                   "target.query_positions");
  check_block_spec_tensor_contract(target.query_kv_seq_lens,
                                   geometry_device,
                                   position_dtype,
                                   "Block-Spec decode geometry",
                                   "target.query_kv_seq_lens");
  check_block_spec_tensor_contract(target.query_new_cache_slots,
                                   geometry_device,
                                   cache_slot_dtype,
                                   "Block-Spec decode geometry",
                                   "target.query_new_cache_slots");
  check_block_spec_tensor_contract(target.target_token_ids,
                                   geometry_device,
                                   torch::kLong,
                                   "Block-Spec decode geometry",
                                   "target.target_token_ids");
  check_block_spec_tensor_contract(target.target_positions,
                                   geometry_device,
                                   position_dtype,
                                   "Block-Spec decode geometry",
                                   "target.target_positions");
  check_block_spec_tensor_contract(target.target_kv_seq_lens,
                                   geometry_device,
                                   position_dtype,
                                   "Block-Spec decode geometry",
                                   "target.target_kv_seq_lens");
  check_block_spec_tensor_contract(target.target_new_cache_slots,
                                   geometry_device,
                                   cache_slot_dtype,
                                   "Block-Spec decode geometry",
                                   "target.target_new_cache_slots");
  check_block_spec_tensor_contract(target.source_block_tables,
                                   geometry_device,
                                   cache_slot_dtype,
                                   "Block-Spec decode geometry",
                                   "target.source_block_tables");
  check_block_spec_tensor_contract(workspace.query_position_offsets,
                                   geometry_device,
                                   position_dtype,
                                   "Block-Spec decode geometry",
                                   "workspace.query_position_offsets");
  check_block_spec_tensor_contract(workspace.query_position_block_indices,
                                   geometry_device,
                                   position_dtype,
                                   "Block-Spec decode geometry",
                                   "workspace.query_position_block_indices");
  check_block_spec_tensor_contract(workspace.query_block_indices,
                                   geometry_device,
                                   torch::kLong,
                                   "Block-Spec decode geometry",
                                   "workspace.query_block_indices");
  check_block_spec_tensor_contract(workspace.query_block_ids,
                                   geometry_device,
                                   cache_slot_dtype,
                                   "Block-Spec decode geometry",
                                   "workspace.query_block_ids");
  check_block_spec_tensor_contract(workspace.query_cache_offsets,
                                   geometry_device,
                                   cache_slot_dtype,
                                   "Block-Spec decode geometry",
                                   "workspace.query_cache_offsets");
  check_block_spec_tensor_contract(workspace.target_position_offsets,
                                   geometry_device,
                                   position_dtype,
                                   "Block-Spec decode geometry",
                                   "workspace.target_position_offsets");
  check_block_spec_tensor_contract(workspace.target_position_block_indices,
                                   geometry_device,
                                   position_dtype,
                                   "Block-Spec decode geometry",
                                   "workspace.target_position_block_indices");
  check_block_spec_tensor_contract(workspace.target_block_indices,
                                   geometry_device,
                                   torch::kLong,
                                   "Block-Spec decode geometry",
                                   "workspace.target_block_indices");
  check_block_spec_tensor_contract(workspace.target_block_ids,
                                   geometry_device,
                                   cache_slot_dtype,
                                   "Block-Spec decode geometry",
                                   "workspace.target_block_ids");
  check_block_spec_tensor_contract(workspace.target_cache_offsets,
                                   geometry_device,
                                   cache_slot_dtype,
                                   "Block-Spec decode geometry",
                                   "workspace.target_cache_offsets");
  const int64_t row_count = continuation.anchor_tokens.numel();
  CHECK_GT(row_count, 0);
  CHECK_EQ(continuation.base_positions.numel(), row_count);
  CHECK_EQ(target.query_token_ids.dim(), 2);
  CHECK_EQ(target.query_token_ids.size(0), row_count);
  CHECK_EQ(target.query_positions.sizes(), target.query_token_ids.sizes());
  CHECK_EQ(target.query_new_cache_slots.sizes(),
           target.query_positions.sizes());
  CHECK_EQ(target.target_token_ids.dim(), 2);
  CHECK_EQ(target.target_token_ids.size(0), row_count);
  CHECK_EQ(target.target_positions.sizes(), target.target_token_ids.sizes());
  CHECK_EQ(target.target_new_cache_slots.sizes(),
           target.target_positions.sizes());
  CHECK_EQ(target.source_block_tables.dim(), 2);
  CHECK_EQ(target.source_block_tables.size(0), row_count);
  const int64_t query_width = target.query_token_ids.size(1);
  const int64_t target_width = target.target_token_ids.size(1);
  CHECK_GE(workspace.query_position_offsets.numel(), query_width);
  CHECK_GE(workspace.target_position_offsets.numel(), target_width);

  target.query_token_ids.select(/*dim=*/1, /*index=*/0)
      .copy_(continuation.anchor_tokens, /*non_blocking=*/true);
  target.target_token_ids.select(/*dim=*/1, /*index=*/0)
      .copy_(continuation.anchor_tokens, /*non_blocking=*/true);

  target.query_positions.copy_(
      continuation.base_positions.view({row_count, 1}));
  target.query_positions.add_(
      workspace.query_position_offsets.narrow(0, 0, query_width)
          .view({1, query_width}));
  target.target_positions.copy_(
      continuation.base_positions.view({row_count, 1}));
  target.target_positions.add_(
      workspace.target_position_offsets.narrow(0, 0, target_width)
          .view({1, target_width}));

  CHECK_EQ(target.query_kv_seq_lens.numel(), row_count)
      << "Block-Spec Query expects one KV length per logical sequence";
  target.query_kv_seq_lens.copy_(continuation.base_positions,
                                 /*non_blocking=*/true);
  target.query_kv_seq_lens.add_(query_width);
  if (target.target_kv_seq_lens.numel() == row_count) {
    target.target_kv_seq_lens.copy_(continuation.base_positions,
                                    /*non_blocking=*/true);
    target.target_kv_seq_lens.add_(target_width);
  } else {
    CHECK_EQ(target.target_kv_seq_lens.numel(), row_count * target_width)
        << "Block-Spec Target KV layout must be chunked or tokenwise";
    target.target_kv_seq_lens.view({row_count, target_width})
        .copy_(target.target_positions, /*non_blocking=*/true);
    target.target_kv_seq_lens.add_(1);
  }

  map_cache_slots_2d_out(
      target.query_positions,
      target.source_block_tables,
      block_size,
      target.cache_slot_mapping_mode,
      target.query_new_cache_slots,
      workspace.query_position_block_indices.narrow(0, 0, row_count)
          .narrow(1, 0, query_width),
      workspace.query_block_indices.narrow(0, 0, row_count)
          .narrow(1, 0, query_width),
      workspace.query_block_ids.narrow(0, 0, row_count)
          .narrow(1, 0, query_width),
      workspace.query_cache_offsets.narrow(0, 0, row_count)
          .narrow(1, 0, query_width));
  map_cache_slots_2d_out(
      target.target_positions,
      target.source_block_tables,
      block_size,
      target.cache_slot_mapping_mode,
      target.target_new_cache_slots,
      workspace.target_position_block_indices.narrow(0, 0, row_count)
          .narrow(1, 0, target_width),
      workspace.target_block_indices.narrow(0, 0, row_count)
          .narrow(1, 0, target_width),
      workspace.target_block_ids.narrow(0, 0, row_count)
          .narrow(1, 0, target_width),
      workspace.target_cache_offsets.narrow(0, 0, row_count)
          .narrow(1, 0, target_width));
}

void patch_block_spec_target_token_ids(const torch::Tensor& draft_token_ids,
                                       torch::Tensor target_token_ids) {
  CHECK(draft_token_ids.defined());
  CHECK(target_token_ids.defined());
  const torch::Device patch_device = draft_token_ids.device();
  check_block_spec_tensor_contract(draft_token_ids,
                                   patch_device,
                                   torch::kLong,
                                   "Block-Spec target-token patch",
                                   "draft_token_ids");
  check_block_spec_tensor_contract(target_token_ids,
                                   patch_device,
                                   torch::kLong,
                                   "Block-Spec target-token patch",
                                   "target_token_ids");
  CHECK_EQ(draft_token_ids.dim(), 2);
  CHECK_EQ(target_token_ids.dim(), 2);
  CHECK_EQ(draft_token_ids.size(0), target_token_ids.size(0));
  CHECK_EQ(draft_token_ids.size(1) + 1, target_token_ids.size(1));
  target_token_ids
      .narrow(
          /*dim=*/1, /*start=*/1, draft_token_ids.size(1))
      .copy_(draft_token_ids, /*non_blocking=*/true);
}

}  // namespace xllm::block_spec_async
