/* Copyright 2026 The xLLM Authors.

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

#include "core/framework/speculative/mtp_async_state.h"

#include <glog/logging.h>

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "core/framework/model/model_args.h"

namespace xllm::mtp_async {
namespace {

torch::Tensor gather_sequence_rows(const torch::Tensor& values,
                                   const torch::Tensor& indices) {
  CHECK_GE(values.dim(), 2);
  CHECK_EQ(values.size(0), indices.numel());
  torch::Tensor gather_index =
      indices.to(torch::dtype(torch::kLong).device(indices.device()))
          .view({-1, 1});
  for (int64_t dim = 2; dim < values.dim(); ++dim) {
    gather_index = gather_index.unsqueeze(-1);
  }
  std::vector<int64_t> expanded_shape = values.sizes().vec();
  expanded_shape[1] = 1;
  gather_index = gather_index.expand(expanded_shape);
  return values.gather(/*dim=*/1, gather_index).squeeze(/*dim=*/1);
}

void check_mtp_tensor_contract(const torch::Tensor& tensor,
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

void check_publish_tensor_contract(const torch::Tensor& tensor,
                                   const torch::Device& expected_device,
                                   torch::ScalarType expected_dtype,
                                   const char* tensor_name) {
  check_mtp_tensor_contract(
      tensor, expected_device, expected_dtype, "MTP publish", tensor_name);
}

}  // namespace

TargetSpecVerifyMode classify_target_spec_verify_mode(
    std::string_view model_type) {
  if (is_qwen3_5_target_model_type(model_type)) {
    return TargetSpecVerifyMode::QWEN3_5_EXPANDED_VERIFY;
  }
  if (model_type == "deepseek_v32") {
    return TargetSpecVerifyMode::DEEPSEEK_V32_EXPANDED_VERIFY;
  }
  if (model_type == "mimo") {
    return TargetSpecVerifyMode::CAUSAL_CHUNKED_PREFILL;
  }
  return TargetSpecVerifyMode::GENERIC;
}

int64_t speculative_verify_block_table_capacity(int64_t max_position_embeddings,
                                                int64_t block_size) {
  CHECK_GT(max_position_embeddings, 0);
  CHECK_GT(block_size, 0);
  return (max_position_embeddings + block_size - 1) / block_size + 1;
}

CombinedDraftExecutionPath classify_combined_draft_execution_path(
    std::string_view model_type) {
  if (model_type == "qwen3_5_mtp" || model_type == "qwen3_5_moe_mtp") {
    return CombinedDraftExecutionPath::QWEN3_5_PAGED_ATTENTION;
  }
  if (model_type == "glm_moe_dsa_mtp") {
    return CombinedDraftExecutionPath::GLM_MOE_DSA_SPARSE_ATTENTION;
  }
  return CombinedDraftExecutionPath::UNSUPPORTED;
}

bool supports_combined_draft_configuration(
    CombinedDraftExecutionPath execution_path,
    std::string_view npu_backend,
    int32_t dp_size) {
  switch (execution_path) {
    case CombinedDraftExecutionPath::QWEN3_5_PAGED_ATTENTION:
      return npu_backend == "TORCH" && dp_size <= 1;
    case CombinedDraftExecutionPath::GLM_MOE_DSA_SPARSE_ATTENTION:
      return npu_backend == "ATB";
    case CombinedDraftExecutionPath::UNSUPPORTED:
      return false;
  }
  return false;
}

std::vector<int64_t> build_predecessor_rows(
    const std::vector<std::string>& previous_request_ids,
    const std::vector<std::string>& current_request_ids) {
  std::unordered_map<std::string, int64_t> previous_rows;
  previous_rows.reserve(previous_request_ids.size());
  for (size_t row = 0; row < previous_request_ids.size(); ++row) {
    auto insert_result = previous_rows.emplace(previous_request_ids[row],
                                               static_cast<int64_t>(row));
    CHECK(insert_result.second) << "Duplicate request id in predecessor Task: "
                                << previous_request_ids[row];
  }

  std::unordered_set<std::string> current_ids;
  current_ids.reserve(current_request_ids.size());
  std::vector<int64_t> predecessor_rows;
  predecessor_rows.reserve(current_request_ids.size());
  for (const std::string& request_id : current_request_ids) {
    CHECK(current_ids.emplace(request_id).second)
        << "Duplicate request id in current Task: " << request_id;
    const auto previous = previous_rows.find(request_id);
    predecessor_rows.emplace_back(
        previous == previous_rows.end() ? -1 : previous->second);
  }
  return predecessor_rows;
}

MtpDeviceStepState allocate_mtp_device_step_state(
    int64_t max_rows,
    int64_t accepted_token_capacity,
    int64_t embedding_size,
    const torch::TensorOptions& token_options,
    const torch::TensorOptions& embedding_options,
    const torch::TensorOptions& position_options) {
  CHECK_GT(max_rows, 0);
  CHECK_GT(accepted_token_capacity, 0);
  CHECK_GT(embedding_size, 0);
  MtpDeviceStepState state;
  state.accepted_tokens =
      torch::empty({max_rows, accepted_token_capacity}, token_options);
  state.accepted_lengths = torch::empty({max_rows}, token_options);
  state.tail_tokens = torch::empty({max_rows}, token_options);
  state.previous_tokens = torch::empty({max_rows}, token_options);
  state.tail_embeddings =
      torch::empty({max_rows, embedding_size}, embedding_options);
  state.previous_embeddings =
      torch::empty({max_rows, embedding_size}, embedding_options);
  state.base_positions = torch::empty({max_rows}, position_options);
  state.base_kv_seq_lens = torch::empty({max_rows}, position_options);
  state.valid_rows =
      torch::zeros({max_rows}, token_options.dtype(torch::kBool));
  return state;
}

MtpPredecessorPatchWorkspace allocate_mtp_predecessor_patch_workspace(
    int64_t max_rows,
    int64_t embedding_size,
    const torch::TensorOptions& token_options,
    const torch::TensorOptions& embedding_options,
    const torch::TensorOptions& position_options) {
  CHECK_GT(max_rows, 0);
  CHECK_GT(embedding_size, 0);
  MtpPredecessorPatchWorkspace workspace;
  workspace.gather_rows =
      torch::empty({max_rows}, token_options.dtype(torch::kLong));
  workspace.gathered_valid_rows =
      torch::empty({max_rows}, token_options.dtype(torch::kBool));
  workspace.continuation_mask =
      torch::empty({max_rows}, token_options.dtype(torch::kBool));
  workspace.gathered_tokens = torch::empty({max_rows}, token_options);
  workspace.patched_tokens = torch::empty({max_rows}, token_options);
  workspace.gathered_previous_tokens = torch::empty({max_rows}, token_options);
  workspace.patched_previous_tokens = torch::empty({max_rows}, token_options);
  workspace.gathered_embeddings =
      torch::empty({max_rows, embedding_size}, embedding_options);
  workspace.patched_embeddings =
      torch::empty({max_rows, embedding_size}, embedding_options);
  workspace.gathered_previous_embeddings =
      torch::empty({max_rows, embedding_size}, embedding_options);
  workspace.patched_previous_embeddings =
      torch::empty({max_rows, embedding_size}, embedding_options);
  workspace.gathered_positions = torch::empty({max_rows}, position_options);
  workspace.patched_positions = torch::empty({max_rows}, position_options);
  workspace.gathered_kv_seq_lens = torch::empty({max_rows}, position_options);
  workspace.patched_kv_seq_lens = torch::empty({max_rows}, position_options);
  return workspace;
}

MtpPublishPatchWorkspace allocate_mtp_publish_patch_workspace(
    int64_t max_rows,
    int64_t accepted_token_capacity,
    int64_t embedding_size,
    const torch::TensorOptions& token_options,
    const torch::TensorOptions& embedding_options) {
  CHECK_GT(max_rows, 0);
  CHECK_GT(accepted_token_capacity, 0);
  CHECK_GT(embedding_size, 0);
  const torch::TensorOptions long_options = token_options.dtype(torch::kLong);
  const torch::TensorOptions bool_options = token_options.dtype(torch::kBool);
  MtpPublishPatchWorkspace workspace;
  workspace.accepted_mask =
      torch::empty({max_rows, accepted_token_capacity}, bool_options);
  workspace.row_indices = torch::arange(max_rows, long_options);
  workspace.last_sequence_indices = torch::empty({max_rows}, long_options);
  workspace.previous_sequence_indices = torch::empty({max_rows}, long_options);
  workspace.last_flat_indices = torch::empty({max_rows}, long_options);
  workspace.previous_flat_indices = torch::empty({max_rows}, long_options);
  workspace.has_previous = torch::empty({max_rows}, bool_options);
  workspace.gathered_previous_tokens = torch::empty({max_rows}, token_options);
  workspace.gathered_previous_embeddings =
      torch::empty({max_rows, embedding_size}, embedding_options);
  return workspace;
}

MtpNextDraftPatchWorkspace allocate_mtp_next_draft_patch_workspace(
    int64_t max_rows,
    const torch::TensorOptions& position_options,
    const torch::TensorOptions& cache_slot_options) {
  CHECK_GT(max_rows, 0);
  CHECK_EQ(position_options.device(), cache_slot_options.device());
  MtpNextDraftPatchWorkspace workspace;
  workspace.position_block_indices = torch::empty({max_rows}, position_options);
  workspace.block_indices =
      torch::empty({max_rows}, position_options.dtype(torch::kLong));
  workspace.block_ids = torch::empty({max_rows}, cache_slot_options);
  workspace.cache_offsets = torch::empty({max_rows}, cache_slot_options);
  return workspace;
}

MtpTargetVerifyPatchWorkspace allocate_mtp_target_verify_patch_workspace(
    int64_t max_rows,
    int64_t max_verify_width,
    int64_t max_block_table_width,
    const torch::TensorOptions& position_options,
    const torch::TensorOptions& cache_slot_options) {
  CHECK_GT(max_rows, 0);
  CHECK_GT(max_verify_width, 0);
  CHECK_GT(max_block_table_width, 0);
  CHECK_EQ(position_options.device(), cache_slot_options.device());
  MtpTargetVerifyPatchWorkspace workspace;
  workspace.position_offsets =
      torch::arange(max_verify_width, position_options);
  workspace.row_positions =
      torch::empty({max_rows, max_verify_width}, position_options);
  workspace.row_kv_seq_lens =
      torch::empty({max_rows, max_verify_width}, position_options);
  workspace.position_block_indices =
      torch::empty({max_rows, max_verify_width}, position_options);
  workspace.block_indices = torch::empty({max_rows, max_verify_width},
                                         position_options.dtype(torch::kLong));
  workspace.block_ids =
      torch::empty({max_rows, max_verify_width}, cache_slot_options);
  workspace.cache_offsets =
      torch::empty({max_rows, max_verify_width}, cache_slot_options);
  const int64_t max_expanded_rows = max_rows * max_verify_width;
  const int64_t max_page_candidates = max_expanded_rows * max_block_table_width;
  workspace.page_counts = torch::empty({max_expanded_rows}, position_options);
  workspace.page_column_indices =
      torch::arange(max_block_table_width, position_options)
          .unsqueeze(/*dim=*/0)
          .expand({max_expanded_rows, max_block_table_width})
          .contiguous();
  workspace.valid_pages =
      torch::empty({max_page_candidates}, position_options.dtype(torch::kBool));
  workspace.valid_page_ranks =
      torch::empty({max_page_candidates}, position_options.dtype(torch::kLong));
  workspace.invalid_page_destinations =
      torch::empty({max_page_candidates}, position_options.dtype(torch::kLong));
  workspace.page_destinations =
      torch::empty({max_page_candidates}, position_options.dtype(torch::kLong));
  return workspace;
}

namespace {

void map_cache_slots_out(const torch::Tensor& positions,
                         const torch::Tensor& block_tables,
                         int32_t block_size,
                         torch::Tensor& destination,
                         torch::Tensor& position_block_indices_storage,
                         torch::Tensor& block_indices_storage,
                         torch::Tensor& block_ids_storage,
                         torch::Tensor& cache_offsets_storage) {
  CHECK_GT(block_size, 0);
  CHECK(positions.defined());
  CHECK(block_tables.defined());
  CHECK(destination.defined());
  CHECK_EQ(positions.dim(), 1);
  CHECK_EQ(block_tables.dim(), 2);
  CHECK_EQ(block_tables.size(0), positions.numel());
  CHECK_EQ(destination.dim(), 1);
  CHECK_EQ(destination.numel(), positions.numel());
  CHECK_EQ(block_tables.scalar_type(), destination.scalar_type());

  const int64_t row_count = positions.numel();
  torch::Tensor position_block_indices =
      position_block_indices_storage.flatten().narrow(0, 0, row_count);
  torch::Tensor block_indices =
      block_indices_storage.flatten().narrow(0, 0, row_count);
  torch::Tensor block_ids = block_ids_storage.flatten().narrow(0, 0, row_count);
  torch::Tensor cache_offsets =
      cache_offsets_storage.flatten().narrow(0, 0, row_count);
  CHECK_EQ(block_indices.scalar_type(), torch::kLong);
  CHECK_EQ(position_block_indices.scalar_type(), positions.scalar_type());
  CHECK_EQ(block_ids.scalar_type(), block_tables.scalar_type());
  CHECK_EQ(cache_offsets.scalar_type(), destination.scalar_type());

  torch::floor_divide_out(position_block_indices, positions, block_size);
  block_indices.copy_(position_block_indices, /*non_blocking=*/true);
  torch::Tensor block_ids_rows = block_ids.unsqueeze(/*dim=*/1);
  torch::gather_out(block_ids_rows,
                    block_tables,
                    /*dim=*/1,
                    block_indices.unsqueeze(/*dim=*/1));
  torch::remainder_out(cache_offsets, positions, block_size);
  torch::mul_out(destination, block_ids, block_size);
  destination.add_(cache_offsets);
}

}  // namespace

void patch_mtp_next_draft_input(const torch::Tensor& draft_tokens,
                                const torch::Tensor& draft_embeddings,
                                const torch::Tensor& base_positions,
                                const torch::Tensor& base_kv_seq_lens,
                                int32_t position_offset,
                                int32_t block_size,
                                MtpNextDraftPatchTarget& target,
                                MtpNextDraftPatchWorkspace& workspace) {
  CHECK_GT(position_offset, 0);
  CHECK(draft_tokens.defined());
  CHECK(draft_embeddings.defined());
  CHECK(base_positions.defined());
  CHECK(base_kv_seq_lens.defined());
  CHECK(target.new_cache_slots.defined())
      << "MTP next-draft patch target.new_cache_slots must be defined";
  const torch::Device patch_device = draft_tokens.device();
  const torch::ScalarType token_dtype = draft_tokens.scalar_type();
  const torch::ScalarType embedding_dtype = draft_embeddings.scalar_type();
  const torch::ScalarType position_dtype = base_positions.scalar_type();
  const torch::ScalarType cache_slot_dtype =
      target.new_cache_slots.scalar_type();
  check_mtp_tensor_contract(draft_tokens,
                            patch_device,
                            token_dtype,
                            "MTP next-draft patch",
                            "draft_tokens");
  check_mtp_tensor_contract(draft_embeddings,
                            patch_device,
                            embedding_dtype,
                            "MTP next-draft patch",
                            "draft_embeddings");
  check_mtp_tensor_contract(base_positions,
                            patch_device,
                            position_dtype,
                            "MTP next-draft patch",
                            "base_positions");
  check_mtp_tensor_contract(base_kv_seq_lens,
                            patch_device,
                            position_dtype,
                            "MTP next-draft patch",
                            "base_kv_seq_lens");
  check_mtp_tensor_contract(target.token_ids,
                            patch_device,
                            token_dtype,
                            "MTP next-draft patch",
                            "target.token_ids");
  check_mtp_tensor_contract(target.input_embeddings,
                            patch_device,
                            embedding_dtype,
                            "MTP next-draft patch",
                            "target.input_embeddings");
  check_mtp_tensor_contract(target.positions,
                            patch_device,
                            position_dtype,
                            "MTP next-draft patch",
                            "target.positions");
  check_mtp_tensor_contract(target.kv_seq_lens,
                            patch_device,
                            position_dtype,
                            "MTP next-draft patch",
                            "target.kv_seq_lens");
  check_mtp_tensor_contract(target.new_cache_slots,
                            patch_device,
                            cache_slot_dtype,
                            "MTP next-draft patch",
                            "target.new_cache_slots");
  if (!target.model_managed_multiblock) {
    check_mtp_tensor_contract(target.block_tables,
                              patch_device,
                              cache_slot_dtype,
                              "MTP next-draft patch",
                              "target.block_tables");
  }
  check_mtp_tensor_contract(workspace.position_block_indices,
                            patch_device,
                            position_dtype,
                            "MTP next-draft patch",
                            "workspace.position_block_indices");
  check_mtp_tensor_contract(workspace.block_indices,
                            patch_device,
                            torch::kLong,
                            "MTP next-draft patch",
                            "workspace.block_indices");
  check_mtp_tensor_contract(workspace.block_ids,
                            patch_device,
                            cache_slot_dtype,
                            "MTP next-draft patch",
                            "workspace.block_ids");
  check_mtp_tensor_contract(workspace.cache_offsets,
                            patch_device,
                            cache_slot_dtype,
                            "MTP next-draft patch",
                            "workspace.cache_offsets");
  const int64_t row_count = base_positions.numel();
  CHECK_GT(row_count, 0);
  CHECK_EQ(base_positions.dim(), 1);
  CHECK_EQ(base_kv_seq_lens.dim(), 1);
  CHECK_EQ(base_kv_seq_lens.numel(), row_count);
  CHECK_EQ(draft_tokens.numel(), row_count);
  CHECK_EQ(draft_embeddings.dim(), 2);
  CHECK_EQ(draft_embeddings.size(0), row_count);
  CHECK_EQ(target.token_ids.numel(), row_count);
  CHECK_EQ(target.input_embeddings.dim(), 2);
  CHECK_EQ(target.input_embeddings.sizes(), draft_embeddings.sizes());
  CHECK_EQ(target.positions.numel(), row_count);
  CHECK_EQ(target.kv_seq_lens.numel(), row_count);
  CHECK_EQ(target.new_cache_slots.numel(), row_count);
  if (!target.model_managed_multiblock) {
    CHECK_EQ(target.block_tables.size(0), row_count);
  }

  target.token_ids.flatten().copy_(draft_tokens.flatten(),
                                   /*non_blocking=*/true);
  target.input_embeddings.copy_(draft_embeddings, /*non_blocking=*/true);
  target.positions.flatten().copy_(base_positions, /*non_blocking=*/true);
  target.positions.add_(position_offset);
  target.kv_seq_lens.flatten().copy_(base_kv_seq_lens,
                                     /*non_blocking=*/true);
  target.kv_seq_lens.add_(position_offset);
  torch::Tensor flat_positions = target.positions.flatten();
  torch::Tensor flat_cache_slots = target.new_cache_slots.flatten();
  if (target.model_managed_multiblock) {
    flat_cache_slots.zero_();
  } else {
    map_cache_slots_out(flat_positions,
                        target.block_tables,
                        block_size,
                        flat_cache_slots,
                        workspace.position_block_indices,
                        workspace.block_indices,
                        workspace.block_ids,
                        workspace.cache_offsets);
  }
}

void patch_mtp_target_verify_input(
    const torch::Tensor& continuation_tokens,
    const std::vector<torch::Tensor>& draft_token_sources,
    const torch::Tensor& base_positions,
    const torch::Tensor& base_kv_seq_lens,
    int32_t block_size,
    MtpTargetVerifyPatchTarget& target,
    MtpTargetVerifyPatchWorkspace& workspace) {
  CHECK(continuation_tokens.defined());
  CHECK(base_positions.defined());
  CHECK(base_kv_seq_lens.defined());
  CHECK(target.new_cache_slots.defined())
      << "MTP target-verify patch target.new_cache_slots must be defined";
  const torch::Device patch_device = continuation_tokens.device();
  const torch::ScalarType token_dtype = continuation_tokens.scalar_type();
  const torch::ScalarType position_dtype = base_positions.scalar_type();
  const torch::ScalarType cache_slot_dtype =
      target.new_cache_slots.scalar_type();
  check_mtp_tensor_contract(continuation_tokens,
                            patch_device,
                            token_dtype,
                            "MTP target-verify patch",
                            "continuation_tokens");
  check_mtp_tensor_contract(base_positions,
                            patch_device,
                            position_dtype,
                            "MTP target-verify patch",
                            "base_positions");
  check_mtp_tensor_contract(base_kv_seq_lens,
                            patch_device,
                            position_dtype,
                            "MTP target-verify patch",
                            "base_kv_seq_lens");
  check_mtp_tensor_contract(target.token_ids,
                            patch_device,
                            token_dtype,
                            "MTP target-verify patch",
                            "target.token_ids");
  check_mtp_tensor_contract(target.positions,
                            patch_device,
                            position_dtype,
                            "MTP target-verify patch",
                            "target.positions");
  check_mtp_tensor_contract(target.kv_seq_lens,
                            patch_device,
                            position_dtype,
                            "MTP target-verify patch",
                            "target.kv_seq_lens");
  check_mtp_tensor_contract(target.new_cache_slots,
                            patch_device,
                            cache_slot_dtype,
                            "MTP target-verify patch",
                            "target.new_cache_slots");
  const bool has_expanded_paged_metadata =
      target.expanded_paged_kv_indptr.defined();
  CHECK_EQ(target.expanded_paged_kv_indices.defined(),
           has_expanded_paged_metadata)
      << "MTP target-verify patch expanded paged metadata must be complete";
  CHECK_EQ(target.expanded_paged_kv_last_page_len.defined(),
           has_expanded_paged_metadata)
      << "MTP target-verify patch expanded paged metadata must be complete";
  if (!target.model_managed_multiblock || has_expanded_paged_metadata) {
    check_mtp_tensor_contract(target.block_tables,
                              patch_device,
                              cache_slot_dtype,
                              "MTP target-verify patch",
                              "target.block_tables");
  }
  if (target.expanded_kv_seq_lens.defined()) {
    check_mtp_tensor_contract(target.expanded_kv_seq_lens,
                              patch_device,
                              position_dtype,
                              "MTP target-verify patch",
                              "target.expanded_kv_seq_lens");
  }
  if (has_expanded_paged_metadata) {
    CHECK(target.expanded_kv_seq_lens.defined())
        << "MTP target-verify patch expanded KV lengths must be defined";
    check_mtp_tensor_contract(target.expanded_paged_kv_indptr,
                              patch_device,
                              position_dtype,
                              "MTP target-verify patch",
                              "target.expanded_paged_kv_indptr");
    check_mtp_tensor_contract(target.expanded_paged_kv_indices,
                              patch_device,
                              cache_slot_dtype,
                              "MTP target-verify patch",
                              "target.expanded_paged_kv_indices");
    check_mtp_tensor_contract(target.expanded_paged_kv_last_page_len,
                              patch_device,
                              position_dtype,
                              "MTP target-verify patch",
                              "target.expanded_paged_kv_last_page_len");
  }
  check_mtp_tensor_contract(workspace.position_offsets,
                            patch_device,
                            position_dtype,
                            "MTP target-verify patch",
                            "workspace.position_offsets");
  check_mtp_tensor_contract(workspace.row_positions,
                            patch_device,
                            position_dtype,
                            "MTP target-verify patch",
                            "workspace.row_positions");
  check_mtp_tensor_contract(workspace.row_kv_seq_lens,
                            patch_device,
                            position_dtype,
                            "MTP target-verify patch",
                            "workspace.row_kv_seq_lens");
  check_mtp_tensor_contract(workspace.position_block_indices,
                            patch_device,
                            position_dtype,
                            "MTP target-verify patch",
                            "workspace.position_block_indices");
  check_mtp_tensor_contract(workspace.block_indices,
                            patch_device,
                            torch::kLong,
                            "MTP target-verify patch",
                            "workspace.block_indices");
  check_mtp_tensor_contract(workspace.block_ids,
                            patch_device,
                            cache_slot_dtype,
                            "MTP target-verify patch",
                            "workspace.block_ids");
  check_mtp_tensor_contract(workspace.cache_offsets,
                            patch_device,
                            cache_slot_dtype,
                            "MTP target-verify patch",
                            "workspace.cache_offsets");
  check_mtp_tensor_contract(workspace.page_counts,
                            patch_device,
                            position_dtype,
                            "MTP target-verify patch",
                            "workspace.page_counts");
  check_mtp_tensor_contract(workspace.page_column_indices,
                            patch_device,
                            position_dtype,
                            "MTP target-verify patch",
                            "workspace.page_column_indices");
  check_mtp_tensor_contract(workspace.valid_pages,
                            patch_device,
                            torch::kBool,
                            "MTP target-verify patch",
                            "workspace.valid_pages");
  check_mtp_tensor_contract(workspace.valid_page_ranks,
                            patch_device,
                            torch::kLong,
                            "MTP target-verify patch",
                            "workspace.valid_page_ranks");
  check_mtp_tensor_contract(workspace.invalid_page_destinations,
                            patch_device,
                            torch::kLong,
                            "MTP target-verify patch",
                            "workspace.invalid_page_destinations");
  check_mtp_tensor_contract(workspace.page_destinations,
                            patch_device,
                            torch::kLong,
                            "MTP target-verify patch",
                            "workspace.page_destinations");
  for (const torch::Tensor& draft_tokens : draft_token_sources) {
    check_mtp_tensor_contract(draft_tokens,
                              patch_device,
                              token_dtype,
                              "MTP target-verify patch",
                              "draft_token_sources");
  }
  CHECK_EQ(base_positions.dim(), 1);
  CHECK_EQ(base_kv_seq_lens.dim(), 1);
  const int64_t row_count = base_positions.numel();
  const int64_t verify_width =
      static_cast<int64_t>(draft_token_sources.size()) + 1;
  CHECK_GT(row_count, 0);
  CHECK_EQ(base_kv_seq_lens.numel(), row_count);
  CHECK_EQ(continuation_tokens.numel(), row_count);
  CHECK_EQ(target.token_ids.numel(), row_count * verify_width);
  CHECK_EQ(target.positions.numel(), row_count * verify_width);
  const int64_t expanded_row_count = row_count * verify_width;
  const int64_t expected_kv_rows =
      target.use_chunked_prefill ? row_count : expanded_row_count;
  CHECK_EQ(target.kv_seq_lens.numel(), expected_kv_rows);
  CHECK_EQ(target.new_cache_slots.numel(), row_count * verify_width);
  if (!target.model_managed_multiblock) {
    CHECK_EQ(target.block_tables.size(0), row_count * verify_width);
  }
  CHECK_GE(workspace.position_offsets.numel(), verify_width);
  CHECK_GE(workspace.row_positions.size(0), row_count);
  CHECK_GE(workspace.row_positions.size(1), verify_width);

  torch::Tensor token_rows = target.token_ids.view({row_count, verify_width});
  token_rows.select(/*dim=*/1, /*index=*/0)
      .copy_(continuation_tokens.flatten(), /*non_blocking=*/true);
  for (size_t draft_step = 0; draft_step < draft_token_sources.size();
       ++draft_step) {
    const torch::Tensor& draft_tokens = draft_token_sources[draft_step];
    CHECK(draft_tokens.defined());
    CHECK_EQ(draft_tokens.numel(), row_count);
    token_rows.select(/*dim=*/1, static_cast<int64_t>(draft_step) + 1)
        .copy_(draft_tokens.flatten(), /*non_blocking=*/true);
  }

  torch::Tensor position_offsets =
      workspace.position_offsets.narrow(0, 0, verify_width);
  torch::Tensor row_positions = workspace.row_positions.narrow(0, 0, row_count)
                                    .narrow(1, 0, verify_width);
  row_positions.copy_(base_positions.unsqueeze(/*dim=*/1),
                      /*non_blocking=*/true);
  row_positions.add_(position_offsets.unsqueeze(/*dim=*/0));
  target.positions.flatten().copy_(row_positions.flatten(),
                                   /*non_blocking=*/true);

  torch::Tensor row_kv_seq_lens =
      workspace.row_kv_seq_lens.narrow(0, 0, row_count)
          .narrow(1, 0, verify_width);
  row_kv_seq_lens.copy_(base_kv_seq_lens.unsqueeze(/*dim=*/1),
                        /*non_blocking=*/true);
  row_kv_seq_lens.add_(position_offsets.unsqueeze(/*dim=*/0));
  if (target.use_chunked_prefill) {
    target.kv_seq_lens.flatten().copy_(
        row_kv_seq_lens.select(/*dim=*/1, verify_width - 1),
        /*non_blocking=*/true);
  } else {
    target.kv_seq_lens.flatten().copy_(row_kv_seq_lens.flatten(),
                                       /*non_blocking=*/true);
  }
  if (target.expanded_kv_seq_lens.defined()) {
    CHECK_EQ(target.expanded_kv_seq_lens.numel(), expanded_row_count);
    target.expanded_kv_seq_lens.flatten().copy_(row_kv_seq_lens.flatten(),
                                                /*non_blocking=*/true);
  }

  torch::Tensor flat_positions = target.positions.flatten();
  torch::Tensor flat_cache_slots = target.new_cache_slots.flatten();
  if (target.model_managed_multiblock) {
    flat_cache_slots.zero_();
  } else {
    map_cache_slots_out(flat_positions,
                        target.block_tables,
                        block_size,
                        flat_cache_slots,
                        workspace.position_block_indices,
                        workspace.block_indices,
                        workspace.block_ids,
                        workspace.cache_offsets);
  }

  if (!target.expanded_paged_kv_indptr.defined()) {
    CHECK(!target.expanded_paged_kv_indices.defined());
    CHECK(!target.expanded_paged_kv_last_page_len.defined());
    return;
  }
  CHECK(target.expanded_kv_seq_lens.defined());
  CHECK(target.block_tables.defined());
  CHECK_EQ(target.block_tables.dim(), 2);
  CHECK_EQ(target.block_tables.size(/*dim=*/0), expanded_row_count);
  const int64_t block_table_width = target.block_tables.size(/*dim=*/1);
  CHECK_GT(block_table_width, 0);
  const int64_t page_candidate_count = expanded_row_count * block_table_width;
  CHECK_EQ(target.expanded_paged_kv_indptr.numel(), expanded_row_count + 1);
  CHECK_GE(target.expanded_paged_kv_indices.numel(), page_candidate_count * 2);
  CHECK_EQ(target.expanded_paged_kv_last_page_len.numel(), expanded_row_count);
  CHECK_GE(workspace.page_counts.numel(), expanded_row_count);
  CHECK_GE(workspace.page_column_indices.size(/*dim=*/0), expanded_row_count);
  CHECK_GE(workspace.page_column_indices.size(/*dim=*/1), block_table_width);
  CHECK_GE(workspace.valid_pages.numel(), page_candidate_count);
  CHECK_GE(workspace.valid_page_ranks.numel(), page_candidate_count);
  CHECK_GE(workspace.invalid_page_destinations.numel(), page_candidate_count);
  CHECK_GE(workspace.page_destinations.numel(), page_candidate_count);

  torch::Tensor expanded_kv_seq_lens = target.expanded_kv_seq_lens.flatten();
  torch::Tensor page_counts = workspace.page_counts.narrow(
      /*dim=*/0, /*start=*/0, /*length=*/expanded_row_count);
  page_counts.copy_(expanded_kv_seq_lens, /*non_blocking=*/true);
  page_counts.add_(block_size - 1);
  page_counts.floor_divide_(block_size);
  CHECK_EQ(target.expanded_paged_kv_indptr.scalar_type(),
           page_counts.scalar_type());
  target.expanded_paged_kv_indptr.select(/*dim=*/0, /*index=*/0).zero_();
  torch::Tensor paged_kv_indptr_values = target.expanded_paged_kv_indptr.narrow(
      /*dim=*/0, /*start=*/1, /*length=*/expanded_row_count);
  torch::cumsum_out(paged_kv_indptr_values,
                    page_counts,
                    /*dim=*/0,
                    /*dtype=*/page_counts.scalar_type());

  target.expanded_paged_kv_last_page_len.copy_(expanded_kv_seq_lens,
                                               /*non_blocking=*/true);
  target.expanded_paged_kv_last_page_len.sub_(1);
  target.expanded_paged_kv_last_page_len.remainder_(block_size);
  target.expanded_paged_kv_last_page_len.add_(1);

  torch::Tensor page_column_indices =
      workspace.page_column_indices
          .narrow(/*dim=*/0, /*start=*/0, /*length=*/expanded_row_count)
          .narrow(/*dim=*/1, /*start=*/0, /*length=*/block_table_width);
  torch::Tensor valid_pages = workspace.valid_pages.narrow(
      /*dim=*/0, /*start=*/0, /*length=*/page_candidate_count);
  torch::Tensor valid_page_rows =
      valid_pages.view({expanded_row_count, block_table_width});
  torch::lt_out(
      valid_page_rows, page_column_indices, page_counts.unsqueeze(/*dim=*/1));
  torch::Tensor valid_page_ranks = workspace.valid_page_ranks.narrow(
      /*dim=*/0, /*start=*/0, /*length=*/page_candidate_count);
  torch::cumsum_out(valid_page_ranks,
                    valid_pages,
                    /*dim=*/0,
                    /*dtype=*/torch::kLong);
  valid_page_ranks.sub_(1);
  torch::Tensor invalid_destinations =
      workspace.invalid_page_destinations.narrow(
          /*dim=*/0, /*start=*/0, /*length=*/page_candidate_count);
  torch::arange_out(invalid_destinations, page_candidate_count);
  invalid_destinations.add_(page_candidate_count);
  torch::Tensor page_destinations = workspace.page_destinations.narrow(
      /*dim=*/0, /*start=*/0, /*length=*/page_candidate_count);
  torch::where_out(
      page_destinations, valid_pages, valid_page_ranks, invalid_destinations);
  target.expanded_paged_kv_indices.zero_();
  target.expanded_paged_kv_indices.scatter_(
      /*dim=*/0, page_destinations, target.block_tables.flatten());
}

void publish_mtp_device_step_state(const AcceptedState& source,
                                   const torch::Tensor& accepted_tokens,
                                   MtpDeviceStepState& destination) {
  CHECK(accepted_tokens.defined());
  CHECK_EQ(accepted_tokens.dim(), 2);
  const int64_t row_count = accepted_tokens.size(0);
  const int64_t accepted_width = accepted_tokens.size(1);
  CHECK_LE(row_count, destination.accepted_tokens.size(0));
  CHECK_EQ(accepted_width, destination.accepted_tokens.size(1))
      << "MTP accepted-token width must match the configured Target Verify "
         "capacity";
  CHECK_EQ(source.accepted_lengths.numel(), row_count);
  CHECK_EQ(source.last_tokens.numel(), row_count);
  CHECK_EQ(source.previous_tokens.numel(), row_count);
  CHECK_EQ(source.last_embeddings.size(0), row_count);
  CHECK_EQ(source.previous_embeddings.size(0), row_count);
  CHECK_EQ(source.base_positions.numel(), row_count);
  CHECK_EQ(source.base_kv_seq_lens.numel(), row_count);

  destination.valid_rows.zero_();
  destination.accepted_tokens.narrow(0, 0, row_count)
      .narrow(1, 0, accepted_width)
      .copy_(accepted_tokens, /*non_blocking=*/true);
  destination.accepted_lengths.narrow(0, 0, row_count)
      .copy_(source.accepted_lengths, /*non_blocking=*/true);
  destination.tail_tokens.narrow(0, 0, row_count)
      .copy_(source.last_tokens, /*non_blocking=*/true);
  destination.previous_tokens.narrow(0, 0, row_count)
      .copy_(source.previous_tokens, /*non_blocking=*/true);
  destination.tail_embeddings.narrow(0, 0, row_count)
      .copy_(source.last_embeddings, /*non_blocking=*/true);
  destination.previous_embeddings.narrow(0, 0, row_count)
      .copy_(source.previous_embeddings, /*non_blocking=*/true);
  destination.base_positions.narrow(0, 0, row_count)
      .copy_(source.base_positions, /*non_blocking=*/true);
  destination.base_kv_seq_lens.narrow(0, 0, row_count)
      .copy_(source.base_kv_seq_lens, /*non_blocking=*/true);
  destination.valid_rows.narrow(0, 0, row_count).fill_(true);
}

void publish_mtp_device_step_state_from_outputs(
    const torch::Tensor& accepted_tokens,
    const torch::Tensor& accepted_embeddings,
    const torch::Tensor& embedding_placeholder,
    const torch::Tensor& base_positions,
    const torch::Tensor& base_kv_seq_lens,
    MtpDeviceStepState& destination,
    MtpPublishPatchWorkspace& workspace) {
  CHECK(accepted_tokens.defined());
  CHECK(accepted_embeddings.defined());
  CHECK(embedding_placeholder.defined());
  CHECK(base_positions.defined());
  CHECK(base_kv_seq_lens.defined());
  CHECK_EQ(accepted_tokens.dim(), 2);
  CHECK_EQ(accepted_tokens.scalar_type(), torch::kLong);
  CHECK_EQ(accepted_embeddings.dim(), 3);
  CHECK(accepted_tokens.is_contiguous());
  CHECK(accepted_embeddings.is_contiguous());

  const torch::Device publish_device = accepted_tokens.device();
  const torch::ScalarType embedding_dtype = accepted_embeddings.scalar_type();
  const torch::ScalarType position_dtype = base_positions.scalar_type();
  check_publish_tensor_contract(accepted_embeddings,
                                publish_device,
                                embedding_dtype,
                                "accepted_embeddings");
  check_publish_tensor_contract(embedding_placeholder,
                                publish_device,
                                embedding_dtype,
                                "embedding_placeholder");
  check_publish_tensor_contract(
      base_positions, publish_device, position_dtype, "base_positions");
  check_publish_tensor_contract(
      base_kv_seq_lens, publish_device, position_dtype, "base_kv_seq_lens");
  check_publish_tensor_contract(destination.accepted_tokens,
                                publish_device,
                                torch::kLong,
                                "destination.accepted_tokens");
  check_publish_tensor_contract(destination.accepted_lengths,
                                publish_device,
                                torch::kLong,
                                "destination.accepted_lengths");
  check_publish_tensor_contract(destination.tail_tokens,
                                publish_device,
                                torch::kLong,
                                "destination.tail_tokens");
  check_publish_tensor_contract(destination.previous_tokens,
                                publish_device,
                                torch::kLong,
                                "destination.previous_tokens");
  check_publish_tensor_contract(destination.tail_embeddings,
                                publish_device,
                                embedding_dtype,
                                "destination.tail_embeddings");
  check_publish_tensor_contract(destination.previous_embeddings,
                                publish_device,
                                embedding_dtype,
                                "destination.previous_embeddings");
  check_publish_tensor_contract(destination.base_positions,
                                publish_device,
                                position_dtype,
                                "destination.base_positions");
  check_publish_tensor_contract(destination.base_kv_seq_lens,
                                publish_device,
                                position_dtype,
                                "destination.base_kv_seq_lens");
  check_publish_tensor_contract(destination.valid_rows,
                                publish_device,
                                torch::kBool,
                                "destination.valid_rows");
  check_publish_tensor_contract(workspace.accepted_mask,
                                publish_device,
                                torch::kBool,
                                "workspace.accepted_mask");
  check_publish_tensor_contract(workspace.row_indices,
                                publish_device,
                                torch::kLong,
                                "workspace.row_indices");
  check_publish_tensor_contract(workspace.last_sequence_indices,
                                publish_device,
                                torch::kLong,
                                "workspace.last_sequence_indices");
  check_publish_tensor_contract(workspace.previous_sequence_indices,
                                publish_device,
                                torch::kLong,
                                "workspace.previous_sequence_indices");
  check_publish_tensor_contract(workspace.last_flat_indices,
                                publish_device,
                                torch::kLong,
                                "workspace.last_flat_indices");
  check_publish_tensor_contract(workspace.previous_flat_indices,
                                publish_device,
                                torch::kLong,
                                "workspace.previous_flat_indices");
  check_publish_tensor_contract(workspace.has_previous,
                                publish_device,
                                torch::kBool,
                                "workspace.has_previous");
  check_publish_tensor_contract(workspace.gathered_previous_tokens,
                                publish_device,
                                torch::kLong,
                                "workspace.gathered_previous_tokens");
  check_publish_tensor_contract(workspace.gathered_previous_embeddings,
                                publish_device,
                                embedding_dtype,
                                "workspace.gathered_previous_embeddings");
  const int64_t row_count = accepted_tokens.size(0);
  const int64_t accepted_width = accepted_tokens.size(1);
  const int64_t embedding_size = accepted_embeddings.size(2);
  CHECK_GT(row_count, 0);
  CHECK_GT(accepted_width, 0);
  CHECK_EQ(accepted_embeddings.size(0), row_count);
  CHECK_EQ(accepted_embeddings.size(1), accepted_width);
  CHECK_EQ(embedding_placeholder.numel(), embedding_size);
  CHECK_GE(base_positions.numel(), row_count);
  CHECK_GE(base_kv_seq_lens.numel(), row_count);
  CHECK_LE(row_count, destination.accepted_tokens.size(0));
  CHECK_EQ(accepted_width, destination.accepted_tokens.size(1))
      << "MTP accepted-token width must match the configured Target Verify "
         "capacity";
  CHECK_EQ(destination.tail_embeddings.size(1), embedding_size);
  CHECK_LE(row_count, workspace.accepted_mask.size(0));
  CHECK_EQ(accepted_width, workspace.accepted_mask.size(1));
  CHECK_LE(row_count, workspace.row_indices.numel());
  CHECK_EQ(workspace.gathered_previous_embeddings.size(1), embedding_size);

  torch::Tensor accepted_mask = workspace.accepted_mask.narrow(0, 0, row_count)
                                    .narrow(1, 0, accepted_width);
  torch::ge_out(accepted_mask, accepted_tokens, 0);
  torch::Tensor accepted_lengths =
      destination.accepted_lengths.narrow(0, 0, row_count);
  torch::sum_out(accepted_lengths,
                 accepted_mask,
                 /*dim=*/{1},
                 /*keepdim=*/false,
                 /*dtype=*/torch::kLong);

  torch::Tensor last_sequence_indices =
      workspace.last_sequence_indices.narrow(0, 0, row_count);
  torch::sub_out(last_sequence_indices, accepted_lengths, 1);
  last_sequence_indices.clamp_min_(0);
  torch::Tensor previous_sequence_indices =
      workspace.previous_sequence_indices.narrow(0, 0, row_count);
  torch::sub_out(previous_sequence_indices, accepted_lengths, 2);
  previous_sequence_indices.clamp_min_(0);
  torch::Tensor has_previous = workspace.has_previous.narrow(0, 0, row_count);
  torch::gt_out(has_previous, accepted_lengths, 1);

  torch::Tensor row_indices = workspace.row_indices.narrow(0, 0, row_count);
  torch::Tensor last_flat_indices =
      workspace.last_flat_indices.narrow(0, 0, row_count);
  torch::mul_out(last_flat_indices, row_indices, accepted_width);
  last_flat_indices.add_(last_sequence_indices);
  torch::Tensor previous_flat_indices =
      workspace.previous_flat_indices.narrow(0, 0, row_count);
  torch::mul_out(previous_flat_indices, row_indices, accepted_width);
  previous_flat_indices.add_(previous_sequence_indices);

  destination.valid_rows.zero_();
  destination.accepted_tokens.narrow(0, 0, row_count)
      .narrow(1, 0, accepted_width)
      .copy_(accepted_tokens, /*non_blocking=*/true);

  const torch::Tensor flat_tokens = accepted_tokens.flatten();
  torch::Tensor tail_tokens = destination.tail_tokens.narrow(0, 0, row_count);
  torch::index_select_out(
      tail_tokens, flat_tokens, /*dim=*/0, last_flat_indices);
  torch::Tensor gathered_previous_tokens =
      workspace.gathered_previous_tokens.narrow(0, 0, row_count);
  torch::index_select_out(gathered_previous_tokens,
                          flat_tokens,
                          /*dim=*/0,
                          previous_flat_indices);
  torch::Tensor previous_tokens =
      destination.previous_tokens.narrow(0, 0, row_count);
  torch::where_out(
      previous_tokens, has_previous, gathered_previous_tokens, tail_tokens);

  const torch::Tensor flat_embeddings =
      accepted_embeddings.flatten(/*start_dim=*/0, /*end_dim=*/1);
  torch::Tensor tail_embeddings =
      destination.tail_embeddings.narrow(0, 0, row_count);
  torch::index_select_out(
      tail_embeddings, flat_embeddings, /*dim=*/0, last_flat_indices);
  torch::Tensor gathered_previous_embeddings =
      workspace.gathered_previous_embeddings.narrow(0, 0, row_count);
  torch::index_select_out(gathered_previous_embeddings,
                          flat_embeddings,
                          /*dim=*/0,
                          previous_flat_indices);
  torch::Tensor previous_embeddings =
      destination.previous_embeddings.narrow(0, 0, row_count);
  torch::Tensor embedding_mask = has_previous.unsqueeze(/*dim=*/1);
  torch::Tensor placeholder = embedding_placeholder.flatten().unsqueeze(0);
  torch::where_out(previous_embeddings,
                   embedding_mask,
                   gathered_previous_embeddings,
                   placeholder);

  torch::Tensor published_positions =
      destination.base_positions.narrow(0, 0, row_count);
  published_positions.copy_(base_positions.flatten().narrow(0, 0, row_count),
                            /*non_blocking=*/true);
  published_positions.add_(accepted_lengths);
  torch::Tensor published_kv_seq_lens =
      destination.base_kv_seq_lens.narrow(0, 0, row_count);
  published_kv_seq_lens.copy_(
      base_kv_seq_lens.flatten().narrow(0, 0, row_count),
      /*non_blocking=*/true);
  published_kv_seq_lens.add_(accepted_lengths);
  destination.valid_rows.narrow(0, 0, row_count).fill_(true);
}

void patch_mtp_continuation_rows(const MtpDeviceStepState& predecessor,
                                 const torch::Tensor& predecessor_rows,
                                 MtpContinuationState& continuation,
                                 MtpPredecessorPatchWorkspace& workspace) {
  CHECK(predecessor_rows.defined());
  CHECK_EQ(predecessor_rows.dim(), 1);
  CHECK_EQ(predecessor_rows.scalar_type(), torch::kLong);
  const int64_t row_count = predecessor_rows.numel();
  CHECK_GT(row_count, 0);
  CHECK(continuation.tail_embeddings.defined());
  CHECK(continuation.base_positions.defined());
  const torch::Device continuation_device = predecessor_rows.device();
  const torch::ScalarType embedding_dtype =
      continuation.tail_embeddings.scalar_type();
  const torch::ScalarType position_dtype =
      continuation.base_positions.scalar_type();
  check_mtp_tensor_contract(predecessor.valid_rows,
                            continuation_device,
                            torch::kBool,
                            "MTP continuation",
                            "predecessor.valid_rows");
  check_mtp_tensor_contract(predecessor.tail_tokens,
                            continuation_device,
                            torch::kLong,
                            "MTP continuation",
                            "predecessor.tail_tokens");
  check_mtp_tensor_contract(predecessor.previous_tokens,
                            continuation_device,
                            torch::kLong,
                            "MTP continuation",
                            "predecessor.previous_tokens");
  check_mtp_tensor_contract(predecessor.tail_embeddings,
                            continuation_device,
                            embedding_dtype,
                            "MTP continuation",
                            "predecessor.tail_embeddings");
  check_mtp_tensor_contract(predecessor.previous_embeddings,
                            continuation_device,
                            embedding_dtype,
                            "MTP continuation",
                            "predecessor.previous_embeddings");
  check_mtp_tensor_contract(predecessor.base_positions,
                            continuation_device,
                            position_dtype,
                            "MTP continuation",
                            "predecessor.base_positions");
  check_mtp_tensor_contract(predecessor.base_kv_seq_lens,
                            continuation_device,
                            position_dtype,
                            "MTP continuation",
                            "predecessor.base_kv_seq_lens");
  check_mtp_tensor_contract(continuation.tail_tokens,
                            continuation_device,
                            torch::kLong,
                            "MTP continuation",
                            "continuation.tail_tokens");
  check_mtp_tensor_contract(continuation.previous_tokens,
                            continuation_device,
                            torch::kLong,
                            "MTP continuation",
                            "continuation.previous_tokens");
  check_mtp_tensor_contract(continuation.tail_embeddings,
                            continuation_device,
                            embedding_dtype,
                            "MTP continuation",
                            "continuation.tail_embeddings");
  check_mtp_tensor_contract(continuation.previous_embeddings,
                            continuation_device,
                            embedding_dtype,
                            "MTP continuation",
                            "continuation.previous_embeddings");
  check_mtp_tensor_contract(continuation.base_positions,
                            continuation_device,
                            position_dtype,
                            "MTP continuation",
                            "continuation.base_positions");
  check_mtp_tensor_contract(continuation.base_kv_seq_lens,
                            continuation_device,
                            position_dtype,
                            "MTP continuation",
                            "continuation.base_kv_seq_lens");
  check_mtp_tensor_contract(workspace.gather_rows,
                            continuation_device,
                            torch::kLong,
                            "MTP continuation",
                            "workspace.gather_rows");
  check_mtp_tensor_contract(workspace.gathered_valid_rows,
                            continuation_device,
                            torch::kBool,
                            "MTP continuation",
                            "workspace.gathered_valid_rows");
  check_mtp_tensor_contract(workspace.continuation_mask,
                            continuation_device,
                            torch::kBool,
                            "MTP continuation",
                            "workspace.continuation_mask");
  check_mtp_tensor_contract(workspace.gathered_tokens,
                            continuation_device,
                            torch::kLong,
                            "MTP continuation",
                            "workspace.gathered_tokens");
  check_mtp_tensor_contract(workspace.patched_tokens,
                            continuation_device,
                            torch::kLong,
                            "MTP continuation",
                            "workspace.patched_tokens");
  check_mtp_tensor_contract(workspace.gathered_previous_tokens,
                            continuation_device,
                            torch::kLong,
                            "MTP continuation",
                            "workspace.gathered_previous_tokens");
  check_mtp_tensor_contract(workspace.patched_previous_tokens,
                            continuation_device,
                            torch::kLong,
                            "MTP continuation",
                            "workspace.patched_previous_tokens");
  check_mtp_tensor_contract(workspace.gathered_embeddings,
                            continuation_device,
                            embedding_dtype,
                            "MTP continuation",
                            "workspace.gathered_embeddings");
  check_mtp_tensor_contract(workspace.patched_embeddings,
                            continuation_device,
                            embedding_dtype,
                            "MTP continuation",
                            "workspace.patched_embeddings");
  check_mtp_tensor_contract(workspace.gathered_previous_embeddings,
                            continuation_device,
                            embedding_dtype,
                            "MTP continuation",
                            "workspace.gathered_previous_embeddings");
  check_mtp_tensor_contract(workspace.patched_previous_embeddings,
                            continuation_device,
                            embedding_dtype,
                            "MTP continuation",
                            "workspace.patched_previous_embeddings");
  check_mtp_tensor_contract(workspace.gathered_positions,
                            continuation_device,
                            position_dtype,
                            "MTP continuation",
                            "workspace.gathered_positions");
  check_mtp_tensor_contract(workspace.patched_positions,
                            continuation_device,
                            position_dtype,
                            "MTP continuation",
                            "workspace.patched_positions");
  check_mtp_tensor_contract(workspace.gathered_kv_seq_lens,
                            continuation_device,
                            position_dtype,
                            "MTP continuation",
                            "workspace.gathered_kv_seq_lens");
  check_mtp_tensor_contract(workspace.patched_kv_seq_lens,
                            continuation_device,
                            position_dtype,
                            "MTP continuation",
                            "workspace.patched_kv_seq_lens");
  CHECK_LE(row_count, workspace.gather_rows.numel());
  CHECK_EQ(continuation.tail_tokens.dim(), 1);
  CHECK_EQ(continuation.tail_tokens.numel(), row_count);
  CHECK_EQ(continuation.previous_tokens.dim(), 1);
  CHECK_EQ(continuation.previous_tokens.numel(), row_count);
  CHECK_EQ(continuation.tail_embeddings.dim(), 2);
  CHECK_EQ(continuation.tail_embeddings.size(0), row_count);
  CHECK_EQ(continuation.previous_embeddings.dim(), 2);
  CHECK_EQ(continuation.previous_embeddings.size(0), row_count);
  CHECK_EQ(continuation.base_positions.dim(), 1);
  CHECK_EQ(continuation.base_positions.numel(), row_count);
  CHECK_EQ(continuation.base_kv_seq_lens.dim(), 1);
  CHECK_EQ(continuation.base_kv_seq_lens.numel(), row_count);
  CHECK_EQ(continuation.tail_embeddings.size(1),
           predecessor.tail_embeddings.size(1));
  CHECK_EQ(continuation.previous_embeddings.size(1),
           predecessor.previous_embeddings.size(1));

  torch::Tensor gather_rows = workspace.gather_rows.narrow(0, 0, row_count);
  torch::Tensor gathered_valid_rows =
      workspace.gathered_valid_rows.narrow(0, 0, row_count);
  torch::Tensor continuation_mask =
      workspace.continuation_mask.narrow(0, 0, row_count);
  torch::ge_out(continuation_mask, predecessor_rows, 0);
  torch::clamp_min_out(gather_rows, predecessor_rows, 0);
  torch::index_select_out(gathered_valid_rows,
                          predecessor.valid_rows,
                          /*dim=*/0,
                          gather_rows);
  torch::logical_and_out(
      continuation_mask, continuation_mask, gathered_valid_rows);

  torch::Tensor gathered_tokens =
      workspace.gathered_tokens.narrow(0, 0, row_count);
  torch::Tensor patched_tokens =
      workspace.patched_tokens.narrow(0, 0, row_count);
  torch::index_select_out(gathered_tokens,
                          predecessor.tail_tokens,
                          /*dim=*/0,
                          gather_rows);
  torch::where_out(patched_tokens,
                   continuation_mask,
                   gathered_tokens,
                   continuation.tail_tokens);
  continuation.tail_tokens.copy_(patched_tokens, /*non_blocking=*/true);

  torch::Tensor gathered_previous_tokens =
      workspace.gathered_previous_tokens.narrow(0, 0, row_count);
  torch::Tensor patched_previous_tokens =
      workspace.patched_previous_tokens.narrow(0, 0, row_count);
  torch::index_select_out(gathered_previous_tokens,
                          predecessor.previous_tokens,
                          /*dim=*/0,
                          gather_rows);
  torch::where_out(patched_previous_tokens,
                   continuation_mask,
                   gathered_previous_tokens,
                   continuation.previous_tokens);
  continuation.previous_tokens.copy_(patched_previous_tokens,
                                     /*non_blocking=*/true);

  torch::Tensor gathered_embeddings =
      workspace.gathered_embeddings.narrow(0, 0, row_count);
  torch::Tensor patched_embeddings =
      workspace.patched_embeddings.narrow(0, 0, row_count);
  torch::index_select_out(gathered_embeddings,
                          predecessor.tail_embeddings,
                          /*dim=*/0,
                          gather_rows);
  torch::Tensor embedding_mask = continuation_mask.unsqueeze(-1);
  torch::where_out(patched_embeddings,
                   embedding_mask,
                   gathered_embeddings,
                   continuation.tail_embeddings);
  continuation.tail_embeddings.copy_(patched_embeddings,
                                     /*non_blocking=*/true);

  torch::Tensor gathered_previous_embeddings =
      workspace.gathered_previous_embeddings.narrow(0, 0, row_count);
  torch::Tensor patched_previous_embeddings =
      workspace.patched_previous_embeddings.narrow(0, 0, row_count);
  torch::index_select_out(gathered_previous_embeddings,
                          predecessor.previous_embeddings,
                          /*dim=*/0,
                          gather_rows);
  torch::where_out(patched_previous_embeddings,
                   embedding_mask,
                   gathered_previous_embeddings,
                   continuation.previous_embeddings);
  continuation.previous_embeddings.copy_(patched_previous_embeddings,
                                         /*non_blocking=*/true);

  torch::Tensor gathered_positions =
      workspace.gathered_positions.narrow(0, 0, row_count);
  torch::Tensor patched_positions =
      workspace.patched_positions.narrow(0, 0, row_count);
  torch::index_select_out(gathered_positions,
                          predecessor.base_positions,
                          /*dim=*/0,
                          gather_rows);
  torch::where_out(patched_positions,
                   continuation_mask,
                   gathered_positions,
                   continuation.base_positions);
  continuation.base_positions.copy_(patched_positions,
                                    /*non_blocking=*/true);

  torch::Tensor gathered_kv_seq_lens =
      workspace.gathered_kv_seq_lens.narrow(0, 0, row_count);
  torch::Tensor patched_kv_seq_lens =
      workspace.patched_kv_seq_lens.narrow(0, 0, row_count);
  torch::index_select_out(gathered_kv_seq_lens,
                          predecessor.base_kv_seq_lens,
                          /*dim=*/0,
                          gather_rows);
  torch::where_out(patched_kv_seq_lens,
                   continuation_mask,
                   gathered_kv_seq_lens,
                   continuation.base_kv_seq_lens);
  continuation.base_kv_seq_lens.copy_(patched_kv_seq_lens,
                                      /*non_blocking=*/true);
}

torch::Tensor materialize_speculative_verify_tokens(
    const torch::Tensor& verify_tokens,
    const std::vector<torch::Tensor>& draft_token_sources) {
  if (draft_token_sources.empty()) {
    return verify_tokens;
  }
  CHECK(verify_tokens.defined());
  CHECK_EQ(verify_tokens.dim(), 1);
  const int64_t verify_width =
      static_cast<int64_t>(draft_token_sources.size()) + 1;
  CHECK_EQ(verify_tokens.numel() % verify_width, 0);
  const int64_t batch_size = verify_tokens.numel() / verify_width;
  torch::Tensor verify_rows = verify_tokens.view({batch_size, verify_width});
  for (size_t step = 0; step < draft_token_sources.size(); ++step) {
    const torch::Tensor& source = draft_token_sources[step];
    CHECK(source.defined());
    CHECK_EQ(source.numel(), batch_size);
    verify_rows.select(/*dim=*/1, static_cast<int64_t>(step) + 1)
        .copy_(source.flatten(), /*non_blocking=*/true);
  }
  return verify_tokens;
}

void map_draft_token_ids_to_target_out(
    const torch::Tensor& draft_to_target_token_ids,
    const torch::Tensor& draft_token_ids,
    torch::Tensor target_token_ids) {
  CHECK(draft_to_target_token_ids.defined());
  CHECK(draft_token_ids.defined());
  CHECK(target_token_ids.defined());
  CHECK_EQ(draft_to_target_token_ids.dim(), 1);
  CHECK_EQ(draft_token_ids.dim(), 1)
      << "Prepared Eagle3 draft token ids must be one-dimensional";
  CHECK_EQ(target_token_ids.dim(), 1)
      << "Prepared Eagle3 target token ids must be one-dimensional";
  CHECK_EQ(draft_to_target_token_ids.scalar_type(), torch::kLong);
  CHECK_EQ(draft_token_ids.scalar_type(), torch::kLong);
  CHECK_EQ(target_token_ids.scalar_type(), torch::kLong);
  CHECK_EQ(draft_token_ids.numel(), target_token_ids.numel());
  CHECK_EQ(draft_to_target_token_ids.device(), draft_token_ids.device());
  CHECK_EQ(draft_token_ids.device(), target_token_ids.device());

  const void* target_token_address = target_token_ids.data_ptr();
  torch::index_select_out(target_token_ids,
                          draft_to_target_token_ids,
                          /*dim=*/0,
                          draft_token_ids);
  CHECK_EQ(target_token_ids.data_ptr(), target_token_address)
      << "Prepared Eagle3 token mapping replaced fixed output storage";
}

void copy_prepared_draft_token_ids_out(const torch::Tensor& draft_token_ids,
                                       torch::Tensor target_token_ids) {
  CHECK(draft_token_ids.defined());
  CHECK(target_token_ids.defined());
  CHECK_EQ(draft_token_ids.dim(), 1)
      << "Prepared MTP draft token ids must be one-dimensional";
  CHECK_EQ(target_token_ids.dim(), 1)
      << "Prepared MTP target token ids must be one-dimensional";
  CHECK_EQ(draft_token_ids.scalar_type(), torch::kLong);
  CHECK_EQ(target_token_ids.scalar_type(), torch::kLong);
  CHECK_EQ(draft_token_ids.numel(), target_token_ids.numel());
  CHECK_EQ(draft_token_ids.device(), target_token_ids.device());

  const void* target_token_address = target_token_ids.data_ptr();
  target_token_ids.copy_(draft_token_ids, /*non_blocking=*/true);
  CHECK_EQ(target_token_ids.data_ptr(), target_token_address)
      << "Prepared MTP token copy replaced fixed output storage";
}

torch::Tensor extract_target_base_kv_seq_lens(
    const torch::Tensor& validate_kv_seq_lens,
    int64_t batch_size,
    int64_t num_validate_tokens,
    bool use_chunked_prefill) {
  CHECK(validate_kv_seq_lens.defined());
  CHECK_GT(batch_size, 0);
  CHECK_GT(num_validate_tokens, 0);
  torch::Tensor flattened = validate_kv_seq_lens.flatten();
  if (use_chunked_prefill) {
    CHECK_GE(flattened.numel(), batch_size);
    return flattened.slice(/*dim=*/0, /*start=*/0, /*end=*/batch_size) -
           (num_validate_tokens - 1);
  }

  const int64_t expanded_rows = batch_size * num_validate_tokens;
  CHECK_GE(flattened.numel(), expanded_rows);
  return flattened.slice(/*dim=*/0, /*start=*/0, /*end=*/expanded_rows)
      .view({batch_size, num_validate_tokens})
      .select(/*dim=*/1, /*index=*/0)
      .contiguous();
}

AcceptedState build_accepted_state(const torch::Tensor& accepted_tokens,
                                   const torch::Tensor& accepted_embeddings,
                                   const torch::Tensor& embedding_placeholder,
                                   const torch::Tensor& base_positions,
                                   const torch::Tensor& base_kv_seq_lens) {
  AcceptedTokenMetadata token_metadata = build_accepted_token_metadata(
      accepted_tokens, base_positions, base_kv_seq_lens);
  CHECK_EQ(accepted_embeddings.dim(), 3);
  const int64_t batch_size = accepted_tokens.size(0);
  CHECK_EQ(accepted_embeddings.size(0), batch_size);

  AcceptedState state;
  state.accepted_lengths = token_metadata.accepted_lengths;
  state.all_draft_accepted =
      state.accepted_lengths.eq(accepted_tokens.size(/*dim=*/1));
  state.last_tokens = token_metadata.last_tokens;
  state.base_positions = token_metadata.base_positions;
  state.base_kv_seq_lens = token_metadata.base_kv_seq_lens;

  torch::Tensor previous_indices = (state.accepted_lengths - 2).clamp_min(0);
  torch::Tensor gathered_previous_tokens =
      gather_sequence_rows(accepted_tokens, previous_indices);
  torch::Tensor has_previous = state.accepted_lengths.gt(1);
  state.previous_tokens =
      torch::where(has_previous, gathered_previous_tokens, state.last_tokens);
  torch::Tensor last_indices = (state.accepted_lengths - 1).clamp_min(0);
  state.last_embeddings =
      gather_sequence_rows(accepted_embeddings, last_indices);
  torch::Tensor gathered_previous_embeddings =
      gather_sequence_rows(accepted_embeddings, previous_indices);
  torch::Tensor placeholder = embedding_placeholder;
  if (placeholder.dim() == 1) {
    placeholder = placeholder.unsqueeze(0);
  }
  placeholder = placeholder.expand_as(gathered_previous_embeddings);
  state.previous_embeddings = torch::where(has_previous.view({batch_size, 1}),
                                           gathered_previous_embeddings,
                                           placeholder);

  return state;
}

AcceptedTokenMetadata build_accepted_token_metadata(
    const torch::Tensor& accepted_tokens,
    const torch::Tensor& base_positions,
    const torch::Tensor& base_kv_seq_lens) {
  CHECK_EQ(accepted_tokens.dim(), 2);
  const int64_t batch_size = accepted_tokens.size(0);
  CHECK_GT(accepted_tokens.size(1), 0);
  CHECK_GE(base_positions.numel(), batch_size);
  CHECK_GE(base_kv_seq_lens.numel(), batch_size);

  AcceptedTokenMetadata metadata;
  metadata.accepted_lengths =
      accepted_tokens.ge(0).sum(/*dim=*/1).to(torch::kLong);
  torch::Tensor last_indices = (metadata.accepted_lengths - 1).clamp_min(0);
  metadata.last_tokens = gather_sequence_rows(accepted_tokens, last_indices);
  metadata.base_positions =
      base_positions.flatten().slice(0, 0, batch_size).to(torch::kLong) +
      metadata.accepted_lengths;
  metadata.base_kv_seq_lens =
      base_kv_seq_lens.flatten().slice(0, 0, batch_size).to(torch::kLong) +
      metadata.accepted_lengths;
  return metadata;
}

torch::Tensor make_row_positions(const AcceptedState& state,
                                 const torch::Tensor& offsets) {
  return state.base_positions.unsqueeze(1) +
         offsets.to(state.base_positions.options()).unsqueeze(0);
}

torch::Tensor make_kv_seq_lens(const AcceptedState& state,
                               const torch::Tensor& offsets,
                               bool use_chunked_prefill) {
  if (use_chunked_prefill) {
    return state.base_kv_seq_lens;
  }
  return (state.base_kv_seq_lens.unsqueeze(1) +
          offsets.to(state.base_kv_seq_lens.options()).unsqueeze(0))
      .flatten();
}

torch::Tensor make_repair_cache_positions(const AcceptedState& state) {
  return torch::where(state.all_draft_accepted,
                      state.base_positions - 1,
                      state.base_positions + 1);
}

torch::Tensor map_positions_to_cache_slots(const torch::Tensor& block_tables,
                                           const torch::Tensor& positions,
                                           int32_t block_size) {
  CHECK_EQ(positions.dim(), 2);
  CHECK(block_tables.defined());
  CHECK_GT(block_size, 0);
  const int64_t batch_size = positions.size(0);
  torch::Tensor position_long =
      positions.to(torch::dtype(torch::kLong).device(positions.device()));
  torch::Tensor block_indices =
      torch::floor_divide(position_long, block_size)
          .to(torch::dtype(torch::kLong).device(position_long.device()));
  torch::Tensor block_ids = block_tables.slice(/*dim=*/0, 0, batch_size)
                                .to(torch::kLong)
                                .gather(/*dim=*/1, block_indices);
  return (block_ids * block_size + position_long.remainder(block_size))
      .to(torch::kInt)
      .flatten();
}

}  // namespace xllm::mtp_async
