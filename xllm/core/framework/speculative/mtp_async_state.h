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

#pragma once

#include <torch/torch.h>

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace xllm::mtp_async {

enum class TargetSpecVerifyMode {
  GENERIC,
  CAUSAL_CHUNKED_PREFILL,
  QWEN3_5_EXPANDED_VERIFY,
  DEEPSEEK_V32_EXPANDED_VERIFY,
};

// Keep target verification policy closed over model types with validated
// layouts. Unknown models retain the generic path.
TargetSpecVerifyMode classify_target_spec_verify_mode(
    std::string_view model_type);

// Shared allocation/launch width for target verification block tables. The
// extra entry covers the speculative token that can cross a block boundary.
int64_t speculative_verify_block_table_capacity(int64_t max_position_embeddings,
                                                int64_t block_size);

enum class CombinedDraftExecutionPath {
  UNSUPPORTED,
  QWEN3_5_PAGED_ATTENTION,
  GLM_MOE_DSA_SPARSE_ATTENTION,
};

CombinedDraftExecutionPath classify_combined_draft_execution_path(
    std::string_view model_type);

bool supports_combined_draft_configuration(
    CombinedDraftExecutionPath execution_path,
    std::string_view npu_backend,
    int32_t dp_size);

struct AcceptedState;

// Maps each current request row to its row in the immediately preceding Task.
// A value of -1 marks a new request whose prepared initial state must be kept.
// This is built by the Engine while request ids are already available, so the
// Worker launch path never performs request-id hashing or a Host scan.
std::vector<int64_t> build_predecessor_rows(
    const std::vector<std::string>& previous_request_ids,
    const std::vector<std::string>& current_request_ids);

// Fixed device-resident state published by one MTP Task and consumed by the
// next Task through predecessor_rows. The Pipeline owns one instance per Slot;
// it does not interpret these fields.
struct MtpDeviceStepState {
  torch::Tensor accepted_tokens;
  torch::Tensor accepted_lengths;
  torch::Tensor tail_tokens;
  torch::Tensor previous_tokens;
  torch::Tensor tail_embeddings;
  torch::Tensor previous_embeddings;
  torch::Tensor base_positions;
  torch::Tensor base_kv_seq_lens;
  torch::Tensor valid_rows;
};

// Slot-local fixed scratch used by patch_mtp_continuation_rows(). All tensors
// are allocated before serving; the patch function only submits out/in-place
// Device operations.
struct MtpPredecessorPatchWorkspace {
  torch::Tensor gather_rows;
  torch::Tensor gathered_valid_rows;
  torch::Tensor continuation_mask;
  torch::Tensor gathered_tokens;
  torch::Tensor patched_tokens;
  torch::Tensor gathered_previous_tokens;
  torch::Tensor patched_previous_tokens;
  torch::Tensor gathered_embeddings;
  torch::Tensor patched_embeddings;
  torch::Tensor gathered_previous_embeddings;
  torch::Tensor patched_previous_embeddings;
  torch::Tensor gathered_positions;
  torch::Tensor patched_positions;
  torch::Tensor gathered_kv_seq_lens;
  torch::Tensor patched_kv_seq_lens;
};

// Slot-local fixed scratch used to publish rejection-sampling outputs without
// constructing a transient AcceptedState. row_indices is initialized once and
// all other tensors are reused by out/in-place operations.
struct MtpPublishPatchWorkspace {
  torch::Tensor accepted_mask;
  torch::Tensor row_indices;
  torch::Tensor last_sequence_indices;
  torch::Tensor previous_sequence_indices;
  torch::Tensor last_flat_indices;
  torch::Tensor previous_flat_indices;
  torch::Tensor has_previous;
  torch::Tensor gathered_previous_tokens;
  torch::Tensor gathered_previous_embeddings;
};

// Fixed Draft input views updated after one proposer invocation. The binding
// owns no storage; every Tensor must point into the Slot-local Prepared input
// arena selected during Prepare.
struct MtpNextDraftPatchTarget {
  torch::Tensor token_ids;
  torch::Tensor input_embeddings;
  torch::Tensor positions;
  torch::Tensor kv_seq_lens;
  torch::Tensor new_cache_slots;
  torch::Tensor block_tables;
  bool model_managed_multiblock = false;
};

// Fixed Target verification views. For the standard paged-cache path,
// block_tables contains one row per verify token so cache-slot calculation can
// stay allocation-free during Launch. Model-managed multiblock paths leave it
// undefined and keep new_cache_slots zeroed for their metadata builder.
struct MtpTargetVerifyPatchTarget {
  torch::Tensor token_ids;
  torch::Tensor positions;
  torch::Tensor kv_seq_lens;
  torch::Tensor new_cache_slots;
  torch::Tensor block_tables;
  torch::Tensor expanded_kv_seq_lens;
  torch::Tensor expanded_paged_kv_indptr;
  torch::Tensor expanded_paged_kv_indices;
  torch::Tensor expanded_paged_kv_last_page_len;
  bool use_chunked_prefill = false;
  bool model_managed_multiblock = false;
};

// Slot-local scratch for MTP_NEXT_DRAFT. The destination input views above are
// not part of this workspace and retain their Prepared arena addresses.
struct MtpNextDraftPatchWorkspace {
  torch::Tensor position_block_indices;
  torch::Tensor block_indices;
  torch::Tensor block_ids;
  torch::Tensor cache_offsets;
};

// Slot-local scratch for MTP_TARGET_VERIFY. position_offsets is initialized
// once and all other tensors are reused by out/in-place operations.
struct MtpTargetVerifyPatchWorkspace {
  torch::Tensor position_offsets;
  torch::Tensor row_positions;
  torch::Tensor row_kv_seq_lens;
  torch::Tensor position_block_indices;
  torch::Tensor block_indices;
  torch::Tensor block_ids;
  torch::Tensor cache_offsets;
  torch::Tensor page_counts;
  torch::Tensor page_column_indices;
  torch::Tensor valid_pages;
  torch::Tensor valid_page_ranks;
  torch::Tensor invalid_page_destinations;
  torch::Tensor page_destinations;
};

// Prepared destinations for TASK_CONTINUATION. New rows already contain
// Worker-provided initial values; only rows with a valid predecessor are
// replaced from MtpDeviceStepState.
struct MtpContinuationState {
  torch::Tensor tail_tokens;
  torch::Tensor previous_tokens;
  torch::Tensor tail_embeddings;
  torch::Tensor previous_embeddings;
  torch::Tensor base_positions;
  torch::Tensor base_kv_seq_lens;
};

MtpDeviceStepState allocate_mtp_device_step_state(
    int64_t max_rows,
    int64_t accepted_token_capacity,
    int64_t embedding_size,
    const torch::TensorOptions& token_options,
    const torch::TensorOptions& embedding_options,
    const torch::TensorOptions& position_options);

MtpPredecessorPatchWorkspace allocate_mtp_predecessor_patch_workspace(
    int64_t max_rows,
    int64_t embedding_size,
    const torch::TensorOptions& token_options,
    const torch::TensorOptions& embedding_options,
    const torch::TensorOptions& position_options);

MtpPublishPatchWorkspace allocate_mtp_publish_patch_workspace(
    int64_t max_rows,
    int64_t accepted_token_capacity,
    int64_t embedding_size,
    const torch::TensorOptions& token_options,
    const torch::TensorOptions& embedding_options);

MtpNextDraftPatchWorkspace allocate_mtp_next_draft_patch_workspace(
    int64_t max_rows,
    const torch::TensorOptions& position_options,
    const torch::TensorOptions& cache_slot_options);

MtpTargetVerifyPatchWorkspace allocate_mtp_target_verify_patch_workspace(
    int64_t max_rows,
    int64_t max_verify_width,
    int64_t max_block_table_width,
    const torch::TensorOptions& position_options,
    const torch::TensorOptions& cache_slot_options);

// Publishes already-computed rejection-sampling state into fixed Slot storage.
// accepted_tokens must retain the full Target Verify width. Prepared rejection
// guarantees a valid first token and a contiguous invalid suffix, so its
// accepted length is in [1, verify_width]. The Host template already advances
// one token; therefore predecessor continuation adds at most verify_width - 1,
// matching Graph plan headroom. Tensor addresses remain unchanged.
void publish_mtp_device_step_state(const AcceptedState& source,
                                   const torch::Tensor& accepted_tokens,
                                   MtpDeviceStepState& destination);

// Publishes raw rejection outputs directly into fixed Slot state under the
// same full-width/prefix-mask contract. The base position/KV inputs point at
// the first verification token; accepted lengths advance them to the next
// draft base.
void publish_mtp_device_step_state_from_outputs(
    const torch::Tensor& accepted_tokens,
    const torch::Tensor& accepted_embeddings,
    const torch::Tensor& embedding_placeholder,
    const torch::Tensor& base_positions,
    const torch::Tensor& base_kv_seq_lens,
    MtpDeviceStepState& destination,
    MtpPublishPatchWorkspace& workspace);

void patch_mtp_continuation_rows(const MtpDeviceStepState& predecessor,
                                 const torch::Tensor& predecessor_rows,
                                 MtpContinuationState& continuation,
                                 MtpPredecessorPatchWorkspace& workspace);

// Updates one fixed later-Draft input from the previous Draft output and the
// Task continuation base. No destination Tensor handle is replaced.
void patch_mtp_next_draft_input(const torch::Tensor& draft_tokens,
                                const torch::Tensor& draft_embeddings,
                                const torch::Tensor& base_positions,
                                const torch::Tensor& base_kv_seq_lens,
                                int32_t position_offset,
                                int32_t block_size,
                                MtpNextDraftPatchTarget& target,
                                MtpNextDraftPatchWorkspace& workspace);

// Materializes the fixed row-major Target verification input from the Task
// continuation token and all Draft proposal token columns.
void patch_mtp_target_verify_input(
    const torch::Tensor& continuation_tokens,
    const std::vector<torch::Tensor>& draft_token_sources,
    const torch::Tensor& base_positions,
    const torch::Tensor& base_kv_seq_lens,
    int32_t block_size,
    MtpTargetVerifyPatchTarget& target,
    MtpTargetVerifyPatchWorkspace& workspace);

// Maps algorithm-specific draft vocabulary ids into target vocabulary ids
// without replacing the Slot-local output storage.
void map_draft_token_ids_to_target_out(
    const torch::Tensor& draft_to_target_token_ids,
    const torch::Tensor& draft_token_ids,
    torch::Tensor target_token_ids);

// Copies prepared Draft token ids into a caller-owned, potentially strided
// Slot column without replacing its storage.
void copy_prepared_draft_token_ids_out(const torch::Tensor& draft_token_ids,
                                       torch::Tensor target_token_ids);

// Materialize proposer-owned token columns into the row-major target verify
// input. Graph replay normally performs this copy internally; eager fallback
// must use the same logical tokens before invoking the model.
torch::Tensor materialize_speculative_verify_tokens(
    const torch::Tensor& verify_tokens,
    const std::vector<torch::Tensor>& draft_token_sources);

// Recover the KV length at the first target-verify token. Chunked-prefill
// stores one post-verify length per sequence, while decode stores one length
// per expanded verification row.
torch::Tensor extract_target_base_kv_seq_lens(
    const torch::Tensor& validate_kv_seq_lens,
    int64_t batch_size,
    int64_t num_validate_tokens,
    bool use_chunked_prefill);

// Device-resident state derived from target verification. base_positions and
// base_kv_seq_lens point at the logical position immediately after the accepted
// prefix and are therefore the base of the next draft iteration.
struct AcceptedState {
  torch::Tensor accepted_lengths;
  torch::Tensor all_draft_accepted;
  torch::Tensor last_tokens;
  torch::Tensor previous_tokens;
  torch::Tensor last_embeddings;
  torch::Tensor previous_embeddings;
  torch::Tensor base_positions;
  torch::Tensor base_kv_seq_lens;
};

struct AcceptedTokenMetadata {
  torch::Tensor accepted_lengths;
  torch::Tensor last_tokens;
  torch::Tensor base_positions;
  torch::Tensor base_kv_seq_lens;
};

AcceptedTokenMetadata build_accepted_token_metadata(
    const torch::Tensor& accepted_tokens,
    const torch::Tensor& base_positions,
    const torch::Tensor& base_kv_seq_lens);

AcceptedState build_accepted_state(const torch::Tensor& accepted_tokens,
                                   const torch::Tensor& accepted_embeddings,
                                   const torch::Tensor& embedding_placeholder,
                                   const torch::Tensor& base_positions,
                                   const torch::Tensor& base_kv_seq_lens);

torch::Tensor make_row_positions(const AcceptedState& state,
                                 const torch::Tensor& offsets);

torch::Tensor make_kv_seq_lens(const AcceptedState& state,
                               const torch::Tensor& offsets,
                               bool use_chunked_prefill);

// The repair row is useful only when all draft tokens were accepted. On a
// rejection it is redirected to a future scratch position so it cannot
// overwrite valid draft KV state.
torch::Tensor make_repair_cache_positions(const AcceptedState& state);

torch::Tensor map_positions_to_cache_slots(const torch::Tensor& block_tables,
                                           const torch::Tensor& positions,
                                           int32_t block_size);

}  // namespace xllm::mtp_async
