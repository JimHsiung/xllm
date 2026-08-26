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

#include <gtest/gtest.h>
#include <torch/torch.h>

#include <utility>

namespace xllm::mtp_async {
namespace {

TEST(MtpAsyncStateTest, ClassifiesClosedTargetSpecVerifyPolicy) {
  const std::pair<std::string_view, TargetSpecVerifyMode> test_cases[] = {
      {"qwen3_5", TargetSpecVerifyMode::QWEN3_5_EXPANDED_VERIFY},
      {"qwen3_5_moe", TargetSpecVerifyMode::QWEN3_5_EXPANDED_VERIFY},
      {"qwen3_5_text", TargetSpecVerifyMode::QWEN3_5_EXPANDED_VERIFY},
      {"qwen3_5_moe_text", TargetSpecVerifyMode::QWEN3_5_EXPANDED_VERIFY},
      {"deepseek_v32", TargetSpecVerifyMode::DEEPSEEK_V32_EXPANDED_VERIFY},
      {"mimo", TargetSpecVerifyMode::CAUSAL_CHUNKED_PREFILL},
      {"qwen3_next", TargetSpecVerifyMode::GENERIC},
      {"qwen3_5_mtp", TargetSpecVerifyMode::GENERIC},
      {"qwen3_5_moe_mtp", TargetSpecVerifyMode::GENERIC},
      {"glm_moe_dsa", TargetSpecVerifyMode::GENERIC},
      {"mimo_mtp", TargetSpecVerifyMode::GENERIC},
      {"unknown_model", TargetSpecVerifyMode::GENERIC},
  };

  for (const auto& [model_type, expected] : test_cases) {
    EXPECT_EQ(classify_target_spec_verify_mode(model_type), expected)
        << "model_type=" << model_type;
  }
}

TEST(MtpAsyncStateTest, ClassifiesSupportedCombinedDraftExecutionPaths) {
  EXPECT_EQ(classify_combined_draft_execution_path("qwen3_5_mtp"),
            CombinedDraftExecutionPath::QWEN3_5_PAGED_ATTENTION);
  EXPECT_EQ(classify_combined_draft_execution_path("qwen3_5_moe_mtp"),
            CombinedDraftExecutionPath::QWEN3_5_PAGED_ATTENTION);
  EXPECT_EQ(classify_combined_draft_execution_path("glm_moe_dsa_mtp"),
            CombinedDraftExecutionPath::GLM_MOE_DSA_SPARSE_ATTENTION);
  EXPECT_EQ(classify_combined_draft_execution_path("mimo_mtp"),
            CombinedDraftExecutionPath::UNSUPPORTED);
}

TEST(MtpAsyncStateTest, RestrictsCombinedDraftToValidatedConfigurations) {
  EXPECT_TRUE(supports_combined_draft_configuration(
      CombinedDraftExecutionPath::QWEN3_5_PAGED_ATTENTION,
      "TORCH",
      /*dp_size=*/1));
  EXPECT_FALSE(supports_combined_draft_configuration(
      CombinedDraftExecutionPath::QWEN3_5_PAGED_ATTENTION,
      "ATB",
      /*dp_size=*/1));
  EXPECT_FALSE(supports_combined_draft_configuration(
      CombinedDraftExecutionPath::QWEN3_5_PAGED_ATTENTION,
      "TORCH",
      /*dp_size=*/2));
  EXPECT_TRUE(supports_combined_draft_configuration(
      CombinedDraftExecutionPath::GLM_MOE_DSA_SPARSE_ATTENTION,
      "ATB",
      /*dp_size=*/1));
  EXPECT_TRUE(supports_combined_draft_configuration(
      CombinedDraftExecutionPath::GLM_MOE_DSA_SPARSE_ATTENTION,
      "ATB",
      /*dp_size=*/2));
  EXPECT_FALSE(supports_combined_draft_configuration(
      CombinedDraftExecutionPath::GLM_MOE_DSA_SPARSE_ATTENTION,
      "TORCH",
      /*dp_size=*/1));
  EXPECT_FALSE(supports_combined_draft_configuration(
      CombinedDraftExecutionPath::UNSUPPORTED, "ATB", /*dp_size=*/1));
}

TEST(MtpAsyncStateTest, BuildsPredecessorRowsAcrossBatchReorder) {
  const std::vector<std::string> previous_ids = {
      "request-a", "request-b", "request-c"};
  const std::vector<std::string> current_ids = {
      "request-c", "request-a", "request-new"};

  EXPECT_EQ(build_predecessor_rows(previous_ids, current_ids),
            (std::vector<int64_t>{2, 0, -1}));
  EXPECT_TRUE(build_predecessor_rows(previous_ids, {}).empty());
}

TEST(MtpAsyncStateTest, RejectsDuplicateRequestIdsInPredecessorMapping) {
  EXPECT_DEATH(
      build_predecessor_rows({"request-a", "request-a"}, {"request-a"}),
      "Duplicate request id in predecessor Task");
  EXPECT_DEATH(
      build_predecessor_rows({"request-a"}, {"request-a", "request-a"}),
      "Duplicate request id in current Task");
}

TEST(MtpAsyncStateTest, ExtractsTargetBaseKvLengthsFromVerifyLayouts) {
  const torch::Tensor chunked_kv_seq_lens =
      torch::tensor({104, 204}, torch::kInt);
  const torch::Tensor decode_kv_seq_lens =
      torch::tensor({101, 102, 103, 104, 201, 202, 203, 204}, torch::kInt);

  EXPECT_TRUE(torch::equal(
      extract_target_base_kv_seq_lens(chunked_kv_seq_lens,
                                      /*batch_size=*/2,
                                      /*num_validate_tokens=*/4,
                                      /*use_chunked_prefill=*/true),
      torch::tensor({101, 201}, torch::kInt)));
  EXPECT_TRUE(torch::equal(
      extract_target_base_kv_seq_lens(decode_kv_seq_lens,
                                      /*batch_size=*/2,
                                      /*num_validate_tokens=*/4,
                                      /*use_chunked_prefill=*/false),
      torch::tensor({101, 201}, torch::kInt)));
}

TEST(MtpAsyncStateTest, ComputesSharedSpecVerifyBlockTableCapacity) {
  EXPECT_EQ(speculative_verify_block_table_capacity(262144, 16), 16385);
  EXPECT_EQ(speculative_verify_block_table_capacity(262144, 32), 8193);
  EXPECT_EQ(speculative_verify_block_table_capacity(262144, 64), 4097);
  EXPECT_EQ(speculative_verify_block_table_capacity(262144, 128), 2049);
  EXPECT_EQ(speculative_verify_block_table_capacity(300000, 128), 2345);
}

TEST(MtpAsyncStateTest, MaterializesDraftColumnsForEagerFallback) {
  torch::Tensor verify_tokens =
      torch::tensor({10, -1, -1, 20, -1, -1}, torch::kInt);
  const std::vector<torch::Tensor> draft_sources = {
      torch::tensor({11, 21}, torch::kLong),
      torch::tensor({12, 22}, torch::kLong)};

  torch::Tensor materialized =
      materialize_speculative_verify_tokens(verify_tokens, draft_sources);

  EXPECT_EQ(materialized.data_ptr(), verify_tokens.data_ptr());
  EXPECT_TRUE(torch::equal(
      materialized, torch::tensor({10, 11, 12, 20, 21, 22}, torch::kInt)));
}

TEST(MtpAsyncStateTest, LeavesOrdinaryEagerTokensUnchanged) {
  const torch::Tensor verify_tokens = torch::tensor({10, 11}, torch::kInt);
  torch::Tensor materialized =
      materialize_speculative_verify_tokens(verify_tokens, {});

  EXPECT_EQ(materialized.data_ptr(), verify_tokens.data_ptr());
  EXPECT_TRUE(torch::equal(materialized, verify_tokens));
}

TEST(MtpAsyncStateTest, BuildsMixedAcceptanceStateWithoutHostRoundTrip) {
  const torch::Tensor accepted_tokens = torch::tensor(
      {{10, 11, 12, 13}, {20, 21, -1, -1}, {30, -1, -1, -1}}, torch::kLong);
  const torch::Tensor accepted_embeddings =
      torch::arange(24, torch::kFloat).reshape({3, 4, 2});
  const torch::Tensor placeholder = torch::tensor({-100.0, -101.0});
  const torch::Tensor base_positions = torch::tensor({100, 200, 300});
  const torch::Tensor base_kv_seq_lens = torch::tensor({101, 201, 301});

  const AcceptedState state = build_accepted_state(accepted_tokens,
                                                   accepted_embeddings,
                                                   placeholder,
                                                   base_positions,
                                                   base_kv_seq_lens);

  EXPECT_TRUE(torch::equal(state.accepted_lengths,
                           torch::tensor({4, 2, 1}, torch::kLong)));
  EXPECT_TRUE(torch::equal(state.all_draft_accepted,
                           torch::tensor({true, false, false})));
  EXPECT_TRUE(torch::equal(state.last_tokens, torch::tensor({13, 21, 30})));
  EXPECT_TRUE(torch::equal(state.previous_tokens, torch::tensor({12, 20, 30})));
  EXPECT_TRUE(torch::equal(state.last_embeddings,
                           torch::stack({accepted_embeddings[0][3],
                                         accepted_embeddings[1][1],
                                         accepted_embeddings[2][0]})));
  EXPECT_TRUE(torch::equal(state.previous_embeddings,
                           torch::stack({accepted_embeddings[0][2],
                                         accepted_embeddings[1][0],
                                         placeholder})));
  EXPECT_TRUE(torch::equal(state.base_positions,
                           torch::tensor({104, 202, 301}, torch::kLong)));
  EXPECT_TRUE(torch::equal(state.base_kv_seq_lens,
                           torch::tensor({105, 203, 302}, torch::kLong)));
}

TEST(MtpAsyncStateTest, PublishesAndPatchesFixedMtpDeviceState) {
  const torch::Tensor accepted_tokens = torch::tensor(
      {{10, 11, 12, 13}, {20, 21, -1, -1}, {30, -1, -1, -1}}, torch::kLong);
  const torch::Tensor accepted_embeddings =
      torch::arange(24, torch::kFloat).reshape({3, 4, 2});
  const torch::Tensor placeholder = torch::tensor({-100.0, -101.0});
  const torch::Tensor base_positions = torch::tensor({100, 200, 300});
  const torch::Tensor base_kv_seq_lens = torch::tensor({101, 201, 301});
  const AcceptedState accepted_state = build_accepted_state(accepted_tokens,
                                                            accepted_embeddings,
                                                            placeholder,
                                                            base_positions,
                                                            base_kv_seq_lens);

  const torch::TensorOptions token_options =
      torch::TensorOptions().dtype(torch::kLong).device(torch::kCPU);
  const torch::TensorOptions embedding_options =
      torch::TensorOptions().dtype(torch::kFloat).device(torch::kCPU);
  MtpDeviceStepState device_state = allocate_mtp_device_step_state(
      /*max_rows=*/4,
      /*accepted_token_capacity=*/4,
      /*embedding_size=*/2,
      token_options,
      embedding_options,
      token_options);
  const void* accepted_tokens_address = device_state.accepted_tokens.data_ptr();
  const void* tail_embeddings_address = device_state.tail_embeddings.data_ptr();
  publish_mtp_device_step_state(accepted_state, accepted_tokens, device_state);

  EXPECT_EQ(device_state.accepted_tokens.data_ptr(), accepted_tokens_address);
  EXPECT_EQ(device_state.tail_embeddings.data_ptr(), tail_embeddings_address);
  EXPECT_TRUE(torch::equal(
      device_state.accepted_tokens.narrow(/*dim=*/0, /*start=*/0, /*length=*/3),
      accepted_tokens));
  EXPECT_TRUE(torch::equal(device_state.valid_rows,
                           torch::tensor({true, true, true, false})));

  MtpPredecessorPatchWorkspace workspace =
      allocate_mtp_predecessor_patch_workspace(
          /*max_rows=*/5,
          /*embedding_size=*/2,
          token_options,
          embedding_options,
          token_options);
  MtpContinuationState continuation;
  continuation.tail_tokens =
      torch::tensor({900, 901, 902, 903, 904}, torch::kLong);
  continuation.previous_tokens =
      torch::tensor({800, 801, 802, 803, 804}, torch::kLong);
  continuation.tail_embeddings = torch::tensor({{-1.0F, -2.0F},
                                                {-3.0F, -4.0F},
                                                {-5.0F, -6.0F},
                                                {-7.0F, -8.0F},
                                                {-9.0F, -10.0F}});
  continuation.previous_embeddings = torch::tensor({{-11.0F, -12.0F},
                                                    {-13.0F, -14.0F},
                                                    {-15.0F, -16.0F},
                                                    {-17.0F, -18.0F},
                                                    {-19.0F, -20.0F}});
  continuation.base_positions =
      torch::tensor({900, 901, 902, 903, 904}, torch::kLong);
  continuation.base_kv_seq_lens =
      torch::tensor({910, 911, 912, 913, 914}, torch::kLong);
  const void* continuation_token_address = continuation.tail_tokens.data_ptr();
  const void* continuation_embedding_address =
      continuation.tail_embeddings.data_ptr();

  // Rows 0/1/3 continue prior rows 2/0/1. Row 2 is new (-1), and row 4
  // points at allocated-but-unpublished state row 3; both keep initial data.
  const torch::Tensor predecessor_rows =
      torch::tensor({2, 0, -1, 1, 3}, torch::kLong);
  patch_mtp_continuation_rows(
      device_state, predecessor_rows, continuation, workspace);

  EXPECT_EQ(continuation.tail_tokens.data_ptr(), continuation_token_address);
  EXPECT_EQ(continuation.tail_embeddings.data_ptr(),
            continuation_embedding_address);
  EXPECT_TRUE(
      torch::equal(continuation.tail_tokens,
                   torch::tensor({30, 13, 902, 21, 904}, torch::kLong)));
  EXPECT_TRUE(
      torch::equal(continuation.previous_tokens,
                   torch::tensor({30, 12, 802, 20, 804}, torch::kLong)));
  EXPECT_TRUE(torch::equal(continuation.tail_embeddings,
                           torch::stack({accepted_embeddings[2][0],
                                         accepted_embeddings[0][3],
                                         torch::tensor({-5.0F, -6.0F}),
                                         accepted_embeddings[1][1],
                                         torch::tensor({-9.0F, -10.0F})})));
  EXPECT_TRUE(torch::equal(continuation.previous_embeddings,
                           torch::stack({placeholder,
                                         accepted_embeddings[0][2],
                                         torch::tensor({-15.0F, -16.0F}),
                                         accepted_embeddings[1][0],
                                         torch::tensor({-19.0F, -20.0F})})));
  EXPECT_TRUE(
      torch::equal(continuation.base_positions,
                   torch::tensor({301, 104, 902, 202, 904}, torch::kLong)));
  EXPECT_TRUE(
      torch::equal(continuation.base_kv_seq_lens,
                   torch::tensor({302, 105, 912, 203, 914}, torch::kLong)));
}

TEST(MtpAsyncStateTest, PublishesRawOutputsWithoutReplacingStateStorage) {
  const torch::Tensor accepted_tokens = torch::tensor(
      {{10, 11, 12, 13}, {20, 21, -1, -1}, {30, -1, -1, -1}}, torch::kLong);
  const torch::Tensor accepted_embeddings =
      torch::arange(24, torch::kFloat).reshape({3, 4, 2});
  const torch::Tensor placeholder = torch::tensor({-100.0F, -101.0F});
  const torch::Tensor base_positions =
      torch::tensor({100, 200, 300}, torch::kInt);
  const torch::Tensor base_kv_seq_lens =
      torch::tensor({101, 201, 301}, torch::kInt);
  const torch::TensorOptions token_options =
      torch::TensorOptions().dtype(torch::kLong).device(torch::kCPU);
  const torch::TensorOptions embedding_options =
      torch::TensorOptions().dtype(torch::kFloat).device(torch::kCPU);
  const torch::TensorOptions position_options =
      torch::TensorOptions().dtype(torch::kInt).device(torch::kCPU);
  MtpDeviceStepState state = allocate_mtp_device_step_state(
      /*max_rows=*/4,
      /*accepted_token_capacity=*/4,
      /*embedding_size=*/2,
      token_options,
      embedding_options,
      position_options);
  MtpPublishPatchWorkspace workspace = allocate_mtp_publish_patch_workspace(
      /*max_rows=*/4,
      /*accepted_token_capacity=*/4,
      /*embedding_size=*/2,
      token_options,
      embedding_options);
  const void* accepted_tokens_address = state.accepted_tokens.data_ptr();
  const void* accepted_lengths_address = state.accepted_lengths.data_ptr();
  const void* tail_embeddings_address = state.tail_embeddings.data_ptr();
  const void* base_positions_address = state.base_positions.data_ptr();

  publish_mtp_device_step_state_from_outputs(accepted_tokens,
                                             accepted_embeddings,
                                             placeholder,
                                             base_positions,
                                             base_kv_seq_lens,
                                             state,
                                             workspace);

  EXPECT_EQ(state.accepted_tokens.data_ptr(), accepted_tokens_address);
  EXPECT_EQ(state.accepted_lengths.data_ptr(), accepted_lengths_address);
  EXPECT_EQ(state.tail_embeddings.data_ptr(), tail_embeddings_address);
  EXPECT_EQ(state.base_positions.data_ptr(), base_positions_address);
  EXPECT_TRUE(torch::equal(
      state.accepted_tokens.narrow(/*dim=*/0, /*start=*/0, /*length=*/3),
      accepted_tokens));
  EXPECT_TRUE(torch::equal(state.accepted_lengths.narrow(0, 0, 3),
                           torch::tensor({4, 2, 1}, torch::kLong)));
  EXPECT_TRUE(torch::equal(state.tail_tokens.narrow(0, 0, 3),
                           torch::tensor({13, 21, 30}, torch::kLong)));
  EXPECT_TRUE(torch::equal(state.previous_tokens.narrow(0, 0, 3),
                           torch::tensor({12, 20, 30}, torch::kLong)));
  EXPECT_TRUE(torch::equal(state.tail_embeddings.narrow(0, 0, 3),
                           torch::stack({accepted_embeddings[0][3],
                                         accepted_embeddings[1][1],
                                         accepted_embeddings[2][0]})));
  EXPECT_TRUE(torch::equal(state.previous_embeddings.narrow(0, 0, 3),
                           torch::stack({accepted_embeddings[0][2],
                                         accepted_embeddings[1][0],
                                         placeholder})));
  EXPECT_TRUE(torch::equal(state.base_positions.narrow(0, 0, 3),
                           torch::tensor({104, 202, 301}, torch::kInt)));
  EXPECT_TRUE(torch::equal(state.base_kv_seq_lens.narrow(0, 0, 3),
                           torch::tensor({105, 203, 302}, torch::kInt)));
  EXPECT_TRUE(
      torch::equal(state.valid_rows, torch::tensor({true, true, true, false})));
}

TEST(MtpAsyncStateTest, RejectsPublishWidthBelowVerifyCapacity) {
  const torch::TensorOptions token_options =
      torch::TensorOptions().dtype(torch::kLong).device(torch::kCPU);
  const torch::TensorOptions embedding_options =
      torch::TensorOptions().dtype(torch::kFloat).device(torch::kCPU);
  const torch::TensorOptions position_options =
      torch::TensorOptions().dtype(torch::kInt).device(torch::kCPU);
  MtpDeviceStepState state = allocate_mtp_device_step_state(
      /*max_rows=*/2,
      /*accepted_token_capacity=*/4,
      /*embedding_size=*/2,
      token_options,
      embedding_options,
      position_options);
  MtpPublishPatchWorkspace workspace = allocate_mtp_publish_patch_workspace(
      /*max_rows=*/2,
      /*accepted_token_capacity=*/4,
      /*embedding_size=*/2,
      token_options,
      embedding_options);

  EXPECT_DEATH(publish_mtp_device_step_state_from_outputs(
                   torch::tensor({{10, 11, -1}}, token_options),
                   torch::zeros({1, 3, 2}, embedding_options),
                   torch::zeros({2}, embedding_options),
                   torch::tensor({100}, position_options),
                   torch::tensor({101}, position_options),
                   state,
                   workspace),
               "must match the configured Target Verify capacity");
}

TEST(MtpAsyncStateTest, RejectsRawPublishSourcesOnDifferentDevices) {
  const torch::TensorOptions token_options =
      torch::TensorOptions().dtype(torch::kLong).device(torch::kCPU);
  const torch::TensorOptions embedding_options =
      torch::TensorOptions().dtype(torch::kFloat).device(torch::kCPU);
  const torch::TensorOptions position_options =
      torch::TensorOptions().dtype(torch::kInt).device(torch::kCPU);
  const torch::TensorOptions meta_embedding_options =
      embedding_options.device(torch::Device("meta"));
  const torch::TensorOptions meta_position_options =
      position_options.device(torch::Device("meta"));
  MtpDeviceStepState state = allocate_mtp_device_step_state(
      /*max_rows=*/1,
      /*accepted_token_capacity=*/4,
      /*embedding_size=*/2,
      token_options,
      embedding_options,
      position_options);
  MtpPublishPatchWorkspace workspace = allocate_mtp_publish_patch_workspace(
      /*max_rows=*/1,
      /*accepted_token_capacity=*/4,
      /*embedding_size=*/2,
      token_options,
      embedding_options);
  const torch::Tensor accepted_tokens =
      torch::tensor({{10, -1, -1, -1}}, token_options);
  const torch::Tensor accepted_embeddings =
      torch::zeros({1, 4, 2}, embedding_options);
  const torch::Tensor placeholder = torch::zeros({2}, embedding_options);
  const torch::Tensor base_positions = torch::tensor({100}, position_options);
  const torch::Tensor base_kv_seq_lens = torch::tensor({101}, position_options);

  EXPECT_DEATH(publish_mtp_device_step_state_from_outputs(
                   accepted_tokens,
                   accepted_embeddings,
                   torch::zeros({2}, meta_embedding_options),
                   base_positions,
                   base_kv_seq_lens,
                   state,
                   workspace),
               "embedding_placeholder");
  EXPECT_DEATH(publish_mtp_device_step_state_from_outputs(
                   accepted_tokens,
                   accepted_embeddings,
                   placeholder,
                   torch::zeros({1}, meta_position_options),
                   base_kv_seq_lens,
                   state,
                   workspace),
               "base_positions");
  EXPECT_DEATH(publish_mtp_device_step_state_from_outputs(
                   accepted_tokens,
                   accepted_embeddings,
                   placeholder,
                   base_positions,
                   torch::zeros({1}, meta_position_options),
                   state,
                   workspace),
               "base_kv_seq_lens");
}

TEST(MtpAsyncStateTest, RejectsRawPublishSourcesWithIncompatibleDtypes) {
  const torch::TensorOptions token_options =
      torch::TensorOptions().dtype(torch::kLong).device(torch::kCPU);
  const torch::TensorOptions embedding_options =
      torch::TensorOptions().dtype(torch::kFloat).device(torch::kCPU);
  const torch::TensorOptions position_options =
      torch::TensorOptions().dtype(torch::kInt).device(torch::kCPU);
  MtpDeviceStepState state = allocate_mtp_device_step_state(
      /*max_rows=*/1,
      /*accepted_token_capacity=*/4,
      /*embedding_size=*/2,
      token_options,
      embedding_options,
      position_options);
  MtpPublishPatchWorkspace workspace = allocate_mtp_publish_patch_workspace(
      /*max_rows=*/1,
      /*accepted_token_capacity=*/4,
      /*embedding_size=*/2,
      token_options,
      embedding_options);
  const torch::Tensor accepted_tokens =
      torch::tensor({{10, -1, -1, -1}}, token_options);
  const torch::Tensor accepted_embeddings =
      torch::zeros({1, 4, 2}, embedding_options);
  const torch::Tensor placeholder = torch::zeros({2}, embedding_options);
  const torch::Tensor base_positions = torch::tensor({100}, position_options);
  const torch::Tensor base_kv_seq_lens = torch::tensor({101}, position_options);

  EXPECT_DEATH(publish_mtp_device_step_state_from_outputs(
                   accepted_tokens,
                   accepted_embeddings,
                   torch::zeros({2}, embedding_options.dtype(torch::kDouble)),
                   base_positions,
                   base_kv_seq_lens,
                   state,
                   workspace),
               "embedding_placeholder");
  EXPECT_DEATH(publish_mtp_device_step_state_from_outputs(
                   accepted_tokens,
                   accepted_embeddings,
                   placeholder,
                   base_positions,
                   torch::tensor({101}, token_options),
                   state,
                   workspace),
               "base_kv_seq_lens");
}

TEST(MtpAsyncStateTest, RejectsContinuationStateWithIncompatibleDeviceOrDtype) {
  const torch::TensorOptions token_options =
      torch::TensorOptions().dtype(torch::kLong).device(torch::kCPU);
  const torch::TensorOptions embedding_options =
      torch::TensorOptions().dtype(torch::kFloat).device(torch::kCPU);
  const torch::TensorOptions position_options =
      torch::TensorOptions().dtype(torch::kInt).device(torch::kCPU);
  MtpDeviceStepState predecessor = allocate_mtp_device_step_state(
      /*max_rows=*/1,
      /*accepted_token_capacity=*/4,
      /*embedding_size=*/2,
      token_options,
      embedding_options,
      position_options);
  MtpPredecessorPatchWorkspace workspace =
      allocate_mtp_predecessor_patch_workspace(
          /*max_rows=*/1,
          /*embedding_size=*/2,
          token_options,
          embedding_options,
          position_options);
  MtpContinuationState continuation;
  continuation.tail_tokens = torch::zeros({1}, token_options);
  continuation.previous_tokens = torch::zeros({1}, token_options);
  continuation.tail_embeddings = torch::zeros({1, 2}, embedding_options);
  continuation.previous_embeddings = torch::zeros({1, 2}, embedding_options);
  continuation.base_positions = torch::zeros({1}, position_options);
  continuation.base_kv_seq_lens = torch::zeros({1}, position_options);
  const torch::Tensor predecessor_rows = torch::zeros({1}, token_options);

  MtpContinuationState wrong_device = continuation;
  wrong_device.tail_embeddings =
      torch::zeros({1, 2}, embedding_options.device(torch::Device("meta")));
  EXPECT_DEATH(patch_mtp_continuation_rows(
                   predecessor, predecessor_rows, wrong_device, workspace),
               "continuation.tail_embeddings");

  MtpContinuationState wrong_dtype = continuation;
  wrong_dtype.base_kv_seq_lens =
      torch::zeros({1}, position_options.dtype(torch::kLong));
  EXPECT_DEATH(patch_mtp_continuation_rows(
                   predecessor, predecessor_rows, wrong_dtype, workspace),
               "continuation.base_kv_seq_lens");
}

TEST(MtpAsyncStateTest, PatchesFixedNextDraftInputWithoutReplacingStorage) {
  const torch::TensorOptions position_options =
      torch::TensorOptions().dtype(torch::kInt).device(torch::kCPU);
  const torch::TensorOptions cache_slot_options = position_options;
  MtpNextDraftPatchWorkspace workspace =
      allocate_mtp_next_draft_patch_workspace(
          /*max_rows=*/4, position_options, cache_slot_options);

  MtpNextDraftPatchTarget target;
  target.token_ids = torch::empty({2}, torch::kLong);
  target.input_embeddings = torch::empty({2, 2}, torch::kFloat);
  target.positions = torch::empty({2}, torch::kInt);
  target.kv_seq_lens = torch::empty({2}, torch::kInt);
  target.new_cache_slots = torch::empty({2}, torch::kInt);
  target.block_tables =
      torch::tensor({{10, 11, 12}, {20, 21, 22}}, torch::kInt);
  const void* token_address = target.token_ids.data_ptr();
  const void* embedding_address = target.input_embeddings.data_ptr();
  const void* position_address = target.positions.data_ptr();
  const void* cache_slot_address = target.new_cache_slots.data_ptr();

  patch_mtp_next_draft_input(
      torch::tensor({101, 201}, torch::kLong),
      torch::tensor({{1.0F, 2.0F}, {3.0F, 4.0F}}, torch::kFloat),
      torch::tensor({3, 5}, torch::kInt),
      torch::tensor({4, 6}, torch::kInt),
      /*position_offset=*/1,
      /*block_size=*/4,
      target,
      workspace);

  EXPECT_EQ(target.token_ids.data_ptr(), token_address);
  EXPECT_EQ(target.input_embeddings.data_ptr(), embedding_address);
  EXPECT_EQ(target.positions.data_ptr(), position_address);
  EXPECT_EQ(target.new_cache_slots.data_ptr(), cache_slot_address);
  EXPECT_TRUE(
      torch::equal(target.token_ids, torch::tensor({101, 201}, torch::kLong)));
  EXPECT_TRUE(
      torch::equal(target.input_embeddings,
                   torch::tensor({{1.0F, 2.0F}, {3.0F, 4.0F}}, torch::kFloat)));
  EXPECT_TRUE(
      torch::equal(target.positions, torch::tensor({4, 6}, torch::kInt)));
  EXPECT_TRUE(
      torch::equal(target.kv_seq_lens, torch::tensor({5, 7}, torch::kInt)));
  EXPECT_TRUE(torch::equal(target.new_cache_slots,
                           torch::tensor({44, 86}, torch::kInt)));
}

TEST(MtpAsyncStateTest, PatchesModelManagedNextDraftInputWithZeroCacheSlots) {
  const torch::TensorOptions position_options =
      torch::TensorOptions().dtype(torch::kInt).device(torch::kCPU);
  MtpNextDraftPatchWorkspace workspace =
      allocate_mtp_next_draft_patch_workspace(
          /*max_rows=*/4, position_options, position_options);

  MtpNextDraftPatchTarget target;
  target.token_ids = torch::empty({2}, torch::kLong);
  target.input_embeddings = torch::empty({2, 2}, torch::kFloat);
  target.positions = torch::empty({2}, torch::kInt);
  target.kv_seq_lens = torch::empty({2}, torch::kInt);
  target.new_cache_slots = torch::full({2}, -1, torch::kInt);
  target.model_managed_multiblock = true;

  patch_mtp_next_draft_input(
      torch::tensor({101, 201}, torch::kLong),
      torch::tensor({{1.0F, 2.0F}, {3.0F, 4.0F}}, torch::kFloat),
      torch::tensor({3, 5}, torch::kInt),
      torch::tensor({4, 6}, torch::kInt),
      /*position_offset=*/1,
      /*block_size=*/4,
      target,
      workspace);

  EXPECT_TRUE(
      torch::equal(target.positions, torch::tensor({4, 6}, torch::kInt)));
  EXPECT_TRUE(
      torch::equal(target.kv_seq_lens, torch::tensor({5, 7}, torch::kInt)));
  EXPECT_TRUE(
      torch::equal(target.new_cache_slots, torch::zeros({2}, torch::kInt)));
}

TEST(MtpAsyncStateTest, RejectsNextDraftDeviceAndDtypeMismatchesBeforeWrites) {
  const torch::TensorOptions token_options =
      torch::TensorOptions().dtype(torch::kLong).device(torch::kCPU);
  const torch::TensorOptions embedding_options =
      torch::TensorOptions().dtype(torch::kFloat).device(torch::kCPU);
  const torch::TensorOptions position_options =
      torch::TensorOptions().dtype(torch::kInt).device(torch::kCPU);
  MtpNextDraftPatchWorkspace workspace =
      allocate_mtp_next_draft_patch_workspace(
          /*max_rows=*/2, position_options, position_options);

  MtpNextDraftPatchTarget target;
  target.token_ids = torch::empty({2}, token_options);
  target.input_embeddings = torch::empty({2, 2}, embedding_options);
  target.positions = torch::empty({2}, position_options);
  target.kv_seq_lens = torch::empty({2}, position_options);
  target.new_cache_slots = torch::empty({2}, position_options);
  target.block_tables = torch::tensor({{10, 11}, {20, 21}}, position_options);
  const torch::Tensor draft_tokens = torch::tensor({101, 201}, token_options);
  const torch::Tensor draft_embeddings =
      torch::tensor({{1.0F, 2.0F}, {3.0F, 4.0F}}, embedding_options);
  const torch::Tensor base_positions = torch::tensor({3, 5}, position_options);
  const torch::Tensor base_kv_seq_lens =
      torch::tensor({4, 6}, position_options);

  target.input_embeddings =
      torch::empty({2, 2}, embedding_options.device(torch::Device("meta")));
  EXPECT_DEATH(patch_mtp_next_draft_input(draft_tokens,
                                          draft_embeddings,
                                          base_positions,
                                          base_kv_seq_lens,
                                          /*position_offset=*/1,
                                          /*block_size=*/4,
                                          target,
                                          workspace),
               "target.input_embeddings.*contract device");

  target.input_embeddings = torch::empty({2, 2}, embedding_options);
  workspace.cache_offsets =
      torch::empty({2}, position_options.dtype(torch::kLong));
  EXPECT_DEATH(patch_mtp_next_draft_input(draft_tokens,
                                          draft_embeddings,
                                          base_positions,
                                          base_kv_seq_lens,
                                          /*position_offset=*/1,
                                          /*block_size=*/4,
                                          target,
                                          workspace),
               "workspace.cache_offsets.*incompatible dtype");
}

TEST(MtpAsyncStateTest, PatchesFixedTargetVerifyInputWithoutReplacingStorage) {
  const torch::TensorOptions position_options =
      torch::TensorOptions().dtype(torch::kInt).device(torch::kCPU);
  const torch::TensorOptions cache_slot_options = position_options;
  MtpTargetVerifyPatchWorkspace workspace =
      allocate_mtp_target_verify_patch_workspace(
          /*max_rows=*/4,
          /*max_verify_width=*/4,
          /*max_block_table_width=*/3,
          position_options,
          cache_slot_options);

  MtpTargetVerifyPatchTarget target;
  target.token_ids = torch::empty({6}, torch::kLong);
  target.positions = torch::empty({6}, torch::kInt);
  target.kv_seq_lens = torch::empty({6}, torch::kInt);
  target.new_cache_slots = torch::empty({6}, torch::kInt);
  target.block_tables = torch::tensor({{10, 11, 12},
                                       {10, 11, 12},
                                       {10, 11, 12},
                                       {20, 21, 22},
                                       {20, 21, 22},
                                       {20, 21, 22}},
                                      torch::kInt);
  const void* token_address = target.token_ids.data_ptr();
  const void* position_address = target.positions.data_ptr();
  const void* kv_seq_lens_address = target.kv_seq_lens.data_ptr();
  const void* cache_slot_address = target.new_cache_slots.data_ptr();
  const std::vector<torch::Tensor> draft_sources = {
      torch::tensor({101, 201}, torch::kLong),
      torch::tensor({102, 202}, torch::kLong)};

  patch_mtp_target_verify_input(torch::tensor({100, 200}, torch::kLong),
                                draft_sources,
                                torch::tensor({3, 5}, torch::kInt),
                                torch::tensor({4, 6}, torch::kInt),
                                /*block_size=*/4,
                                target,
                                workspace);

  EXPECT_EQ(target.token_ids.data_ptr(), token_address);
  EXPECT_EQ(target.positions.data_ptr(), position_address);
  EXPECT_EQ(target.kv_seq_lens.data_ptr(), kv_seq_lens_address);
  EXPECT_EQ(target.new_cache_slots.data_ptr(), cache_slot_address);
  EXPECT_TRUE(torch::equal(
      target.token_ids,
      torch::tensor({100, 101, 102, 200, 201, 202}, torch::kLong)));
  EXPECT_TRUE(torch::equal(target.positions,
                           torch::tensor({3, 4, 5, 5, 6, 7}, torch::kInt)));
  EXPECT_TRUE(torch::equal(target.kv_seq_lens,
                           torch::tensor({4, 5, 6, 6, 7, 8}, torch::kInt)));
  EXPECT_TRUE(
      torch::equal(target.new_cache_slots,
                   torch::tensor({43, 44, 45, 85, 86, 87}, torch::kInt)));
}

TEST(MtpAsyncStateTest,
     PatchesModelManagedTargetVerifyInputWithZeroCacheSlots) {
  const torch::TensorOptions position_options =
      torch::TensorOptions().dtype(torch::kInt).device(torch::kCPU);
  MtpTargetVerifyPatchWorkspace workspace =
      allocate_mtp_target_verify_patch_workspace(
          /*max_rows=*/4,
          /*max_verify_width=*/4,
          /*max_block_table_width=*/3,
          position_options,
          position_options);

  MtpTargetVerifyPatchTarget target;
  target.token_ids = torch::empty({6}, torch::kLong);
  target.positions = torch::empty({6}, torch::kInt);
  target.kv_seq_lens = torch::empty({6}, torch::kInt);
  target.new_cache_slots = torch::full({6}, -1, torch::kInt);
  target.model_managed_multiblock = true;
  const std::vector<torch::Tensor> draft_sources = {
      torch::tensor({101, 201}, torch::kLong),
      torch::tensor({102, 202}, torch::kLong)};

  patch_mtp_target_verify_input(torch::tensor({100, 200}, torch::kLong),
                                draft_sources,
                                torch::tensor({3, 5}, torch::kInt),
                                torch::tensor({4, 6}, torch::kInt),
                                /*block_size=*/4,
                                target,
                                workspace);

  EXPECT_TRUE(torch::equal(
      target.token_ids,
      torch::tensor({100, 101, 102, 200, 201, 202}, torch::kLong)));
  EXPECT_TRUE(torch::equal(target.positions,
                           torch::tensor({3, 4, 5, 5, 6, 7}, torch::kInt)));
  EXPECT_TRUE(torch::equal(target.kv_seq_lens,
                           torch::tensor({4, 5, 6, 6, 7, 8}, torch::kInt)));
  EXPECT_TRUE(
      torch::equal(target.new_cache_slots, torch::zeros({6}, torch::kInt)));
}

TEST(MtpAsyncStateTest,
     PatchesChunkedTargetVerifyExpandedMetadataWithoutReplacingStorage) {
  const torch::TensorOptions position_options =
      torch::TensorOptions().dtype(torch::kInt).device(torch::kCPU);
  MtpTargetVerifyPatchWorkspace workspace =
      allocate_mtp_target_verify_patch_workspace(
          /*max_rows=*/2,
          /*max_verify_width=*/3,
          /*max_block_table_width=*/3,
          position_options,
          position_options);

  MtpTargetVerifyPatchTarget target;
  target.token_ids = torch::empty({6}, torch::kLong);
  target.positions = torch::empty({6}, torch::kInt);
  target.kv_seq_lens = torch::empty({2}, torch::kInt);
  target.new_cache_slots = torch::empty({6}, torch::kInt);
  target.block_tables = torch::tensor({{10, 11, 12},
                                       {10, 11, 12},
                                       {10, 11, 12},
                                       {20, 21, 22},
                                       {20, 21, 22},
                                       {20, 21, 22}},
                                      torch::kInt);
  target.expanded_kv_seq_lens = torch::empty({6}, torch::kInt);
  target.expanded_paged_kv_indptr = torch::empty({7}, torch::kInt);
  target.expanded_paged_kv_indices = torch::empty({36}, torch::kInt);
  target.expanded_paged_kv_last_page_len = torch::empty({6}, torch::kInt);
  target.use_chunked_prefill = true;
  const void* kv_seq_lens_address = target.kv_seq_lens.data_ptr();
  const void* expanded_kv_address = target.expanded_kv_seq_lens.data_ptr();
  const void* indptr_address = target.expanded_paged_kv_indptr.data_ptr();
  const void* indices_address = target.expanded_paged_kv_indices.data_ptr();
  const void* last_page_address =
      target.expanded_paged_kv_last_page_len.data_ptr();
  const std::vector<torch::Tensor> draft_sources = {
      torch::tensor({101, 201}, torch::kLong),
      torch::tensor({102, 202}, torch::kLong)};

  patch_mtp_target_verify_input(torch::tensor({100, 200}, torch::kLong),
                                draft_sources,
                                torch::tensor({3, 6}, torch::kInt),
                                torch::tensor({4, 7}, torch::kInt),
                                /*block_size=*/4,
                                target,
                                workspace);

  EXPECT_EQ(target.kv_seq_lens.data_ptr(), kv_seq_lens_address);
  EXPECT_EQ(target.expanded_kv_seq_lens.data_ptr(), expanded_kv_address);
  EXPECT_EQ(target.expanded_paged_kv_indptr.data_ptr(), indptr_address);
  EXPECT_EQ(target.expanded_paged_kv_indices.data_ptr(), indices_address);
  EXPECT_EQ(target.expanded_paged_kv_last_page_len.data_ptr(),
            last_page_address);
  EXPECT_TRUE(
      torch::equal(target.kv_seq_lens, torch::tensor({6, 9}, torch::kInt)));
  EXPECT_TRUE(torch::equal(target.expanded_kv_seq_lens,
                           torch::tensor({4, 5, 6, 7, 8, 9}, torch::kInt)));
  EXPECT_TRUE(torch::equal(target.expanded_paged_kv_indptr,
                           torch::tensor({0, 1, 3, 5, 7, 9, 12}, torch::kInt)));
  EXPECT_TRUE(torch::equal(target.expanded_paged_kv_last_page_len,
                           torch::tensor({4, 1, 2, 3, 4, 1}, torch::kInt)));
  EXPECT_TRUE(torch::equal(
      target.expanded_paged_kv_indices.narrow(
          /*dim=*/0, /*start=*/0, /*length=*/12),
      torch::tensor({10, 10, 11, 10, 11, 20, 21, 20, 21, 20, 21, 22},
                    torch::kInt)));
}

TEST(MtpAsyncStateTest,
     RejectsTargetVerifyDeviceAndDtypeMismatchesBeforeWrites) {
  const torch::TensorOptions token_options =
      torch::TensorOptions().dtype(torch::kLong).device(torch::kCPU);
  const torch::TensorOptions position_options =
      torch::TensorOptions().dtype(torch::kInt).device(torch::kCPU);
  MtpTargetVerifyPatchWorkspace workspace =
      allocate_mtp_target_verify_patch_workspace(
          /*max_rows=*/2,
          /*max_verify_width=*/3,
          /*max_block_table_width=*/2,
          position_options,
          position_options);

  MtpTargetVerifyPatchTarget target;
  target.token_ids = torch::empty({6}, token_options);
  target.positions = torch::empty({6}, position_options);
  target.kv_seq_lens = torch::empty({6}, position_options);
  target.new_cache_slots = torch::empty({6}, position_options);
  target.block_tables = torch::tensor(
      {{10, 11}, {10, 11}, {10, 11}, {20, 21}, {20, 21}, {20, 21}},
      position_options);
  const torch::Tensor continuation_tokens =
      torch::tensor({100, 200}, token_options);
  std::vector<torch::Tensor> draft_sources = {
      torch::tensor({101, 201}, token_options),
      torch::tensor({102, 202}, token_options)};
  const torch::Tensor base_positions = torch::tensor({3, 5}, position_options);
  const torch::Tensor base_kv_seq_lens =
      torch::tensor({4, 6}, position_options);

  target.positions =
      torch::empty({6}, position_options.device(torch::Device("meta")));
  EXPECT_DEATH(patch_mtp_target_verify_input(continuation_tokens,
                                             draft_sources,
                                             base_positions,
                                             base_kv_seq_lens,
                                             /*block_size=*/4,
                                             target,
                                             workspace),
               "target.positions.*contract device");

  target.positions = torch::empty({6}, position_options);
  draft_sources[1] = torch::tensor({102, 202}, torch::kInt);
  EXPECT_DEATH(patch_mtp_target_verify_input(continuation_tokens,
                                             draft_sources,
                                             base_positions,
                                             base_kv_seq_lens,
                                             /*block_size=*/4,
                                             target,
                                             workspace),
               "draft_token_sources.*incompatible dtype");
}

TEST(MtpAsyncStateTest, MapsDraftTokensToTargetWithoutReplacingStorage) {
  const torch::Tensor draft_to_target_token_ids =
      torch::tensor({10, 20, 30, 40, 50}, torch::kLong);
  const torch::Tensor draft_token_ids = torch::tensor({3, 1, 4}, torch::kLong);
  torch::Tensor target_token_ids = torch::empty({3}, torch::kLong);
  const void* target_address = target_token_ids.data_ptr();

  map_draft_token_ids_to_target_out(
      draft_to_target_token_ids, draft_token_ids, target_token_ids);

  EXPECT_EQ(target_token_ids.data_ptr(), target_address);
  EXPECT_TRUE(torch::equal(target_token_ids,
                           torch::tensor({40, 20, 50}, torch::kLong)));
}

TEST(MtpAsyncStateTest, MapsDraftTokensIntoStridedFixedColumn) {
  const torch::Tensor draft_to_target_token_ids =
      torch::tensor({10, 20, 30, 40, 50}, torch::kLong);
  const torch::Tensor draft_token_ids = torch::tensor({3, 1, 4}, torch::kLong);
  torch::Tensor fixed_token_matrix = torch::full({3, 2}, -1, torch::kLong);
  torch::Tensor target_token_ids =
      fixed_token_matrix.select(/*dim=*/1, /*index=*/1);
  ASSERT_FALSE(target_token_ids.is_contiguous());
  const void* target_address = target_token_ids.data_ptr();

  for (int32_t iteration = 0; iteration < 100; ++iteration) {
    map_draft_token_ids_to_target_out(
        draft_to_target_token_ids, draft_token_ids, target_token_ids);
    EXPECT_EQ(target_token_ids.data_ptr(), target_address);
  }

  EXPECT_TRUE(torch::equal(
      fixed_token_matrix,
      torch::tensor({{-1, 40}, {-1, 20}, {-1, 50}}, torch::kLong)));
}

TEST(MtpAsyncStateTest, CopiesDraftTokensIntoStridedFixedColumn) {
  const torch::Tensor draft_token_ids = torch::tensor({3, 1, 4}, torch::kLong);
  torch::Tensor fixed_token_matrix = torch::full({3, 2}, -1, torch::kLong);
  torch::Tensor target_token_ids =
      fixed_token_matrix.select(/*dim=*/1, /*index=*/0);
  ASSERT_FALSE(target_token_ids.is_contiguous());
  const void* target_address = target_token_ids.data_ptr();

  copy_prepared_draft_token_ids_out(draft_token_ids, target_token_ids);

  EXPECT_EQ(target_token_ids.data_ptr(), target_address);
  EXPECT_TRUE(
      torch::equal(fixed_token_matrix,
                   torch::tensor({{3, -1}, {1, -1}, {4, -1}}, torch::kLong)));
}

TEST(MtpAsyncStateTest, BuildsTargetMetadataWithoutEmbeddingGather) {
  const torch::Tensor accepted_tokens = torch::tensor(
      {{10, 11, 12, 13}, {20, 21, -1, -1}, {30, -1, -1, -1}}, torch::kLong);
  const torch::Tensor base_positions = torch::tensor({100, 200, 300});
  const torch::Tensor base_kv_seq_lens = torch::tensor({101, 201, 301});

  const AcceptedTokenMetadata metadata = build_accepted_token_metadata(
      accepted_tokens, base_positions, base_kv_seq_lens);

  EXPECT_TRUE(torch::equal(metadata.accepted_lengths,
                           torch::tensor({4, 2, 1}, torch::kLong)));
  EXPECT_TRUE(torch::equal(metadata.last_tokens, torch::tensor({13, 21, 30})));
  EXPECT_TRUE(torch::equal(metadata.base_positions,
                           torch::tensor({104, 202, 301}, torch::kLong)));
  EXPECT_TRUE(torch::equal(metadata.base_kv_seq_lens,
                           torch::tensor({105, 203, 302}, torch::kLong)));
}

TEST(MtpAsyncStateTest, BuildsRowMetadataForChunkedAndDecodeLayouts) {
  AcceptedState state;
  state.base_positions = torch::tensor({104, 202, 301}, torch::kLong);
  state.base_kv_seq_lens = torch::tensor({105, 203, 302}, torch::kLong);
  const torch::Tensor offsets = torch::tensor({-1, 0}, torch::kLong);

  EXPECT_TRUE(torch::equal(
      make_row_positions(state, offsets),
      torch::tensor({{103, 104}, {201, 202}, {300, 301}}, torch::kLong)));
  EXPECT_TRUE(torch::equal(
      make_kv_seq_lens(state, offsets, /*use_chunked_prefill=*/true),
      state.base_kv_seq_lens));
  EXPECT_TRUE(torch::equal(
      make_kv_seq_lens(state, offsets, /*use_chunked_prefill=*/false),
      torch::tensor({104, 105, 202, 203, 301, 302}, torch::kLong)));
}

TEST(MtpAsyncStateTest, RedirectsUnusedRepairRowsToScratchPositions) {
  AcceptedState state;
  state.base_positions = torch::tensor({104, 202, 301}, torch::kLong);
  state.all_draft_accepted = torch::tensor({true, false, false});

  EXPECT_TRUE(torch::equal(make_repair_cache_positions(state),
                           torch::tensor({103, 203, 302}, torch::kLong)));
}

TEST(MtpAsyncStateTest, MapsPositionsAcrossCacheBlockBoundaries) {
  const torch::Tensor block_tables =
      torch::tensor({{10, 11, 12}, {20, 21, 22}}, torch::kInt);
  const torch::Tensor positions = torch::tensor({{3, 4}, {7, 8}}, torch::kLong);

  EXPECT_TRUE(torch::equal(
      map_positions_to_cache_slots(block_tables, positions, /*block_size=*/4),
      torch::tensor({43, 44, 87, 88}, torch::kInt)));
}

TEST(MtpAsyncStateTest, BuildsLaterDraftMetadataFromAcceptedDeviceBase) {
  AcceptedState state;
  state.base_positions = torch::tensor({3, 7}, torch::kLong);
  state.base_kv_seq_lens = torch::tensor({4, 8}, torch::kInt);
  const torch::Tensor offsets = torch::tensor({2}, torch::kLong);
  const torch::Tensor positions = make_row_positions(state, offsets);
  const torch::Tensor block_tables =
      torch::tensor({{10, 11, 12}, {20, 21, 22}}, torch::kInt);

  EXPECT_TRUE(torch::equal(positions, torch::tensor({{5}, {9}}, torch::kLong)));
  EXPECT_TRUE(torch::equal(
      make_kv_seq_lens(state, offsets, /*use_chunked_prefill=*/false),
      torch::tensor({6, 10}, torch::kLong)));
  EXPECT_TRUE(torch::equal(
      map_positions_to_cache_slots(block_tables, positions, /*block_size=*/4),
      torch::tensor({45, 89}, torch::kInt)));
}

}  // namespace
}  // namespace xllm::mtp_async
