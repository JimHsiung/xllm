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

#include "runtime/prepared_task/dspark_prepared_sampling.h"

#include <glog/logging.h>

#include <cstddef>

namespace xllm::dspark_detail {
namespace {

PreparedSamplingStepWorkspace& mutable_step(
    int32_t block_step,
    PreparedSamplingWorkspace& workspace) {
  CHECK_GE(block_step, 0);
  CHECK_LT(static_cast<size_t>(block_step), workspace.steps.size());
  return workspace.steps[static_cast<size_t>(block_step)];
}

void check_sampling_tensor_contract(const torch::Tensor& tensor,
                                    const torch::Device& expected_device,
                                    torch::ScalarType expected_dtype,
                                    const char* tensor_name) {
  CHECK(tensor.defined()) << "Prepared DSpark sampling " << tensor_name
                          << " must be defined";
  CHECK_EQ(tensor.device(), expected_device)
      << "Prepared DSpark sampling " << tensor_name
      << " must share the contract device";
  CHECK_EQ(tensor.scalar_type(), expected_dtype)
      << "Prepared DSpark sampling " << tensor_name
      << " has an incompatible dtype";
}

void check_sampling_step_contract(const PreparedSamplingStepWorkspace& step,
                                  const torch::Device& expected_device,
                                  torch::ScalarType logit_dtype) {
  check_sampling_tensor_contract(step.previous_token_ids,
                                 expected_device,
                                 torch::kLong,
                                 "step.previous_token_ids");
  check_sampling_tensor_contract(step.greedy_token_output,
                                 expected_device,
                                 torch::kLong,
                                 "step.greedy_token_output");
  check_sampling_tensor_contract(
      step.token_ids, expected_device, torch::kLong, "step.token_ids");
  check_sampling_tensor_contract(step.proposal_probs,
                                 expected_device,
                                 torch::kFloat32,
                                 "step.proposal_probs");
  check_sampling_tensor_contract(step.markov_embeddings,
                                 expected_device,
                                 logit_dtype,
                                 "step.markov_embeddings");
  check_sampling_tensor_contract(
      step.markov_bias, expected_device, logit_dtype, "step.markov_bias");
  check_sampling_tensor_contract(
      step.step_logits, expected_device, logit_dtype, "step.step_logits");
}

void check_sampling_workspace_contract(
    const PreparedSamplingWorkspace& workspace,
    bool check_all_steps) {
  CHECK(workspace.token_ids.defined())
      << "Prepared DSpark sampling workspace.token_ids must be defined";
  CHECK(workspace.step_logits.defined())
      << "Prepared DSpark sampling workspace.step_logits must be defined";
  const torch::Device expected_device = workspace.token_ids.device();
  const torch::ScalarType logit_dtype = workspace.step_logits.scalar_type();
  check_sampling_tensor_contract(workspace.token_ids,
                                 expected_device,
                                 torch::kLong,
                                 "workspace.token_ids");
  check_sampling_tensor_contract(workspace.previous_token_ids,
                                 expected_device,
                                 torch::kLong,
                                 "workspace.previous_token_ids");
  check_sampling_tensor_contract(workspace.greedy_token_outputs,
                                 expected_device,
                                 torch::kLong,
                                 "workspace.greedy_token_outputs");
  check_sampling_tensor_contract(workspace.proposal_probs,
                                 expected_device,
                                 torch::kFloat32,
                                 "workspace.proposal_probs");
  check_sampling_tensor_contract(workspace.markov_embeddings,
                                 expected_device,
                                 logit_dtype,
                                 "workspace.markov_embeddings");
  check_sampling_tensor_contract(workspace.markov_bias,
                                 expected_device,
                                 logit_dtype,
                                 "workspace.markov_bias");
  check_sampling_tensor_contract(workspace.step_logits,
                                 expected_device,
                                 logit_dtype,
                                 "workspace.step_logits");
  CHECK_EQ(workspace.steps.size(),
           static_cast<size_t>(workspace.token_ids.size(/*dim=*/1)))
      << "Prepared DSpark sampling step count must match token width";
  if (check_all_steps) {
    for (const PreparedSamplingStepWorkspace& step : workspace.steps) {
      check_sampling_step_contract(step, expected_device, logit_dtype);
    }
  }
}

}  // namespace

PreparedSamplingWorkspace allocate_prepared_sampling_workspace(
    int64_t max_rows,
    int32_t speculative_width,
    int64_t markov_rank,
    int64_t draft_vocab_size,
    const torch::TensorOptions& token_options,
    const torch::TensorOptions& logit_options) {
  CHECK_GT(max_rows, 0);
  CHECK_GT(speculative_width, 0);
  CHECK_GT(markov_rank, 0);
  CHECK_GT(draft_vocab_size, 0);
  CHECK_EQ(token_options.device(), logit_options.device());
  CHECK_EQ(token_options.dtype(), torch::kLong);

  PreparedSamplingWorkspace workspace;
  workspace.token_ids =
      torch::empty({max_rows, speculative_width}, token_options);
  workspace.previous_token_ids =
      torch::empty({speculative_width, max_rows}, token_options);
  workspace.greedy_token_outputs =
      torch::empty({speculative_width, max_rows}, token_options);
  workspace.proposal_probs = torch::ones({max_rows, speculative_width},
                                         logit_options.dtype(torch::kFloat32));
  workspace.markov_embeddings =
      torch::empty({max_rows, markov_rank}, logit_options);
  workspace.markov_bias =
      torch::empty({max_rows, draft_vocab_size}, logit_options);
  workspace.step_logits =
      torch::empty({max_rows, draft_vocab_size}, logit_options);

  workspace.steps.reserve(static_cast<size_t>(speculative_width));
  for (int32_t block_step = 0; block_step < speculative_width; ++block_step) {
    workspace.steps.emplace_back(PreparedSamplingStepWorkspace{
        workspace.previous_token_ids.select(/*dim=*/0, block_step),
        workspace.greedy_token_outputs.select(/*dim=*/0, block_step),
        workspace.token_ids.select(/*dim=*/1, block_step),
        workspace.proposal_probs.select(/*dim=*/1, block_step),
        workspace.markov_embeddings,
        workspace.markov_bias,
        workspace.step_logits});
  }
  reset_prepared_sampling_workspace(workspace);
  return workspace;
}

void reset_prepared_sampling_workspace(PreparedSamplingWorkspace& workspace) {
  check_sampling_workspace_contract(workspace, /*check_all_steps=*/true);
  workspace.token_ids.fill_(-1);
  workspace.previous_token_ids.fill_(-1);
  workspace.greedy_token_outputs.fill_(-1);
  workspace.proposal_probs.fill_(1.0F);
}

void prepare_sampling_step(const torch::Tensor& anchor_token_ids,
                           int64_t row_count,
                           int32_t block_step,
                           PreparedSamplingWorkspace& workspace) {
  check_sampling_workspace_contract(workspace, /*check_all_steps=*/false);
  CHECK(anchor_token_ids.defined());
  CHECK_EQ(anchor_token_ids.dim(), 1);
  CHECK_EQ(anchor_token_ids.scalar_type(), torch::kLong);
  CHECK_GT(row_count, 0);
  CHECK_EQ(anchor_token_ids.numel(), row_count);
  PreparedSamplingStepWorkspace& step = mutable_step(block_step, workspace);
  check_sampling_step_contract(
      step, workspace.token_ids.device(), workspace.step_logits.scalar_type());
  CHECK_LE(row_count, step.previous_token_ids.numel());
  CHECK_EQ(anchor_token_ids.device(), workspace.token_ids.device())
      << "Prepared DSpark sampling anchor_token_ids must share the contract "
         "device";

  torch::Tensor previous_token_ids =
      step.previous_token_ids.narrow(/*dim=*/0, /*start=*/0, row_count);
  if (block_step == 0) {
    previous_token_ids.copy_(anchor_token_ids, /*non_blocking=*/true);
    return;
  }
  const PreparedSamplingStepWorkspace& previous_step =
      workspace.steps[static_cast<size_t>(block_step - 1)];
  check_sampling_step_contract(previous_step,
                               workspace.token_ids.device(),
                               workspace.step_logits.scalar_type());
  previous_token_ids.copy_(previous_step.greedy_token_output.narrow(
                               /*dim=*/0, /*start=*/0, row_count),
                           /*non_blocking=*/true);
}

void commit_sampling_step(int64_t row_count,
                          int32_t block_step,
                          PreparedSamplingWorkspace& workspace) {
  check_sampling_workspace_contract(workspace, /*check_all_steps=*/false);
  CHECK_GT(row_count, 0);
  PreparedSamplingStepWorkspace& step = mutable_step(block_step, workspace);
  check_sampling_step_contract(
      step, workspace.token_ids.device(), workspace.step_logits.scalar_type());
  CHECK_LE(row_count, step.greedy_token_output.numel());
  step.token_ids.narrow(/*dim=*/0, /*start=*/0, row_count)
      .copy_(step.greedy_token_output.narrow(
                 /*dim=*/0, /*start=*/0, row_count),
             /*non_blocking=*/true);
  step.proposal_probs.narrow(/*dim=*/0, /*start=*/0, row_count).fill_(1.0F);
}

}  // namespace xllm::dspark_detail
