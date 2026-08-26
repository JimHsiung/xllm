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
#include <vector>

namespace xllm::dspark_detail {

// Stable views consumed by one DSpark Markov sample/broadcast descriptor.
// The full-vocabulary buffers are shared across steps because the descriptors
// execute sequentially on one stream; token-chain buffers remain step-local.
struct PreparedSamplingStepWorkspace {
  torch::Tensor previous_token_ids;
  torch::Tensor greedy_token_output;
  torch::Tensor token_ids;
  torch::Tensor proposal_probs;
  torch::Tensor markov_embeddings;
  torch::Tensor markov_bias;
  torch::Tensor step_logits;
};

// Slot-local fixed storage for DSpark's statically expanded Markov loop.
struct PreparedSamplingWorkspace {
  torch::Tensor token_ids;
  torch::Tensor previous_token_ids;
  torch::Tensor greedy_token_outputs;
  torch::Tensor proposal_probs;
  torch::Tensor markov_embeddings;
  torch::Tensor markov_bias;
  torch::Tensor step_logits;
  std::vector<PreparedSamplingStepWorkspace> steps;
};

PreparedSamplingWorkspace allocate_prepared_sampling_workspace(
    int64_t max_rows,
    int32_t speculative_width,
    int64_t markov_rank,
    int64_t draft_vocab_size,
    const torch::TensorOptions& token_options,
    const torch::TensorOptions& logit_options);

void reset_prepared_sampling_workspace(PreparedSamplingWorkspace& workspace);

// Writes the fixed previous-token view for one descriptor. Step zero consumes
// the Task anchor; later steps consume the preceding broadcast output.
void prepare_sampling_step(const torch::Tensor& anchor_token_ids,
                           int64_t row_count,
                           int32_t block_step,
                           PreparedSamplingWorkspace& workspace);

// Publishes the broadcast-complete greedy output to the fixed [B, N] token
// matrix and its greedy proposal-probability column.
void commit_sampling_step(int64_t row_count,
                          int32_t block_step,
                          PreparedSamplingWorkspace& workspace);

}  // namespace xllm::dspark_detail
