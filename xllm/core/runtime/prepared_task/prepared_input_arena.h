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

#include "runtime/forward_params.h"

namespace xllm {

// Returns true only for a speculative Target Verify input whose dynamic model
// tensors are fully bound to one Prepared Slot Arena. Expanded verify also
// requires fixed-capacity tokenwise KV/block/paged metadata; graph-local tiling
// and static causal-conv tasks are bound separately by the ACL Graph executor.
bool prepared_spec_verify_graph_contract_is_complete(const ForwardInput& input);

class PreparedInputArena final {
 public:
  PreparedInputArena(const torch::Device& device, uint64_t capacity_bytes);

  // Starts a new Task layout and stages its first input at the beginning of
  // the fixed Arena. Subsequent model invocations in the same Task must use
  // stage_next() so every input owns a non-overlapping, stable subrange.
  bool stage(const ForwardInput& input, ForwardInput& staged_input);
  void begin_partitioned_task(int32_t partition_count);
  bool stage_next(const ForwardInput& input, ForwardInput& staged_input);
  // Rebinds metadata created by Worker prepare after stage(). This is only
  // needed by the single-invocation LLM adapter; speculative invocation
  // builders finish their metadata before stage_next().
  bool stage_generated_metadata(ForwardInput& staged_input);

  uint64_t capacity_bytes() const { return capacity_bytes_; }
  uint64_t used_bytes() const { return used_bytes_; }

 private:
  torch::Device device_;
  uint64_t capacity_bytes_ = 0;
  uint64_t used_bytes_ = 0;
  uint64_t partition_stride_bytes_ = 0;
  int32_t partition_count_ = 0;
  int32_t next_partition_ = 0;
  torch::Tensor host_buffer_;
  torch::Tensor device_buffer_;
  detail::ForwardInputBufferPlan input_plan_;
  detail::ForwardInputBufferPlan host_metadata_plan_;
};

}  // namespace xllm
