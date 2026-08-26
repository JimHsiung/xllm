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
#include <memory>
#include <optional>

#include "core/framework/speculative/block_spec_async_state.h"
#include "runtime/prepared_task/block_spec_prepared_task_plan.h"

namespace xllm {

// Narrow contract for DFlash/DSpark Worker backends. The Adapter fixes Task
// topology; the backend owns Slot storage, model bindings, Patch submission,
// Context-KV publication, and stream Events.
class BlockSpecPreparedTaskBackend {
 public:
  virtual ~BlockSpecPreparedTaskBackend() = default;

  virtual void initialize_state_thread() {}
  virtual void initialize_launch_thread() {}
  virtual void prepare_slot_for_reuse(ExecutionSlot& /*slot*/) {}

  virtual void prepare(const ForwardInput& input,
                       ExecutionSlot& slot,
                       const BlockSpecPreparedTaskPlan& plan) = 0;
  virtual void launch(const BlockSpecPreparedInvocation& invocation,
                      ExecutionSlot& slot,
                      ExecutionSlot* predecessor_slot) = 0;
  virtual std::optional<ForwardOutput> consume(ExecutionSlot& slot) = 0;
};

struct BlockSpecPreparedTaskBufferConfig {
  torch::Device device = torch::Device(torch::kCPU);
  uint64_t input_arena_capacity_bytes = 0;
  int32_t slot_count = 0;
  int32_t block_size = 0;
  int64_t max_rows = 0;
  int64_t accepted_token_capacity = 0;
  int64_t context_hidden_size = 0;
  torch::ScalarType token_dtype = torch::kLong;
  torch::ScalarType hidden_dtype = torch::kFloat;
  torch::ScalarType position_dtype = torch::kInt;
  torch::ScalarType cache_slot_dtype = torch::kInt;
};

// Rejection outputs and fixed metadata bound by the Worker-specific Prepare
// step. Tensor handles can change between Tasks; all base-owned destinations
// remain at their startup addresses.
struct BlockSpecPreparedPublishSource {
  torch::Tensor accepted_tokens;
  torch::Tensor accepted_context_hidden;
  torch::Tensor base_positions;
  torch::Tensor block_tables;
  block_spec_async::CacheSlotMappingMode cache_slot_mapping_mode =
      block_spec_async::CacheSlotMappingMode::LINEAR;
};

// Common fixed-storage and lifecycle implementation. Derived Worker backends
// only stage/bind algorithm inputs, apply the continuation to their fixed
// block-draft input, and dispatch individual invocations.
class BlockSpecPreparedTaskBackendBase : public BlockSpecPreparedTaskBackend {
 public:
  explicit BlockSpecPreparedTaskBackendBase(
      const BlockSpecPreparedTaskBufferConfig& config);
  ~BlockSpecPreparedTaskBackendBase() override;

  BlockSpecPreparedTaskBackendBase(const BlockSpecPreparedTaskBackendBase&) =
      delete;
  BlockSpecPreparedTaskBackendBase& operator=(
      const BlockSpecPreparedTaskBackendBase&) = delete;

  void prepare_slot_for_reuse(ExecutionSlot& slot) final;
  void prepare(const ForwardInput& input,
               ExecutionSlot& slot,
               const BlockSpecPreparedTaskPlan& plan) final;
  void launch(const BlockSpecPreparedInvocation& invocation,
              ExecutionSlot& slot,
              ExecutionSlot* predecessor_slot) final;
  std::optional<ForwardOutput> consume(ExecutionSlot& slot) final;

 protected:
  void bind_publish_source(int32_t slot_id,
                           const BlockSpecPreparedPublishSource& source);
  block_spec_async::BlockSpecDeviceStepState& mutable_device_step_state(
      int32_t slot_id);
  const block_spec_async::BlockSpecDeviceStepState& device_step_state(
      int32_t slot_id) const;
  void patch_continuation_state(
      int32_t slot_id,
      const block_spec_async::BlockSpecDeviceStepState& predecessor_state,
      const torch::Tensor& predecessor_rows,
      block_spec_async::BlockSpecContinuationState& continuation);
  bool stage_primary_input(int32_t slot_id,
                           const ForwardInput& input,
                           ForwardInput& staged_input);
  bool stage_invocation_input(int32_t slot_id,
                              const ForwardInput& input,
                              ForwardInput& staged_input);

  virtual void prepare_staged_input(const ForwardInput& input,
                                    ExecutionSlot& slot,
                                    const BlockSpecPreparedTaskPlan& plan) = 0;
  virtual void launch_predecessor_patch(
      const block_spec_async::BlockSpecDeviceStepState& predecessor_state,
      ExecutionSlot& slot) = 0;
  virtual void launch_model_invocation(
      const BlockSpecPreparedInvocation& invocation,
      ExecutionSlot& slot) = 0;
  virtual c10::Stream task_stream() const = 0;
  virtual StreamEventPtr record_task_stream_event() const = 0;
  virtual bool wait_prepare_stream_event(const StreamEventPtr& event) const = 0;
  virtual void consume_ready(ExecutionSlot& /*slot*/) {}

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace xllm
