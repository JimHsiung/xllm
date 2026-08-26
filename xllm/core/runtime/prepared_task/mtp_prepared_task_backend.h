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

#include "core/framework/speculative/mtp_async_state.h"
#include "runtime/prepared_task/mtp_prepared_task_plan.h"

namespace xllm {

// Narrow execution contract for the pure-MTP Prepared path. The backend owns
// Slot-local arenas, DeviceStepState, Patch workspace, model bindings, and
// stream Events. Prepare and Launch may call the backend concurrently for
// different Slots. The Adapter owns Task classification and fixed topology.
class MtpPreparedTaskBackend {
 public:
  virtual ~MtpPreparedTaskBackend() = default;

  virtual void initialize_state_thread() {}
  virtual void initialize_launch_thread() {}
  virtual void prepare_slot_for_reuse(ExecutionSlot& /*slot*/) {}

  virtual void prepare(const ForwardInput& input,
                       ExecutionSlot& slot,
                       const MtpPreparedTaskPlan& plan) = 0;
  virtual void launch(const MtpPreparedInvocation& invocation,
                      ExecutionSlot& slot,
                      ExecutionSlot* predecessor_slot) = 0;
  virtual std::optional<ForwardOutput> consume(ExecutionSlot& slot) = 0;
};

// Startup-only dimensions and dtypes for all fixed Slot storage. Primitive
// fields are explicit so a Worker-specific backend cannot silently infer a
// different capacity for one Slot.
struct MtpPreparedTaskBufferConfig {
  torch::Device device = torch::Device(torch::kCPU);
  uint64_t input_arena_capacity_bytes = 0;
  int32_t slot_count = 0;
  int64_t max_rows = 0;
  int64_t max_input_tokens = 0;
  int64_t accepted_token_capacity = 0;
  int64_t max_verify_width = 0;
  int64_t max_block_table_width = 0;
  int64_t embedding_size = 0;
  torch::ScalarType token_dtype = torch::kLong;
  torch::ScalarType embedding_dtype = torch::kFloat;
  torch::ScalarType position_dtype = torch::kInt;
};

// Fixed-address rejection outputs bound by the Worker-specific Prepare step.
// The Tensor handles may be replaced between Tasks, but their destinations in
// MtpDeviceStepState and MtpPublishPatchWorkspace never are.
struct MtpPreparedPublishSource {
  torch::Tensor accepted_tokens;
  torch::Tensor accepted_embeddings;
  torch::Tensor embedding_placeholder;
  torch::Tensor base_positions;
  torch::Tensor base_kv_seq_lens;
};

// Common resource and lifecycle implementation for MTP Worker backends.
// Derived classes only build fixed Draft/Target bindings in Prepare and submit
// model/collective invocations in Launch. Cross-Task continuation, state
// publication, Event ordering, and Host Consume stay centralized here.
class MtpPreparedTaskBackendBase : public MtpPreparedTaskBackend {
 public:
  explicit MtpPreparedTaskBackendBase(
      const MtpPreparedTaskBufferConfig& config);
  ~MtpPreparedTaskBackendBase() override;

  MtpPreparedTaskBackendBase(const MtpPreparedTaskBackendBase&) = delete;
  MtpPreparedTaskBackendBase& operator=(const MtpPreparedTaskBackendBase&) =
      delete;

  void prepare_slot_for_reuse(ExecutionSlot& slot) final;
  void prepare(const ForwardInput& input,
               ExecutionSlot& slot,
               const MtpPreparedTaskPlan& plan) final;
  void launch(const MtpPreparedInvocation& invocation,
              ExecutionSlot& slot,
              ExecutionSlot* predecessor_slot) final;
  std::optional<ForwardOutput> consume(ExecutionSlot& slot) final;

 protected:
  void bind_continuation_state(
      int32_t slot_id,
      const mtp_async::MtpContinuationState& continuation);
  void bind_publish_source(int32_t slot_id,
                           const MtpPreparedPublishSource& source);
  mtp_async::MtpDeviceStepState& mutable_device_step_state(int32_t slot_id);
  const mtp_async::MtpDeviceStepState& device_step_state(int32_t slot_id) const;
  mtp_async::MtpNextDraftPatchWorkspace& mutable_next_draft_patch_workspace(
      int32_t slot_id);
  mtp_async::MtpTargetVerifyPatchWorkspace&
  mutable_target_verify_patch_workspace(int32_t slot_id);
  bool stage_primary_input(int32_t slot_id,
                           const ForwardInput& input,
                           ForwardInput& staged_input);
  bool stage_invocation_input(int32_t slot_id,
                              const ForwardInput& input,
                              ForwardInput& staged_input);

  virtual void prepare_staged_input(const ForwardInput& input,
                                    ExecutionSlot& slot,
                                    const MtpPreparedTaskPlan& plan) = 0;
  virtual void launch_model_invocation(const MtpPreparedInvocation& invocation,
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
