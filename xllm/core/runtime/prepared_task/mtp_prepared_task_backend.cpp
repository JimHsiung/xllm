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

#include "runtime/prepared_task/mtp_prepared_task_backend.h"

#include <glog/logging.h>

#include <cstddef>
#include <utility>
#include <vector>

#include "runtime/prepared_task/prepared_input_arena.h"

namespace xllm {
namespace {

size_t checked_slot_index(int32_t slot_id, size_t slot_count) {
  CHECK_GE(slot_id, 0);
  const size_t slot_index = static_cast<size_t>(slot_id);
  CHECK_LT(slot_index, slot_count);
  return slot_index;
}

bool is_final_invocation(MtpPreparedInvocationKind kind) {
  return kind == MtpPreparedInvocationKind::PUBLISH_STEP_STATE ||
         kind == MtpPreparedInvocationKind::EMPTY_COLLECTIVE;
}

}  // namespace

class MtpPreparedTaskBackendBase::Impl final {
 public:
  explicit Impl(const MtpPreparedTaskBufferConfig& config) {
    CHECK(config.slot_count == 1 || config.slot_count == 2);
    CHECK_GT(config.input_arena_capacity_bytes, 0);
    CHECK_GT(config.max_rows, 0);
    CHECK_GT(config.accepted_token_capacity, 0);
    CHECK_GT(config.max_verify_width, 0);
    CHECK_EQ(config.accepted_token_capacity, config.max_verify_width)
        << "Prepared MTP accepted-token capacity must match the maximum "
           "Target Verify width";
    CHECK_GT(config.max_block_table_width, 0);
    CHECK_GT(config.embedding_size, 0);

    const torch::TensorOptions token_options =
        torch::TensorOptions().dtype(config.token_dtype).device(config.device);
    const torch::TensorOptions embedding_options =
        torch::TensorOptions()
            .dtype(config.embedding_dtype)
            .device(config.device);
    const torch::TensorOptions position_options =
        torch::TensorOptions()
            .dtype(config.position_dtype)
            .device(config.device);

    slots_.reserve(static_cast<size_t>(config.slot_count));
    for (int32_t slot_id = 0; slot_id < config.slot_count; ++slot_id) {
      SlotResources resources;
      resources.input_arena = std::make_unique<PreparedInputArena>(
          config.device, config.input_arena_capacity_bytes);
      resources.device_step_state = mtp_async::allocate_mtp_device_step_state(
          config.max_rows,
          config.accepted_token_capacity,
          config.embedding_size,
          token_options,
          embedding_options,
          position_options);
      resources.predecessor_workspace =
          mtp_async::allocate_mtp_predecessor_patch_workspace(
              config.max_rows,
              config.embedding_size,
              token_options,
              embedding_options,
              position_options);
      resources.publish_workspace =
          mtp_async::allocate_mtp_publish_patch_workspace(
              config.max_rows,
              config.accepted_token_capacity,
              config.embedding_size,
              token_options,
              embedding_options);
      resources.next_draft_workspace =
          mtp_async::allocate_mtp_next_draft_patch_workspace(
              config.max_rows, position_options, position_options);
      resources.target_verify_workspace =
          mtp_async::allocate_mtp_target_verify_patch_workspace(
              config.max_rows,
              config.max_verify_width,
              config.max_block_table_width,
              position_options,
              position_options);
      slots_.emplace_back(std::move(resources));
    }
  }

  struct SlotResources {
    std::unique_ptr<PreparedInputArena> input_arena;
    mtp_async::MtpDeviceStepState device_step_state;
    mtp_async::MtpPredecessorPatchWorkspace predecessor_workspace;
    mtp_async::MtpPublishPatchWorkspace publish_workspace;
    mtp_async::MtpNextDraftPatchWorkspace next_draft_workspace;
    mtp_async::MtpTargetVerifyPatchWorkspace target_verify_workspace;
    mtp_async::MtpContinuationState continuation;
    MtpPreparedPublishSource publish_source;
    bool continuation_bound = false;
    bool publish_source_bound = false;
    int32_t input_partition_count = 0;
    bool primary_input_staged = false;
  };

  SlotResources& slot(int32_t slot_id) {
    return slots_[checked_slot_index(slot_id, slots_.size())];
  }

  const SlotResources& slot(int32_t slot_id) const {
    return slots_[checked_slot_index(slot_id, slots_.size())];
  }

 private:
  std::vector<SlotResources> slots_;
};

MtpPreparedTaskBackendBase::MtpPreparedTaskBackendBase(
    const MtpPreparedTaskBufferConfig& config)
    : impl_(std::make_unique<Impl>(config)) {}

MtpPreparedTaskBackendBase::~MtpPreparedTaskBackendBase() = default;

void MtpPreparedTaskBackendBase::prepare_slot_for_reuse(ExecutionSlot& slot) {
  if (slot.reuse_fence == nullptr) {
    return;
  }
  CHECK(wait_prepare_stream_event(slot.reuse_fence))
      << "Failed to wait for the MTP predecessor-state reuse fence";
}

void MtpPreparedTaskBackendBase::prepare(const ForwardInput& input,
                                         ExecutionSlot& slot,
                                         const MtpPreparedTaskPlan& plan) {
  Impl::SlotResources& resources = impl_->slot(slot.slot_id);
  resources.continuation = mtp_async::MtpContinuationState();
  resources.publish_source = MtpPreparedPublishSource();
  resources.continuation_bound = false;
  resources.publish_source_bound = false;
  CHECK_GT(plan.input_partition_count, 0);
  resources.input_partition_count = plan.input_partition_count;
  resources.primary_input_staged = false;
  prepare_staged_input(input, slot, plan);
  CHECK(resources.primary_input_staged)
      << "MTP Worker Backend did not stage the primary Task input";
}

void MtpPreparedTaskBackendBase::launch(const MtpPreparedInvocation& invocation,
                                        ExecutionSlot& slot,
                                        ExecutionSlot* predecessor_slot) {
  c10::StreamGuard stream_guard(task_stream());
  Impl::SlotResources& resources = impl_->slot(slot.slot_id);
  if (invocation.kind == MtpPreparedInvocationKind::TASK_CONTINUATION_PATCH) {
    if (predecessor_slot == nullptr) {
      // A decode Task without an immediately preceding Prepared Task keeps
      // the Worker-provided initial rows. The fixed invocation remains in the
      // template, but it has no cross-Task state to gather.
      launch_model_invocation(invocation, slot);
      return;
    }
    CHECK(resources.continuation_bound)
        << "TASK_CONTINUATION requires fixed destination bindings";
    const Impl::SlotResources& predecessor_resources =
        impl_->slot(predecessor_slot->slot_id);
    mtp_async::patch_mtp_continuation_rows(
        predecessor_resources.device_step_state,
        slot.prepared_input.input_params.embedding.predecessor_rows,
        resources.continuation,
        resources.predecessor_workspace);
    launch_model_invocation(invocation, slot);
    predecessor_slot->reuse_fence = record_task_stream_event();
    CHECK(predecessor_slot->reuse_fence != nullptr)
        << "Failed to record the MTP predecessor-state reuse fence";
    return;
  }

  if (invocation.kind == MtpPreparedInvocationKind::PUBLISH_STEP_STATE) {
    CHECK(resources.publish_source_bound)
        << "MTP state publication requires fixed rejection-output bindings";
    const MtpPreparedPublishSource& source = resources.publish_source;
    CHECK(source.accepted_tokens.device() ==
          resources.device_step_state.accepted_tokens.device())
        << "MTP rejection output and DeviceStepState must share a device";
    CHECK(source.accepted_tokens.device() ==
          resources.publish_workspace.accepted_mask.device())
        << "MTP rejection output and Publish workspace must share a device";
    CHECK(source.accepted_embeddings.device() ==
          resources.device_step_state.tail_embeddings.device())
        << "MTP rejection embeddings and DeviceStepState must share a device";
    CHECK(source.embedding_placeholder.device() ==
          resources.device_step_state.tail_embeddings.device())
        << "MTP embedding placeholder and DeviceStepState must share a device";
    CHECK(source.base_positions.device() ==
          resources.device_step_state.base_positions.device())
        << "MTP base positions and DeviceStepState must share a device";
    CHECK(source.base_kv_seq_lens.device() ==
          resources.device_step_state.base_kv_seq_lens.device())
        << "MTP base KV lengths and DeviceStepState must share a device";
    CHECK_EQ(source.accepted_tokens.scalar_type(),
             resources.device_step_state.accepted_tokens.scalar_type())
        << "MTP rejection output and DeviceStepState must share a dtype";
    CHECK_EQ(source.accepted_embeddings.scalar_type(),
             resources.device_step_state.tail_embeddings.scalar_type())
        << "MTP rejection embeddings and DeviceStepState must share a dtype";
    CHECK_EQ(source.embedding_placeholder.scalar_type(),
             resources.device_step_state.tail_embeddings.scalar_type())
        << "MTP embedding placeholder and DeviceStepState must share a dtype";
    CHECK_EQ(source.base_positions.scalar_type(),
             resources.device_step_state.base_positions.scalar_type())
        << "MTP base positions and DeviceStepState must share a dtype";
    CHECK_EQ(source.base_kv_seq_lens.scalar_type(),
             resources.device_step_state.base_kv_seq_lens.scalar_type())
        << "MTP base KV lengths and DeviceStepState must share a dtype";
    mtp_async::publish_mtp_device_step_state_from_outputs(
        source.accepted_tokens,
        source.accepted_embeddings,
        source.embedding_placeholder,
        source.base_positions,
        source.base_kv_seq_lens,
        resources.device_step_state,
        resources.publish_workspace);
  }

  launch_model_invocation(invocation, slot);
  if (is_final_invocation(invocation.kind)) {
    slot.task_output_event = record_task_stream_event();
    CHECK(slot.task_output_event != nullptr)
        << "Failed to record the MTP Prepared task completion event";
    if (slot.output.has_value()) {
      slot.output->ready_event = slot.task_output_event;
    }
  }
}

std::optional<ForwardOutput> MtpPreparedTaskBackendBase::consume(
    ExecutionSlot& slot) {
  CHECK(slot.task_output_event != nullptr);
  CHECK(slot.task_output_event->synchronize())
      << "Failed to wait for MTP Prepared task output event";
  consume_ready(slot);
  if (slot.output.has_value()) {
    slot.output->retained_inputs.clear();
  }
  return std::move(slot.output);
}

void MtpPreparedTaskBackendBase::bind_continuation_state(
    int32_t slot_id,
    const mtp_async::MtpContinuationState& continuation) {
  Impl::SlotResources& resources = impl_->slot(slot_id);
  CHECK(continuation.tail_tokens.defined());
  CHECK(continuation.previous_tokens.defined());
  CHECK(continuation.tail_embeddings.defined());
  CHECK(continuation.previous_embeddings.defined());
  CHECK(continuation.base_positions.defined());
  CHECK(continuation.base_kv_seq_lens.defined());
  CHECK(continuation.tail_tokens.device() ==
        resources.device_step_state.tail_tokens.device())
      << "MTP continuation tokens and DeviceStepState must share a device";
  CHECK(continuation.previous_tokens.device() ==
        resources.device_step_state.previous_tokens.device())
      << "MTP previous tokens and DeviceStepState must share a device";
  CHECK(continuation.tail_embeddings.device() ==
        resources.device_step_state.tail_embeddings.device())
      << "MTP continuation embeddings and DeviceStepState must share a device";
  CHECK(continuation.previous_embeddings.device() ==
        resources.device_step_state.previous_embeddings.device())
      << "MTP previous embeddings and DeviceStepState must share a device";
  CHECK(continuation.base_positions.device() ==
        resources.device_step_state.base_positions.device())
      << "MTP continuation positions and DeviceStepState must share a device";
  CHECK(continuation.base_kv_seq_lens.device() ==
        resources.device_step_state.base_kv_seq_lens.device())
      << "MTP continuation KV lengths and DeviceStepState must share a device";
  CHECK_EQ(continuation.tail_tokens.scalar_type(),
           resources.device_step_state.tail_tokens.scalar_type())
      << "MTP continuation tokens and DeviceStepState must share a dtype";
  CHECK_EQ(continuation.previous_tokens.scalar_type(),
           resources.device_step_state.previous_tokens.scalar_type())
      << "MTP previous tokens and DeviceStepState must share a dtype";
  CHECK_EQ(continuation.tail_embeddings.scalar_type(),
           resources.device_step_state.tail_embeddings.scalar_type())
      << "MTP continuation embeddings and DeviceStepState must share a dtype";
  CHECK_EQ(continuation.previous_embeddings.scalar_type(),
           resources.device_step_state.previous_embeddings.scalar_type())
      << "MTP previous embeddings and DeviceStepState must share a dtype";
  CHECK_EQ(continuation.base_positions.scalar_type(),
           resources.device_step_state.base_positions.scalar_type())
      << "MTP continuation positions and DeviceStepState must share a dtype";
  CHECK_EQ(continuation.base_kv_seq_lens.scalar_type(),
           resources.device_step_state.base_kv_seq_lens.scalar_type())
      << "MTP continuation KV lengths and DeviceStepState must share a dtype";
  resources.continuation = continuation;
  resources.continuation_bound = true;
}

void MtpPreparedTaskBackendBase::bind_publish_source(
    int32_t slot_id,
    const MtpPreparedPublishSource& source) {
  Impl::SlotResources& resources = impl_->slot(slot_id);
  CHECK(source.accepted_tokens.defined());
  CHECK(source.accepted_embeddings.defined());
  CHECK(source.embedding_placeholder.defined());
  CHECK(source.base_positions.defined());
  CHECK(source.base_kv_seq_lens.defined());
  resources.publish_source = source;
  resources.publish_source_bound = true;
}

mtp_async::MtpDeviceStepState&
MtpPreparedTaskBackendBase::mutable_device_step_state(int32_t slot_id) {
  return impl_->slot(slot_id).device_step_state;
}

const mtp_async::MtpDeviceStepState&
MtpPreparedTaskBackendBase::device_step_state(int32_t slot_id) const {
  return impl_->slot(slot_id).device_step_state;
}

mtp_async::MtpNextDraftPatchWorkspace&
MtpPreparedTaskBackendBase::mutable_next_draft_patch_workspace(
    int32_t slot_id) {
  return impl_->slot(slot_id).next_draft_workspace;
}

mtp_async::MtpTargetVerifyPatchWorkspace&
MtpPreparedTaskBackendBase::mutable_target_verify_patch_workspace(
    int32_t slot_id) {
  return impl_->slot(slot_id).target_verify_workspace;
}

bool MtpPreparedTaskBackendBase::stage_primary_input(
    int32_t slot_id,
    const ForwardInput& input,
    ForwardInput& staged_input) {
  Impl::SlotResources& resources = impl_->slot(slot_id);
  CHECK_GT(resources.input_partition_count, 0);
  CHECK(!resources.primary_input_staged)
      << "MTP primary Task input may only be staged once";
  resources.input_arena->begin_partitioned_task(
      resources.input_partition_count);
  const bool staged = resources.input_arena->stage_next(input, staged_input);
  resources.primary_input_staged = staged;
  return staged;
}

bool MtpPreparedTaskBackendBase::stage_invocation_input(
    int32_t slot_id,
    const ForwardInput& input,
    ForwardInput& staged_input) {
  Impl::SlotResources& resources = impl_->slot(slot_id);
  CHECK(resources.primary_input_staged)
      << "MTP primary Task input must be staged before model invocations";
  return resources.input_arena->stage_next(input, staged_input);
}

}  // namespace xllm
