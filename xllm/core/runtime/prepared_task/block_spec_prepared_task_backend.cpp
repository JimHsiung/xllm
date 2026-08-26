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

#include "runtime/prepared_task/block_spec_prepared_task_backend.h"

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

bool is_final_invocation(BlockSpecPreparedInvocationKind kind) {
  return kind == BlockSpecPreparedInvocationKind::WRITE_CONTEXT_KV ||
         kind == BlockSpecPreparedInvocationKind::EMPTY_COLLECTIVE;
}

void check_block_spec_publish_source_contract(
    const BlockSpecPreparedPublishSource& source,
    const block_spec_async::BlockSpecDeviceStepState& destination) {
  CHECK(source.accepted_tokens.defined());
  CHECK(source.accepted_context_hidden.defined());
  CHECK(source.base_positions.defined());
  CHECK(source.block_tables.defined());
  const torch::Device expected_device = destination.accepted_tokens.device();
  CHECK_EQ(source.accepted_tokens.device(), expected_device)
      << "Block-Spec accepted tokens and DeviceStepState must share a device";
  CHECK_EQ(source.accepted_context_hidden.device(), expected_device)
      << "Block-Spec accepted context hidden and DeviceStepState must share "
         "a device";
  CHECK_EQ(source.base_positions.device(), expected_device)
      << "Block-Spec base positions and DeviceStepState must share a device";
  CHECK_EQ(source.block_tables.device(), expected_device)
      << "Block-Spec block tables and DeviceStepState must share a device";
  CHECK_EQ(source.accepted_tokens.scalar_type(),
           destination.accepted_tokens.scalar_type())
      << "Block-Spec accepted tokens and DeviceStepState must share a dtype";
  CHECK_EQ(source.accepted_context_hidden.scalar_type(),
           destination.context_hidden.scalar_type())
      << "Block-Spec accepted context hidden and DeviceStepState must share "
         "a dtype";
  CHECK_EQ(source.base_positions.scalar_type(),
           destination.context_positions.scalar_type())
      << "Block-Spec base positions and DeviceStepState must share a dtype";
  CHECK_EQ(source.block_tables.scalar_type(),
           destination.context_cache_slots.scalar_type())
      << "Block-Spec block tables and DeviceStepState must share a dtype";
}

}  // namespace

class BlockSpecPreparedTaskBackendBase::Impl final {
 public:
  explicit Impl(const BlockSpecPreparedTaskBufferConfig& config)
      : block_size_(config.block_size) {
    CHECK(config.slot_count == 1 || config.slot_count == 2);
    CHECK_GT(config.input_arena_capacity_bytes, 0);
    CHECK_GT(config.block_size, 0);
    CHECK_GT(config.max_rows, 0);
    CHECK_GT(config.accepted_token_capacity, 0);
    CHECK_GT(config.context_hidden_size, 0);

    const torch::TensorOptions token_options =
        torch::TensorOptions().dtype(config.token_dtype).device(config.device);
    const torch::TensorOptions hidden_options =
        torch::TensorOptions().dtype(config.hidden_dtype).device(config.device);
    const torch::TensorOptions position_options =
        torch::TensorOptions()
            .dtype(config.position_dtype)
            .device(config.device);
    const torch::TensorOptions cache_slot_options =
        torch::TensorOptions()
            .dtype(config.cache_slot_dtype)
            .device(config.device);

    slots_.reserve(static_cast<size_t>(config.slot_count));
    for (int32_t slot_id = 0; slot_id < config.slot_count; ++slot_id) {
      SlotResources resources;
      resources.input_arena = std::make_unique<PreparedInputArena>(
          config.device, config.input_arena_capacity_bytes);
      resources.device_step_state =
          block_spec_async::allocate_block_spec_device_step_state(
              config.max_rows,
              config.accepted_token_capacity,
              config.context_hidden_size,
              token_options,
              hidden_options,
              position_options,
              cache_slot_options);
      resources.context_kv_workspace =
          block_spec_async::allocate_block_spec_context_kv_patch_workspace(
              config.max_rows,
              config.accepted_token_capacity,
              position_options,
              cache_slot_options);
      resources.predecessor_workspace =
          block_spec_async::allocate_block_spec_predecessor_patch_workspace(
              config.max_rows,
              config.context_hidden_size,
              token_options,
              hidden_options,
              position_options);
      slots_.emplace_back(std::move(resources));
    }
  }

  struct SlotResources {
    std::unique_ptr<PreparedInputArena> input_arena;
    block_spec_async::BlockSpecDeviceStepState device_step_state;
    block_spec_async::BlockSpecContextKvPatchWorkspace context_kv_workspace;
    block_spec_async::BlockSpecPredecessorPatchWorkspace predecessor_workspace;
    BlockSpecPreparedPublishSource publish_source;
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

  int32_t block_size() const { return block_size_; }

 private:
  int32_t block_size_ = 0;
  std::vector<SlotResources> slots_;
};

BlockSpecPreparedTaskBackendBase::BlockSpecPreparedTaskBackendBase(
    const BlockSpecPreparedTaskBufferConfig& config)
    : impl_(std::make_unique<Impl>(config)) {}

BlockSpecPreparedTaskBackendBase::~BlockSpecPreparedTaskBackendBase() = default;

void BlockSpecPreparedTaskBackendBase::prepare_slot_for_reuse(
    ExecutionSlot& slot) {
  if (slot.reuse_fence == nullptr) {
    return;
  }
  CHECK(wait_prepare_stream_event(slot.reuse_fence))
      << "Failed to wait for the Block-Spec predecessor-state reuse fence";
}

void BlockSpecPreparedTaskBackendBase::prepare(
    const ForwardInput& input,
    ExecutionSlot& slot,
    const BlockSpecPreparedTaskPlan& plan) {
  Impl::SlotResources& resources = impl_->slot(slot.slot_id);
  resources.publish_source = BlockSpecPreparedPublishSource();
  resources.publish_source_bound = false;
  CHECK_GT(plan.input_partition_count, 0);
  resources.input_partition_count = plan.input_partition_count;
  resources.primary_input_staged = false;
  prepare_staged_input(input, slot, plan);
  CHECK(resources.primary_input_staged)
      << "Block-Spec Worker Backend did not stage the primary Task input";
}

void BlockSpecPreparedTaskBackendBase::launch(
    const BlockSpecPreparedInvocation& invocation,
    ExecutionSlot& slot,
    ExecutionSlot* predecessor_slot) {
  c10::StreamGuard stream_guard(task_stream());
  Impl::SlotResources& resources = impl_->slot(slot.slot_id);
  if (invocation.kind ==
      BlockSpecPreparedInvocationKind::TASK_CONTINUATION_PATCH) {
    if (predecessor_slot != nullptr) {
      const Impl::SlotResources& predecessor_resources =
          impl_->slot(predecessor_slot->slot_id);
      launch_predecessor_patch(predecessor_resources.device_step_state, slot);
      predecessor_slot->reuse_fence = record_task_stream_event();
      CHECK(predecessor_slot->reuse_fence != nullptr)
          << "Failed to record the Block-Spec predecessor-state reuse fence";
    }
    launch_model_invocation(invocation, slot);
    return;
  }

  if (invocation.kind ==
      BlockSpecPreparedInvocationKind::PUBLISH_CONTEXT_KV_STATE) {
    CHECK(resources.publish_source_bound)
        << "Block-Spec Context-KV publication requires fixed bindings";
    const BlockSpecPreparedPublishSource& source = resources.publish_source;
    check_block_spec_publish_source_contract(source,
                                             resources.device_step_state);
    block_spec_async::publish_block_spec_context_kv_state(
        source.accepted_tokens,
        source.accepted_context_hidden,
        source.base_positions,
        source.block_tables,
        impl_->block_size(),
        source.cache_slot_mapping_mode,
        resources.device_step_state,
        resources.context_kv_workspace);
  }

  launch_model_invocation(invocation, slot);
  if (is_final_invocation(invocation.kind)) {
    slot.task_output_event = record_task_stream_event();
    CHECK(slot.task_output_event != nullptr)
        << "Failed to record the Block-Spec Prepared task completion event";
    if (slot.output.has_value()) {
      slot.output->ready_event = slot.task_output_event;
    }
  }
}

std::optional<ForwardOutput> BlockSpecPreparedTaskBackendBase::consume(
    ExecutionSlot& slot) {
  CHECK(slot.task_output_event != nullptr);
  CHECK(slot.task_output_event->synchronize())
      << "Failed to wait for Block-Spec Prepared task output event";
  consume_ready(slot);
  if (slot.output.has_value()) {
    slot.output->retained_inputs.clear();
  }
  return std::move(slot.output);
}

void BlockSpecPreparedTaskBackendBase::bind_publish_source(
    int32_t slot_id,
    const BlockSpecPreparedPublishSource& source) {
  Impl::SlotResources& resources = impl_->slot(slot_id);
  check_block_spec_publish_source_contract(source, resources.device_step_state);
  resources.publish_source = source;
  resources.publish_source_bound = true;
}

block_spec_async::BlockSpecDeviceStepState&
BlockSpecPreparedTaskBackendBase::mutable_device_step_state(int32_t slot_id) {
  return impl_->slot(slot_id).device_step_state;
}

const block_spec_async::BlockSpecDeviceStepState&
BlockSpecPreparedTaskBackendBase::device_step_state(int32_t slot_id) const {
  return impl_->slot(slot_id).device_step_state;
}

void BlockSpecPreparedTaskBackendBase::patch_continuation_state(
    int32_t slot_id,
    const block_spec_async::BlockSpecDeviceStepState& predecessor_state,
    const torch::Tensor& predecessor_rows,
    block_spec_async::BlockSpecContinuationState& continuation) {
  Impl::SlotResources& resources = impl_->slot(slot_id);
  block_spec_async::patch_block_spec_continuation_rows(
      predecessor_state,
      predecessor_rows,
      continuation,
      resources.predecessor_workspace);
}

bool BlockSpecPreparedTaskBackendBase::stage_primary_input(
    int32_t slot_id,
    const ForwardInput& input,
    ForwardInput& staged_input) {
  Impl::SlotResources& resources = impl_->slot(slot_id);
  CHECK_GT(resources.input_partition_count, 0);
  CHECK(!resources.primary_input_staged)
      << "Block-Spec primary Task input may only be staged once";
  resources.input_arena->begin_partitioned_task(
      resources.input_partition_count);
  const bool staged = resources.input_arena->stage_next(input, staged_input);
  resources.primary_input_staged = staged;
  return staged;
}

bool BlockSpecPreparedTaskBackendBase::stage_invocation_input(
    int32_t slot_id,
    const ForwardInput& input,
    ForwardInput& staged_input) {
  Impl::SlotResources& resources = impl_->slot(slot_id);
  CHECK(resources.primary_input_staged)
      << "Block-Spec primary Task input must be staged before invocations";
  return resources.input_arena->stage_next(input, staged_input);
}

}  // namespace xllm
