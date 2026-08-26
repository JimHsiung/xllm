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

#include "runtime/prepared_task/llm_prepared_task_adapter.h"

#include <glog/logging.h>

#include <utility>

namespace xllm {

LlmPreparedTaskAdapter::LlmPreparedTaskAdapter(LLMWorkerImpl* worker,
                                               uint64_t arena_capacity_bytes,
                                               int32_t slot_count)
    : worker_(CHECK_NOTNULL(worker)) {
  CHECK(slot_count == 1 || slot_count == 2);
  input_arenas_.reserve(slot_count);
  for (int32_t slot_id = 0; slot_id < slot_count; ++slot_id) {
    input_arenas_.emplace_back(std::make_unique<PreparedInputArena>(
        worker_->device(), arena_capacity_bytes));
  }
}

void LlmPreparedTaskAdapter::initialize_state_thread() {
  worker_->initialize_prepared_task_thread();
}

void LlmPreparedTaskAdapter::initialize_launch_thread() {
  worker_->initialize_prepared_task_thread();
}

PreparedTaskKind LlmPreparedTaskAdapter::classify(
    const ForwardInput& input) const {
  const BatchForwardType& forward_type =
      input.input_params.meta.batch_forward_type;
  if (forward_type.is_empty()) {
    return PreparedTaskKind::EMPTY;
  }
  if (forward_type.is_decode()) {
    return PreparedTaskKind::DECODE;
  }
  return PreparedTaskKind::PREFILL_LIKE;
}

void LlmPreparedTaskAdapter::prepare(const ForwardInput& input,
                                     ExecutionSlot& slot) {
  CHECK(input.json_object_states.empty() &&
        input.json_object_state_snapshots.empty())
      << "JSON grammar is not supported by PreparedTaskPipeline phases 1-2";
  c10::StreamGuard stream_guard =
      worker_->prepared_prepare_stream().set_stream_guard();
  ForwardInput staged_input;
  CHECK_GE(slot.slot_id, 0);
  CHECK_LT(static_cast<size_t>(slot.slot_id), input_arenas_.size());
  CHECK(input_arenas_[slot.slot_id]->stage(input, staged_input))
      << "ForwardInput is not supported by the fixed Prepared input arena";
  worker_->prepare_work_before_execute(staged_input, slot.prepared_input);
  if (worker_->prepared_graph_enabled()) {
    worker_->prepare_prepared_graph_input(slot.slot_id, slot.prepared_input);
  }
  CHECK(input_arenas_[slot.slot_id]->stage_generated_metadata(
      slot.prepared_input))
      << "Worker-generated metadata is not supported by the fixed Prepared "
         "input arena";
  if (worker_->prepared_graph_enabled()) {
    slot.model_binding =
        worker_->bind_prepared_task(slot.slot_id, slot.prepared_input);
  } else {
    slot.model_binding.reset();
  }
  slot.prepared_input.metadata_ready_event =
      worker_->prepared_prepare_stream().record_event_or_sync();
  CHECK(slot.prepared_input.metadata_ready_event != nullptr)
      << "Failed to record Prepared Worker metadata-ready event";
}

void LlmPreparedTaskAdapter::launch(ExecutionSlot& slot) {
  if (worker_->enable_schedule_overlap()) {
    worker_->patch_prepared_task_input_for_schedule_overlap(
        slot.prepared_input);
  }
  slot.output =
      worker_->execute_prepared_task(slot.prepared_input, slot.model_binding);
  // Record an unconditional task event: non-driver ranks can legitimately
  // return no ForwardOutput but still need a safe Slot reuse fence.
  slot.task_output_event = worker_->record_prepared_task_event();
  if (slot.output.has_value()) {
    slot.output->ready_event = slot.task_output_event;
  }
  if (worker_->enable_schedule_overlap()) {
    worker_->publish_prepared_task_output(slot.prepared_input, slot.output);
  }
}

std::optional<ForwardOutput> LlmPreparedTaskAdapter::consume(
    ExecutionSlot& slot) {
  CHECK(slot.task_output_event != nullptr);
  CHECK(slot.task_output_event->synchronize())
      << "Failed to wait for Prepared task output event";
  if (slot.output.has_value()) {
    slot.output->retained_inputs.clear();
  }
  return std::move(slot.output);
}

}  // namespace xllm
