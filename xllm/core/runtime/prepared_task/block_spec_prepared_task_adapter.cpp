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

#include "runtime/prepared_task/block_spec_prepared_task_adapter.h"

#include <glog/logging.h>

#include <cstddef>
#include <utility>

namespace xllm {
namespace {

size_t checked_slot_count(int32_t slot_count) {
  CHECK(slot_count == 1 || slot_count == 2);
  return static_cast<size_t>(slot_count);
}

}  // namespace

BlockSpecPreparedTaskAdapter::BlockSpecPreparedTaskAdapter(
    std::unique_ptr<BlockSpecPreparedTaskBackend> backend,
    BlockSpecAlgorithm algorithm,
    int32_t speculative_width,
    int32_t slot_count)
    : backend_(std::move(backend)),
      prefill_plan_(
          build_block_spec_prepared_task_plan(algorithm,
                                              PreparedTaskKind::PREFILL_LIKE,
                                              speculative_width)),
      decode_plan_(build_block_spec_prepared_task_plan(algorithm,
                                                       PreparedTaskKind::DECODE,
                                                       speculative_width)),
      empty_plan_(build_block_spec_prepared_task_plan(algorithm,
                                                      PreparedTaskKind::EMPTY,
                                                      speculative_width)),
      continuation_launched_(checked_slot_count(slot_count), 0U) {
  CHECK(backend_ != nullptr);
}

void BlockSpecPreparedTaskAdapter::initialize_state_thread() {
  backend_->initialize_state_thread();
}

void BlockSpecPreparedTaskAdapter::initialize_launch_thread() {
  backend_->initialize_launch_thread();
}

PreparedTaskKind BlockSpecPreparedTaskAdapter::classify(
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

bool BlockSpecPreparedTaskAdapter::consumes_predecessor_state(
    const ForwardInput& input) const {
  return classify(input) == PreparedTaskKind::DECODE &&
         input.input_params.embedding.predecessor_rows.defined();
}

void BlockSpecPreparedTaskAdapter::prepare_slot_for_reuse(ExecutionSlot& slot) {
  backend_->prepare_slot_for_reuse(slot);
}

void BlockSpecPreparedTaskAdapter::prepare(const ForwardInput& input,
                                           ExecutionSlot& slot) {
  const BlockSpecPreparedTaskPlan& plan = plan_for_kind(slot.task_kind);
  continuation_launched_[static_cast<size_t>(slot.slot_id)] = 0U;
  backend_->prepare(input, slot, plan);
}

void BlockSpecPreparedTaskAdapter::launch_predecessor_continuation(
    ExecutionSlot& slot,
    ExecutionSlot& predecessor_slot) {
  const BlockSpecPreparedTaskPlan& plan = plan_for_kind(slot.task_kind);
  CHECK(!plan.invocations.empty());
  CHECK(plan.invocations.front().kind ==
        BlockSpecPreparedInvocationKind::TASK_CONTINUATION_PATCH);
  const size_t slot_index = static_cast<size_t>(slot.slot_id);
  CHECK_EQ(continuation_launched_[slot_index], 0U);
  backend_->launch(plan.invocations.front(), slot, &predecessor_slot);
  continuation_launched_[slot_index] = 1U;
}

void BlockSpecPreparedTaskAdapter::launch(ExecutionSlot& slot) {
  const BlockSpecPreparedTaskPlan& plan = plan_for_kind(slot.task_kind);
  const bool continuation_launched =
      continuation_launched_[static_cast<size_t>(slot.slot_id)] != 0U;
  for (const BlockSpecPreparedInvocation& invocation : plan.invocations) {
    if (continuation_launched &&
        invocation.kind ==
            BlockSpecPreparedInvocationKind::TASK_CONTINUATION_PATCH) {
      continue;
    }
    backend_->launch(invocation, slot, /*predecessor_slot=*/nullptr);
  }
}

std::optional<ForwardOutput> BlockSpecPreparedTaskAdapter::consume(
    ExecutionSlot& slot) {
  return backend_->consume(slot);
}

const BlockSpecPreparedTaskPlan& BlockSpecPreparedTaskAdapter::plan_for_kind(
    PreparedTaskKind task_kind) const {
  switch (task_kind) {
    case PreparedTaskKind::PREFILL_LIKE:
      return prefill_plan_;
    case PreparedTaskKind::DECODE:
      return decode_plan_;
    case PreparedTaskKind::EMPTY:
      return empty_plan_;
  }
  LOG(FATAL) << "Unsupported Prepared Block-Spec Task kind";
  return empty_plan_;
}

}  // namespace xllm
