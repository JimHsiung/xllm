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

#include <cstdint>
#include <optional>

#include "runtime/forward_params.h"
#include "runtime/prepared_executor.h"

namespace xllm {

enum class PreparedTaskKind : int8_t {
  PREFILL_LIKE = 0,
  DECODE,
  EMPTY,
};

enum class ExecutionSlotState : int8_t {
  FREE = 0,
  PREPARING,
  READY,
  RUNNING,
  COMPLETED,
  CONSUMING,
};

enum class PreparedPipelineLifecycle : int8_t {
  CREATED = 0,
  RUNNING,
  QUIESCENT,
  STOPPED,
};

struct ExecutionSlot {
  int32_t slot_id = 0;
  uint64_t task_seq_no = 0;
  PreparedTaskKind task_kind = PreparedTaskKind::EMPTY;
  ExecutionSlotState state = ExecutionSlotState::FREE;
  std::optional<int32_t> predecessor_slot_id;
  bool successor_consumes_state = false;
  bool successor_read_enqueued = false;
  ForwardInput prepared_input;
  std::optional<PreparedSlotBinding> model_binding;
  std::optional<ForwardOutput> output;
  StreamEventPtr task_output_event;
  StreamEventPtr reuse_fence;
};

}  // namespace xllm
