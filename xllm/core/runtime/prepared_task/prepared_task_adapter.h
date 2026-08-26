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

#include <optional>

#include "runtime/prepared_task/prepared_task_types.h"

namespace xllm {

class PreparedTaskAdapter {
 public:
  virtual ~PreparedTaskAdapter() = default;

  virtual void initialize_state_thread() {}
  virtual void initialize_launch_thread() {}

  virtual PreparedTaskKind classify(const ForwardInput& input) const = 0;
  virtual bool consumes_predecessor_state(const ForwardInput& /*input*/) const {
    return false;
  }
  virtual void prepare_slot_for_reuse(ExecutionSlot& /*slot*/) {}
  virtual void prepare(const ForwardInput& input, ExecutionSlot& slot) = 0;
  virtual void launch_predecessor_continuation(
      ExecutionSlot& /*slot*/,
      ExecutionSlot& /*predecessor_slot*/) {}
  virtual void launch(ExecutionSlot& slot) = 0;
  virtual std::optional<ForwardOutput> consume(ExecutionSlot& slot) = 0;
};

}  // namespace xllm
