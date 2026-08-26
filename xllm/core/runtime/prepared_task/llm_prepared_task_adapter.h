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
#include <memory>
#include <optional>
#include <vector>

#include "runtime/llm_worker_impl.h"
#include "runtime/prepared_task/prepared_input_arena.h"
#include "runtime/prepared_task/prepared_task_adapter.h"

namespace xllm {

class LlmPreparedTaskAdapter final : public PreparedTaskAdapter {
 public:
  LlmPreparedTaskAdapter(LLMWorkerImpl* worker,
                         uint64_t arena_capacity_bytes,
                         int32_t slot_count);

  void initialize_state_thread() override;
  void initialize_launch_thread() override;
  PreparedTaskKind classify(const ForwardInput& input) const override;
  void prepare(const ForwardInput& input, ExecutionSlot& slot) override;
  void launch(ExecutionSlot& slot) override;
  std::optional<ForwardOutput> consume(ExecutionSlot& slot) override;

 private:
  LLMWorkerImpl* worker_ = nullptr;
  std::vector<std::unique_ptr<PreparedInputArena>> input_arenas_;
};

}  // namespace xllm
