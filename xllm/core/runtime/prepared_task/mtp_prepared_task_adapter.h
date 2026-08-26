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

#include "runtime/prepared_task/mtp_prepared_task_backend.h"
#include "runtime/prepared_task/prepared_task_adapter.h"

namespace xllm {

class MtpPreparedTaskAdapter final : public PreparedTaskAdapter {
 public:
  MtpPreparedTaskAdapter(std::unique_ptr<MtpPreparedTaskBackend> backend,
                         int32_t num_speculative_tokens,
                         int32_t slot_count);

  void initialize_state_thread() override;
  void initialize_launch_thread() override;
  PreparedTaskKind classify(const ForwardInput& input) const override;
  bool consumes_predecessor_state(const ForwardInput& input) const override;
  void prepare_slot_for_reuse(ExecutionSlot& slot) override;
  void prepare(const ForwardInput& input, ExecutionSlot& slot) override;
  void launch_predecessor_continuation(
      ExecutionSlot& slot,
      ExecutionSlot& predecessor_slot) override;
  void launch(ExecutionSlot& slot) override;
  std::optional<ForwardOutput> consume(ExecutionSlot& slot) override;

 private:
  const MtpPreparedTaskPlan& plan_for_kind(PreparedTaskKind task_kind) const;

  std::unique_ptr<MtpPreparedTaskBackend> backend_;
  MtpPreparedTaskPlan prefill_plan_;
  MtpPreparedTaskPlan decode_plan_;
  MtpPreparedTaskPlan empty_plan_;
  // Do not use vector<bool>: different Slot bits may share one machine word
  // while Prepare and Launch operate concurrently.
  std::vector<uint8_t> continuation_launched_;
};

}  // namespace xllm
