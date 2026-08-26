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

#include "runtime/prepared_task/mtp_prepared_task_plan.h"

#include <glog/logging.h>

#include <cstddef>
#include <limits>

namespace xllm {

MtpPreparedTaskPlan build_mtp_prepared_task_plan(
    PreparedTaskKind task_kind,
    int32_t num_speculative_tokens) {
  CHECK_GT(num_speculative_tokens, 0);
  CHECK_LE(num_speculative_tokens,
           (std::numeric_limits<int32_t>::max() - 4) / 2);

  MtpPreparedTaskPlan plan;
  plan.task_kind = task_kind;
  if (task_kind == PreparedTaskKind::EMPTY) {
    plan.input_partition_count = 1;
    plan.invocations.emplace_back(
        MtpPreparedInvocation{MtpPreparedInvocationKind::EMPTY_COLLECTIVE,
                              /*invocation_index=*/0,
                              /*draft_step=*/-1,
                              /*input_partition=*/0});
    return plan;
  }

  if (task_kind == PreparedTaskKind::PREFILL_LIKE) {
    // Partition 0 is the staged Target input; partition 1 is the fixed Draft
    // prefill destination patched from Target output.
    plan.input_partition_count = 2;
    plan.invocations.reserve(/*target + patch + draft + publish=*/4);
    plan.invocations.emplace_back(
        MtpPreparedInvocation{MtpPreparedInvocationKind::TARGET_PREFILL,
                              /*invocation_index=*/0,
                              /*draft_step=*/-1,
                              /*input_partition=*/0});
    plan.invocations.emplace_back(
        MtpPreparedInvocation{MtpPreparedInvocationKind::PREFILL_TO_DRAFT_PATCH,
                              /*invocation_index=*/1,
                              /*draft_step=*/-1,
                              /*input_partition=*/1});
    plan.invocations.emplace_back(
        MtpPreparedInvocation{MtpPreparedInvocationKind::DRAFT_PREFILL,
                              /*invocation_index=*/2,
                              /*draft_step=*/0,
                              /*input_partition=*/1});
    plan.invocations.emplace_back(
        MtpPreparedInvocation{MtpPreparedInvocationKind::PUBLISH_STEP_STATE,
                              /*invocation_index=*/3,
                              /*draft_step=*/-1,
                              /*input_partition=*/-1});
    return plan;
  }

  CHECK(task_kind == PreparedTaskKind::DECODE);
  // Partition 0 retains the Engine input/block-table source. Draft step i
  // uses partition i + 1 and Target Verify uses the final partition.
  plan.input_partition_count = num_speculative_tokens + 2;
  const int32_t invocation_count = num_speculative_tokens * 2 + 4;
  plan.invocations.reserve(static_cast<size_t>(invocation_count));
  plan.invocations.emplace_back(
      MtpPreparedInvocation{MtpPreparedInvocationKind::TASK_CONTINUATION_PATCH,
                            /*invocation_index=*/0,
                            /*draft_step=*/-1,
                            /*input_partition=*/1});
  for (int32_t draft_step = 0; draft_step < num_speculative_tokens;
       ++draft_step) {
    const int32_t draft_invocation_index =
        static_cast<int32_t>(plan.invocations.size());
    plan.invocations.emplace_back(
        MtpPreparedInvocation{MtpPreparedInvocationKind::DRAFT_DECODE,
                              draft_invocation_index,
                              draft_step,
                              /*input_partition=*/draft_step + 1});
    if (draft_step + 1 == num_speculative_tokens) {
      continue;
    }
    const int32_t patch_invocation_index =
        static_cast<int32_t>(plan.invocations.size());
    plan.invocations.emplace_back(
        MtpPreparedInvocation{MtpPreparedInvocationKind::NEXT_DRAFT_PATCH,
                              patch_invocation_index,
                              draft_step + 1,
                              /*input_partition=*/draft_step + 2});
  }
  plan.invocations.emplace_back(
      MtpPreparedInvocation{MtpPreparedInvocationKind::TARGET_VERIFY_PATCH,
                            static_cast<int32_t>(plan.invocations.size()),
                            /*draft_step=*/-1,
                            /*input_partition=*/num_speculative_tokens + 1});
  plan.invocations.emplace_back(
      MtpPreparedInvocation{MtpPreparedInvocationKind::TARGET_VERIFY,
                            static_cast<int32_t>(plan.invocations.size()),
                            /*draft_step=*/-1,
                            /*input_partition=*/num_speculative_tokens + 1});
  plan.invocations.emplace_back(
      MtpPreparedInvocation{MtpPreparedInvocationKind::REJECTION_SAMPLE,
                            static_cast<int32_t>(plan.invocations.size()),
                            /*draft_step=*/-1,
                            /*input_partition=*/-1});
  plan.invocations.emplace_back(
      MtpPreparedInvocation{MtpPreparedInvocationKind::PUBLISH_STEP_STATE,
                            static_cast<int32_t>(plan.invocations.size()),
                            /*draft_step=*/-1,
                            /*input_partition=*/-1});
  CHECK_EQ(plan.invocations.size(), static_cast<size_t>(invocation_count));
  return plan;
}

}  // namespace xllm
