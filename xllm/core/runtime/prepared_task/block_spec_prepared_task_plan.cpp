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

#include "runtime/prepared_task/block_spec_prepared_task_plan.h"

#include <glog/logging.h>

#include <cstddef>
#include <limits>

namespace xllm {
namespace {

void append_invocation(BlockSpecPreparedTaskPlan& plan,
                       BlockSpecPreparedInvocationKind kind,
                       int32_t block_step,
                       int32_t input_partition) {
  plan.invocations.emplace_back(
      BlockSpecPreparedInvocation{kind,
                                  static_cast<int32_t>(plan.invocations.size()),
                                  block_step,
                                  input_partition});
}

}  // namespace

BlockSpecPreparedTaskPlan build_block_spec_prepared_task_plan(
    BlockSpecAlgorithm algorithm,
    PreparedTaskKind task_kind,
    int32_t speculative_width) {
  CHECK_GT(speculative_width, 0);
  CHECK_LE(speculative_width, (std::numeric_limits<int32_t>::max() - 7) / 2);

  BlockSpecPreparedTaskPlan plan;
  plan.algorithm = algorithm;
  plan.task_kind = task_kind;
  plan.speculative_width = speculative_width;
  if (task_kind == PreparedTaskKind::EMPTY) {
    append_invocation(plan,
                      BlockSpecPreparedInvocationKind::EMPTY_COLLECTIVE,
                      /*block_step=*/-1,
                      /*input_partition=*/0);
    return plan;
  }

  if (task_kind == PreparedTaskKind::PREFILL_LIKE) {
    // Prefill context length is not bounded by the speculative Decode width.
    // The future Worker backend writes prompt Context-KV directly from the
    // Target output instead of forcing it through Decode DeviceStepState.
    plan.invocations.reserve(2);
    append_invocation(plan,
                      BlockSpecPreparedInvocationKind::TARGET_PREFILL,
                      /*block_step=*/-1,
                      /*input_partition=*/0);
    append_invocation(plan,
                      BlockSpecPreparedInvocationKind::WRITE_CONTEXT_KV,
                      /*block_step=*/-1,
                      /*input_partition=*/-1);
    return plan;
  }

  CHECK(task_kind == PreparedTaskKind::DECODE);
  // Partition 0 retains the Engine input, partition 1 is the fixed block-draft
  // input, and partition 2 is the fixed Target validation input.
  plan.input_partition_count = 3;
  const int32_t algorithm_invocation_count =
      algorithm == BlockSpecAlgorithm::DSPARK ? speculative_width * 2 : 0;
  plan.invocations.reserve(static_cast<size_t>(algorithm_invocation_count + 7));
  append_invocation(plan,
                    BlockSpecPreparedInvocationKind::TASK_CONTINUATION_PATCH,
                    /*block_step=*/-1,
                    /*input_partition=*/1);
  append_invocation(plan,
                    BlockSpecPreparedInvocationKind::BLOCK_DRAFT,
                    /*block_step=*/-1,
                    /*input_partition=*/1);
  if (algorithm == BlockSpecAlgorithm::DSPARK) {
    for (int32_t block_step = 0; block_step < speculative_width; ++block_step) {
      append_invocation(plan,
                        BlockSpecPreparedInvocationKind::DSPARK_MARKOV_SAMPLE,
                        block_step,
                        /*input_partition=*/-1);
      append_invocation(plan,
                        BlockSpecPreparedInvocationKind::DSPARK_TOKEN_BROADCAST,
                        block_step,
                        /*input_partition=*/-1);
    }
  }
  append_invocation(plan,
                    BlockSpecPreparedInvocationKind::TARGET_VALIDATE_PATCH,
                    /*block_step=*/-1,
                    /*input_partition=*/2);
  append_invocation(plan,
                    BlockSpecPreparedInvocationKind::TARGET_VALIDATE,
                    /*block_step=*/-1,
                    /*input_partition=*/2);
  append_invocation(plan,
                    BlockSpecPreparedInvocationKind::REJECTION_SAMPLE,
                    /*block_step=*/-1,
                    /*input_partition=*/-1);
  append_invocation(plan,
                    BlockSpecPreparedInvocationKind::PUBLISH_CONTEXT_KV_STATE,
                    /*block_step=*/-1,
                    /*input_partition=*/-1);
  append_invocation(plan,
                    BlockSpecPreparedInvocationKind::WRITE_CONTEXT_KV,
                    /*block_step=*/-1,
                    /*input_partition=*/-1);
  CHECK_EQ(plan.invocations.size(),
           static_cast<size_t>(algorithm_invocation_count + 7));
  return plan;
}

}  // namespace xllm
