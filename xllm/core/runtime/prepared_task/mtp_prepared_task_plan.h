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
#include <vector>

#include "runtime/prepared_task/prepared_task_types.h"

namespace xllm {

enum class MtpPreparedInvocationKind : int8_t {
  TARGET_PREFILL = 0,
  PREFILL_TO_DRAFT_PATCH,
  DRAFT_PREFILL,
  TASK_CONTINUATION_PATCH,
  DRAFT_DECODE,
  NEXT_DRAFT_PATCH,
  TARGET_VERIFY_PATCH,
  TARGET_VERIFY,
  REJECTION_SAMPLE,
  PUBLISH_STEP_STATE,
  EMPTY_COLLECTIVE,
};

// CPU-only descriptor produced during Prepare. invocation_index is stable
// within one Task, draft_step is set only for Draft/next-Draft entries, and
// input_partition identifies the fixed Slot Arena subrange consumed or
// patched by the invocation. Non-input invocations keep -1.
struct MtpPreparedInvocation {
  MtpPreparedInvocationKind kind = MtpPreparedInvocationKind::EMPTY_COLLECTIVE;
  int32_t invocation_index = 0;
  int32_t draft_step = -1;
  int32_t input_partition = -1;
};

struct MtpPreparedTaskPlan {
  PreparedTaskKind task_kind = PreparedTaskKind::EMPTY;
  int32_t input_partition_count = 1;
  std::vector<MtpPreparedInvocation> invocations;
};

MtpPreparedTaskPlan build_mtp_prepared_task_plan(
    PreparedTaskKind task_kind,
    int32_t num_speculative_tokens);

}  // namespace xllm
