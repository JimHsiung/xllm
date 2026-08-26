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

enum class BlockSpecAlgorithm : int8_t {
  DFLASH = 0,
  DSPARK,
};

enum class BlockSpecPreparedInvocationKind : int8_t {
  TARGET_PREFILL = 0,
  TASK_CONTINUATION_PATCH,
  BLOCK_DRAFT,
  DSPARK_MARKOV_SAMPLE,
  DSPARK_TOKEN_BROADCAST,
  TARGET_VALIDATE_PATCH,
  TARGET_VALIDATE,
  REJECTION_SAMPLE,
  PUBLISH_CONTEXT_KV_STATE,
  WRITE_CONTEXT_KV,
  EMPTY_COLLECTIVE,
};

// CPU-only descriptor for a fixed Block-Spec launch. block_step is set only
// for DSpark's Markov sample/broadcast pairs. input_partition identifies the
// stable Slot Arena partition used by model inputs; launch-only patches keep
// -1 unless they mutate one of those partitions.
struct BlockSpecPreparedInvocation {
  BlockSpecPreparedInvocationKind kind =
      BlockSpecPreparedInvocationKind::EMPTY_COLLECTIVE;
  int32_t invocation_index = 0;
  int32_t block_step = -1;
  int32_t input_partition = -1;
};

struct BlockSpecPreparedTaskPlan {
  BlockSpecAlgorithm algorithm = BlockSpecAlgorithm::DFLASH;
  PreparedTaskKind task_kind = PreparedTaskKind::EMPTY;
  int32_t speculative_width = 0;
  int32_t input_partition_count = 1;
  std::vector<BlockSpecPreparedInvocation> invocations;
};

BlockSpecPreparedTaskPlan build_block_spec_prepared_task_plan(
    BlockSpecAlgorithm algorithm,
    PreparedTaskKind task_kind,
    int32_t speculative_width);

}  // namespace xllm
