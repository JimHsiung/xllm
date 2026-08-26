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
#include <vector>

#include "framework/kv_cache/kv_cache.h"
#include "framework/model/model_output.h"
#include "runtime/forward_params.h"

namespace xllm {

enum class PreparedInvocationMode : int8_t {
  EAGER = 0,
  GRAPH_REPLAY,
  GRAPH_CAPTURE,
};

// CPU-only decision produced during Prepare and consumed unchanged by Launch.
// native_handle owns a backend-specific executable for replay bindings.
struct PreparedSlotBinding {
  int32_t slot_id = -1;
  uint64_t graph_key = 0;
  PreparedInvocationMode mode = PreparedInvocationMode::EAGER;
  std::shared_ptr<void> native_handle;
  // Backend graph-task ready events were recorded during Prepare and precede
  // the invocation metadata-ready event consumed by the Launch stream.
  bool static_graph_tasks_prepared = false;
};

// Optional capability implemented only by executors with an explicit Prepared
// execution path. Legacy ExecutorImpl::run() remains unchanged.
class PreparedExecutor {
 public:
  virtual ~PreparedExecutor() = default;

  virtual void prepare_prepared_graph_input(
      int32_t slot_id,
      ForwardInput& input,
      std::vector<KVCache>& kv_caches) = 0;

  virtual PreparedSlotBinding bind_prepared(
      int32_t slot_id,
      const ForwardInput& input,
      std::vector<KVCache>& kv_caches) = 0;

  virtual ModelOutput launch_prepared(const PreparedSlotBinding& binding,
                                      const ForwardInput& input,
                                      std::vector<KVCache>& kv_caches) = 0;
};

}  // namespace xllm
