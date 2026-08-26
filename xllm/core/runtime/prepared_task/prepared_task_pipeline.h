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

#include <folly/futures/Future.h>

#include <cstdint>
#include <memory>
#include <optional>

#include "runtime/prepared_task/prepared_task_adapter.h"

namespace xllm {

class PreparedTaskPipeline final {
 public:
  PreparedTaskPipeline(std::unique_ptr<PreparedTaskAdapter> adapter,
                       int32_t slot_count);
  ~PreparedTaskPipeline();

  PreparedTaskPipeline(const PreparedTaskPipeline&) = delete;
  PreparedTaskPipeline& operator=(const PreparedTaskPipeline&) = delete;

  folly::SemiFuture<std::optional<ForwardOutput>> submit(
      const ForwardInput& input);
  folly::SemiFuture<std::optional<ForwardOutput>> get_last_step_result();

  void quiesce();
  void resume();
  void shutdown();

  PreparedPipelineLifecycle lifecycle() const;
  int32_t slot_count() const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace xllm
