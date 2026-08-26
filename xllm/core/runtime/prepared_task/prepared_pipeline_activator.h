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

#include <glog/logging.h>

#include <cstdint>
#include <utility>

namespace xllm {

enum class PreparedPipelineActivationStage : int8_t {
  DISABLED = 0,
  IMMEDIATE,
  AFTER_CACHE_ALLOCATION,
};

// Startup-only exactly-once guard for Prepared Pipeline construction. It
// keeps cache-dependent speculative initialization separate from immediate
// LLM initialization without adding a runtime execution-path branch.
class PreparedPipelineActivator final {
 public:
  void configure(PreparedPipelineActivationStage stage) {
    CHECK(stage_ == PreparedPipelineActivationStage::DISABLED);
    CHECK(stage != PreparedPipelineActivationStage::DISABLED);
    stage_ = stage;
  }

  template <typename Factory>
  void initialize_at_construction(Factory&& factory) {
    CHECK(stage_ == PreparedPipelineActivationStage::IMMEDIATE);
    initialize(std::forward<Factory>(factory));
  }

  template <typename Factory>
  void finish_cache_allocation(bool success, Factory&& factory) {
    if (!success ||
        stage_ != PreparedPipelineActivationStage::AFTER_CACHE_ALLOCATION ||
        initialized_) {
      return;
    }
    initialize(std::forward<Factory>(factory));
  }

  bool initialized() const { return initialized_; }

 private:
  template <typename Factory>
  void initialize(Factory&& factory) {
    CHECK(!initialized_) << "PreparedTaskPipeline may only be initialized once";
    std::forward<Factory>(factory)();
    initialized_ = true;
  }

  PreparedPipelineActivationStage stage_ =
      PreparedPipelineActivationStage::DISABLED;
  bool initialized_ = false;
};

}  // namespace xllm
