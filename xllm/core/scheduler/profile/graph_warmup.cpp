/* Copyright 2025-2026 The xLLM Authors.

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

#include "scheduler/profile/graph_warmup.h"

#include <absl/strings/str_cat.h>
#include <glog/logging.h>
#include <torch/torch.h>

#include <atomic>
#include <iomanip>
#include <sstream>

#include "framework/request/sequence.h"

namespace xllm {
namespace {

constexpr int32_t kGraphWarmupBarWidth = 20;

}  // namespace

GraphWarmupPlan graph_warmup_plan(InstanceRole role) {
  if (role == InstanceRole::PREFILL) {
    return GraphWarmupPlan::PREFILL_ONLY;
  }
  if (role == InstanceRole::DECODE) {
    return GraphWarmupPlan::DECODE_ONLY;
  }

  return GraphWarmupPlan::UNIFIED;
}

int32_t graph_warmup_invocations_per_bucket(bool enable_prepared_task_pipeline,
                                            bool enable_schedule_overlap) {
  return enable_prepared_task_pipeline && enable_schedule_overlap ? 2 : 1;
}

std::vector<int32_t> graph_warmup_accepted_length_schedule(
    int32_t num_speculative_tokens,
    bool enable_hybrid_mtp_variants,
    int32_t invocations_per_variant) {
  CHECK_GE(num_speculative_tokens, 0);
  CHECK_GT(invocations_per_variant, 0);

  const int32_t variant_count =
      enable_hybrid_mtp_variants && num_speculative_tokens > 0
          ? num_speculative_tokens + 1
          : 1;
  std::vector<int32_t> schedule;
  schedule.reserve(static_cast<size_t>(variant_count) *
                   static_cast<size_t>(invocations_per_variant));
  for (int32_t accepted_prefix_length = 1;
       accepted_prefix_length <= variant_count;
       ++accepted_prefix_length) {
    for (int32_t invocation = 0; invocation < invocations_per_variant;
         ++invocation) {
      schedule.emplace_back(accepted_prefix_length);
    }
  }
  return schedule;
}

std::string graph_warmup_progress(int32_t completed,
                                  int32_t total,
                                  int32_t token_bucket,
                                  double latency_ms) {
  CHECK_GT(total, 0);
  CHECK_GE(completed, 0);
  CHECK_LE(completed, total);
  CHECK_GT(token_bucket, 0);
  CHECK_GE(latency_ms, 0.0);

  const int32_t filled = static_cast<int32_t>(
      (static_cast<int64_t>(completed) * kGraphWarmupBarWidth + total / 2) /
      total);

  std::string bar;
  bar.reserve(kGraphWarmupBarWidth);
  bar.append(static_cast<size_t>(filled), '#');
  bar.append(static_cast<size_t>(kGraphWarmupBarWidth - filled), '-');

  const double percent =
      static_cast<double>(completed) * 100.0 / static_cast<double>(total);

  std::ostringstream oss;
  oss << "Graph warmup progress: [" << bar << "] " << completed << "/" << total
      << " " << std::fixed << std::setprecision(1) << percent
      << "%, token_bucket=" << token_bucket
      << ", latency=" << std::setprecision(2) << latency_ms << " ms";
  return oss.str();
}

std::string next_warmup_request_id() {
  static std::atomic<int64_t> counter{0};
  const int64_t id = counter.fetch_add(1, std::memory_order_relaxed);
  return absl::StrCat("warmup_", id);
}

void prepare_warmup_decode_sequence(Sequence* sequence,
                                    int64_t embedding_width,
                                    int32_t num_speculative_tokens,
                                    int32_t accepted_prefix_length) {
  CHECK(sequence != nullptr);
  CHECK_GT(accepted_prefix_length, 0);
  if (num_speculative_tokens <= 0) {
    CHECK_EQ(accepted_prefix_length, 1)
        << "non-speculative warmup only supports accepted length 1";
    return;
  }

  CHECK_GT(embedding_width, 0);
  CHECK_LE(accepted_prefix_length, num_speculative_tokens + 1)
      << "warmup accepted length exceeds target verify width";
  sequence->set_graph_warmup_speculative_accepted_length(
      accepted_prefix_length);
  // Placeholder bootstrap hidden states; the worker converts dtype/device and
  // only the [1, embedding_width] shape matters for the batch input builder.
  sequence->update_mtp_bootstrap_embedding(
      torch::zeros({1, embedding_width}, /*options=*/torch::kFloat));
}

}  // namespace xllm
