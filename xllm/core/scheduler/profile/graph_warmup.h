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

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "common/types.h"

namespace xllm {

class Sequence;

enum class GraphWarmupPlan : int8_t {
  UNIFIED = 0,
  PREFILL_ONLY = 1,
  DECODE_ONLY = 2,
};

GraphWarmupPlan graph_warmup_plan(InstanceRole role);

// Prepared Graph caches are keyed by an explicit fixed-address Slot. Schedule
// overlap owns two Slots, so every decode bucket must execute once per Slot
// during startup. Legacy Graph and single-Slot Prepared Graph keep one warmup
// invocation per bucket.
int32_t graph_warmup_invocations_per_bucket(bool enable_prepared_task_pipeline,
                                            bool enable_schedule_overlap);

// Returns the accepted-prefix length for every invocation of one decode
// bucket. Hybrid MTP target-verify Graph keys specialize on the real accepted
// length, so every value in [1, num_speculative_tokens + 1] must be captured.
// Each value is repeated consecutively for every fixed-address Prepared Slot
// so alternating Slot assignment covers the full variant set on both Slots.
// Non-hybrid and non-speculative Graphs keep the single length-1 variant.
std::vector<int32_t> graph_warmup_accepted_length_schedule(
    int32_t num_speculative_tokens,
    bool enable_hybrid_mtp_variants,
    int32_t invocations_per_variant);

std::string graph_warmup_progress(int32_t completed,
                                  int32_t total,
                                  int32_t token_bucket,
                                  double latency_ms);

// Returns a process-unique request id for synthetic profiling/warmup requests.
// Distinct ids keep these requests separable from each other (and from real
// requests) in the embedding cache, so stale decode state from a recycled
// embedding block cannot be mistaken for a warmup request's own state.
std::string next_warmup_request_id();

// Prepares a synthetic decode sequence for graph warmup. When speculative
// decoding is enabled (MTP), the worker's decode path requires a valid decode
// state written through the MTP bootstrap channel before it validates the
// per-token decode state. This injects a placeholder bootstrap embedding of
// shape [1, embedding_width] so the bootstrap path runs during graph capture;
// the embedding values are irrelevant because warmup only captures the graph.
// Does nothing when speculative decoding is disabled.
void prepare_warmup_decode_sequence(Sequence* sequence,
                                    int64_t embedding_width,
                                    int32_t num_speculative_tokens,
                                    int32_t accepted_prefix_length = 1);

}  // namespace xllm
