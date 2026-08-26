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

#include <folly/futures/Future.h>
#include <torch/torch.h>

#include "executor.h"
#include "forward_params.h"
#include "framework/model/causal_lm.h"
#include "framework/model/model_args.h"
#include "framework/model/model_input_params.h"
#include "framework/quant_args.h"
#include "framework/state_dict/state_dict.h"
#include "options.h"
#include "runtime/worker_impl.h"

namespace xllm {

// Optional fixed destinations for Prepared model outputs whose shapes are
// known at startup. Undefined fields preserve the existing allocation path.
struct PreparedModelOutputWorkspace {
  torch::Tensor next_tokens;
  torch::Tensor selected_embeddings;
};

void check_prepared_model_output_binding(
    const SampleOutput& sample_output,
    const PreparedModelOutputWorkspace& output_workspace);

torch::Tensor gather_prepared_selected_embeddings(
    const torch::Tensor& embeddings,
    const torch::Tensor& selected_token_idxes,
    const torch::Tensor& destination);

class LLMWorkerImpl : public WorkerImpl {
 public:
  enum class ForwardSyncPolicy : int8_t {
    LEGACY = 0,
    NO_SYNC,
  };

  LLMWorkerImpl(const ParallelArgs& parallel_args,
                const torch::Device& device,
                const runtime::Options& options);

  ~LLMWorkerImpl() override = default;

  // initialize model, cache manager. blocking call
  bool init_model(ModelContext& context) override;

  std::optional<ForwardOutput> step(const ForwardInput& input) override;

  std::optional<ForwardOutput> step_no_sync(const ForwardInput& input);
  virtual std::optional<ForwardOutput> execute_no_sync_on_stream(
      const ForwardInput& input,
      Stream& compute_stream,
      bool record_ready_event = true);

  std::optional<ForwardOutput> execute_prepared_task(
      const ForwardInput& input,
      const std::optional<PreparedSlotBinding>& binding);
  std::optional<ForwardOutput> execute_prepared_on_stream(
      const ForwardInput& input,
      Stream& compute_stream,
      const std::optional<PreparedSlotBinding>& binding,
      const PreparedModelOutputWorkspace* output_workspace = nullptr);
  PreparedSlotBinding bind_prepared_task(int32_t slot_id,
                                         const ForwardInput& input);
  void prepare_prepared_graph_input(int32_t slot_id, ForwardInput& input);
  bool prepared_graph_enabled() const;
  void patch_prepared_task_input_for_schedule_overlap(ForwardInput& input);
  void publish_prepared_task_output(const ForwardInput& input,
                                    const std::optional<ForwardOutput>& output);

  folly::SemiFuture<std::optional<ForwardOutput>> step_async_no_sync(
      const ForwardInput& input);

  std::optional<ForwardOutput> step_internal(
      const ForwardInput& input,
      ForwardSyncPolicy sync_policy = ForwardSyncPolicy::LEGACY,
      bool record_ready_event = true,
      const PreparedSlotBinding* prepared_binding = nullptr,
      const PreparedModelOutputWorkspace* output_workspace = nullptr,
      bool retain_input_for_async_output = true);

 protected:
  std::optional<ForwardOutput> step_for_schedule_overlap(
      const ForwardInput& input) override;
  ForwardInput update_input_by_last_step_output_for_schedule_overlap(
      ForwardInput& input) override;

 public:
#if defined(USE_NPU)
  bool prepare_static_mtp_graph_tasks(const SpecVerifyGraphTaskSignal& signal,
                                      const Stream& signal_stream);

  layer::NpuLmHead get_npu_lm_head() { return model_->get_npu_lm_head(); };

  void set_npu_lm_head(layer::NpuLmHead& head) {
    model_->set_npu_lm_head(head);
  };

  layer::NpuWordEmbedding get_npu_word_embedding() {
    return model_->get_npu_word_embedding();
  };

  void set_npu_word_embedding(layer::NpuWordEmbedding& embedding) {
    model_->set_npu_word_embedding(embedding);
  };

#endif
  layer::LmHead get_lm_head() { return model_->get_lm_head(); };

  void set_lm_head(layer::LmHead& head) { model_->set_lm_head(head); };

  layer::WordEmbedding get_word_embedding() {
    return model_->get_word_embedding();
  };

  void set_word_embedding(layer::WordEmbedding& embedding) {
    model_->set_word_embedding(embedding);
  };

  torch::Tensor dspark_markov_bias(const torch::Tensor& previous_token_ids) {
    return model_->dspark_markov_bias(previous_token_ids);
  }

  void dspark_markov_bias_out(const torch::Tensor& previous_token_ids,
                              torch::Tensor markov_embedding,
                              torch::Tensor output) {
    model_->dspark_markov_bias_out(
        previous_token_ids, markov_embedding, output);
  }

  torch::Tensor dspark_confidence_probs(const torch::Tensor& hidden_all,
                                        const torch::Tensor& prev_matrix) {
    return model_->dspark_confidence_probs(hidden_all, prev_matrix);
  }
  bool has_dspark_confidence_head() const {
    return model_->has_dspark_confidence_head();
  }

  bool share_weights_from(LLMWorkerImpl& source) {
    return model_->share_weights_from(*source.model_);
  }

  // DFlash-specific delegate: eagerly project target hidden into the draft's
  // per-layer KV cache. Runs outside the executor because the pass has no
  // attention and its shape doesn't match the decode graph. See CausalLM.
  ModelOutput write_context_kv(const torch::Tensor& target_hidden,
                               const torch::Tensor& positions,
                               const torch::Tensor& device_cache_slots,
                               const ModelInputParams& input_params) {
    return model_->write_context_kv(
        target_hidden, positions, device_cache_slots, kv_caches_, input_params);
  }

 protected:
  std::unique_ptr<BeamSearcher> beam_searcher_;

 private:
  std::optional<ForwardOutput> execute_on_stream(
      const ForwardInput& input,
      Stream& compute_stream,
      bool record_ready_event,
      const PreparedSlotBinding* prepared_binding,
      const PreparedModelOutputWorkspace* output_workspace = nullptr,
      bool retain_input_for_async_output = true);
};

}  // namespace xllm
