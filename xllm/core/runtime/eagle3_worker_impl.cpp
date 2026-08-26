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

#include "eagle3_worker_impl.h"

#include <glog/logging.h>

#include "common/global_flags.h"
#include "core/framework/config/speculative_config.h"
#include "core/framework/speculative/mtp_async_state.h"
#include "framework/model_loader.h"

namespace xllm {

namespace {

runtime::Options eagle3_main_options(const runtime::Options& options) {
  auto opts = options;
  opts.enable_schedule_overlap(false)
      .is_draft_engine(false)
      .enable_graph_aux_hidden_states(true);
  return opts;
}

runtime::Options eagle3_draft_options(const runtime::Options& options) {
  auto opts = options;
  opts.enable_schedule_overlap(false)
      .is_draft_engine(true)
      .num_decoding_tokens(1)
      .num_speculative_tokens(0)
      .enable_graph_aux_hidden_states(false);
  return opts;
}

}  // namespace

Eagle3WorkerImpl::Eagle3WorkerImpl(const ParallelArgs& parallel_args,
                                   const torch::Device& device,
                                   const runtime::Options& options)
    : MTPWorkerImpl(
          parallel_args,
          device,
          options,
          eagle3_main_options(options),
          eagle3_draft_options(options),
          ::xllm::SpeculativeConfig::get_instance().enable_opt_validate_probs(),
          /*enable_adaptive_speculative_decode=*/false) {
  CHECK_LE(parallel_args.cp_size(), 1)
      << "EAGLE-3 speculative decoding does not support context parallelism "
         "(cp_size > 1).";
}

bool Eagle3WorkerImpl::init_model(const std::string& model_weights_path,
                                  int32_t random_seed,
                                  MasterStatus master_status) {
  // Call parent's init_model first
  bool result =
      MTPWorkerImpl::init_model(model_weights_path, random_seed, master_status);

  // Load hot_token_id_ directly from state_dict (EAGLE-3 specific)
  // This should be done after draft model is loaded
  if (draft_impl_->get_status() == WorkerImpl::Status::LOADED) {
    // d2t stores diffs between draft id and target id
    // hot_token_id = d2t + arange(d2t.size(0))
    auto model_loader = ModelLoader::create(model_weights_path);
    auto& state_dicts = model_loader->get_state_dicts();
    for (const auto& state_dict : state_dicts) {
      torch::Tensor d2t_tensor = state_dict->get_tensor("d2t");
      if (d2t_tensor.defined()) {
        auto arange_tensor = torch::arange(d2t_tensor.size(0));
        hot_token_id_ = d2t_tensor + arange_tensor;
        hot_token_id_ = hot_token_id_.to(torch::kLong).to(device_);
        LOG(INFO) << "Eagle3WorkerImpl: Loaded d2t tensor from state_dict, "
                     "hot_token_id size: "
                  << hot_token_id_.size(0);
        break;
      }
    }

    const int64_t draft_vocab_size =
        draft_impl_->context_.get_model_args().vocab_size();
    const int64_t target_vocab_size =
        impl_->context_.get_model_args().vocab_size();
    CHECK_GT(draft_vocab_size, 0);
    CHECK_GT(target_vocab_size, 0);
    if (!hot_token_id_.defined()) {
      CHECK_EQ(draft_vocab_size, target_vocab_size)
          << "Eagle3 draft vocab is a strict target subset but the draft "
             "checkpoint has no root-level d2t token mapping.";
    } else {
      CHECK_EQ(hot_token_id_.dim(), 1)
          << "Eagle3 d2t token mapping must be one-dimensional.";
      CHECK_EQ(hot_token_id_.numel(), draft_vocab_size)
          << "Eagle3 d2t token mapping must cover the complete draft vocab.";
      if (hot_token_id_.numel() > 0) {
        const int64_t minimum_target_token =
            hot_token_id_.min().item<int64_t>();
        const int64_t maximum_target_token =
            hot_token_id_.max().item<int64_t>();
        CHECK_GE(minimum_target_token, 0)
            << "Eagle3 d2t token mapping contains a negative target token.";
        CHECK_LT(maximum_target_token, target_vocab_size)
            << "Eagle3 d2t token mapping exceeds the target vocab.";
      }
    }
  }

  return result;
}

int64_t Eagle3WorkerImpl::get_embedding_placeholder_size() {
  const int64_t target_hidden = context_.get_model_args().hidden_size();
  return 3 * target_hidden;
}

void Eagle3WorkerImpl::process_draft_sample_output(
    SampleOutput& sample_output) {
  // Keep probability compression behavior fully aligned with MTP.
  MTPWorkerImpl::process_draft_sample_output(sample_output);

  // EAGLE-3 specific: map draft token IDs to target token IDs.
  if (!hot_token_id_.defined() || !sample_output.next_tokens.defined() ||
      sample_output.next_tokens.numel() == 0) {
    return;
  }

  sample_output.next_tokens =
      hot_token_id_.index_select(0, sample_output.next_tokens);
}

void Eagle3WorkerImpl::process_prepared_draft_sample_output(
    SampleOutput& sample_output,
    torch::Tensor fixed_token_ids) {
  MTPWorkerImpl::process_draft_sample_output(sample_output);
  CHECK(sample_output.next_tokens.defined());
  CHECK_EQ(sample_output.next_tokens.numel(), fixed_token_ids.numel());
  if (hot_token_id_.defined() && sample_output.next_tokens.numel() > 0) {
    mtp_async::map_draft_token_ids_to_target_out(
        hot_token_id_, sample_output.next_tokens, fixed_token_ids);
  } else {
    mtp_async::copy_prepared_draft_token_ids_out(sample_output.next_tokens,
                                                 fixed_token_ids);
  }
  sample_output.next_tokens = fixed_token_ids;
}

}  // namespace xllm
