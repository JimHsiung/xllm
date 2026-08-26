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

#include "runtime/dflash_worker_impl.h"

#include <glog/logging.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "common/metrics.h"
#include "core/framework/config/kernel_config.h"
#include "core/framework/config/scheduler_config.h"
#include "core/framework/config/speculative_config.h"
#include "core/framework/sampling/rejection_sampler.h"
#include "core/framework/speculative/adaptive_pruning_helpers.h"
#include "core/framework/speculative/speculative_profile_registry.h"
#include "framework/model/model_args.h"
#include "framework/parallel_state/process_group.h"
#include "framework/sampling/sampling_params.h"
#if defined(USE_NPU) || defined(USE_MLU)
#include "framework/kv_cache_transfer/mooncake_kv_cache_transfer.h"
#endif
#if defined(USE_NPU)
#include "core/layers/npu_torch/deepseek_sparse_attention.h"
#include "framework/kv_cache_transfer/kv_transfer_completion.h"
#endif
#include "core/framework/speculative/spec_input_builder.h"
#include "core/framework/speculative/spec_verify.h"
#include "runtime/prepared_task/block_spec_prepared_task_backend.h"
#include "util/json_reader.h"
#include "util/timer.h"
#include "util/verbose_trace_logger.h"

namespace xllm {
namespace dflash_detail {

PreparedDecodeCacheSlotViews view_prepared_decode_cache_slots(
    const torch::Tensor& query_cache_slots,
    const torch::Tensor& target_cache_slots,
    int64_t batch_size,
    int32_t num_speculative_tokens,
    bool sample_from_anchor) {
  CHECK_GT(batch_size, 0);
  CHECK_GT(num_speculative_tokens, 0);
  const int64_t query_width =
      decode_draft_width(num_speculative_tokens, sample_from_anchor);
  const int64_t target_width = num_speculative_tokens + 1;
  CHECK_EQ(query_cache_slots.numel(), batch_size * query_width);
  CHECK_EQ(target_cache_slots.numel(), batch_size * target_width);
  return PreparedDecodeCacheSlotViews{
      query_cache_slots.view({batch_size, query_width}),
      target_cache_slots.view({batch_size, target_width})};
}

void check_fixed_prepared_output_binding(const torch::Tensor& actual,
                                         const torch::Tensor& fixed_storage,
                                         const torch::Device& expected_device,
                                         torch::ScalarType expected_dtype,
                                         int64_t expected_numel,
                                         std::string_view tensor_name) {
  CHECK_GT(expected_numel, 0);
  CHECK(actual.defined()) << "Prepared Block-Spec " << tensor_name
                          << " must be defined";
  CHECK(fixed_storage.defined())
      << "Prepared Block-Spec fixed " << tensor_name << " must be defined";
  CHECK_EQ(actual.numel(), expected_numel)
      << "Prepared Block-Spec " << tensor_name
      << " must have the configured element count";
  CHECK_EQ(fixed_storage.numel(), expected_numel)
      << "Prepared Block-Spec fixed " << tensor_name
      << " must have the configured element count";
  CHECK_EQ(actual.device(), expected_device)
      << "Prepared Block-Spec " << tensor_name
      << " must share the Worker device";
  CHECK_EQ(fixed_storage.device(), expected_device)
      << "Prepared Block-Spec fixed " << tensor_name
      << " must share the Worker device";
  CHECK_EQ(actual.scalar_type(), expected_dtype)
      << "Prepared Block-Spec " << tensor_name << " has an incompatible dtype";
  CHECK_EQ(fixed_storage.scalar_type(), expected_dtype)
      << "Prepared Block-Spec fixed " << tensor_name
      << " has an incompatible dtype";
  CHECK(actual.is_contiguous())
      << "Prepared Block-Spec " << tensor_name << " must be contiguous";
  CHECK(fixed_storage.is_contiguous())
      << "Prepared Block-Spec fixed " << tensor_name << " must be contiguous";
  CHECK_EQ(actual.data_ptr(), fixed_storage.data_ptr())
      << "Prepared Block-Spec " << tensor_name
      << " replaced the fixed output storage";
}

void check_context_kv_write_tensor_contract(
    const torch::Tensor& context_hidden,
    const torch::Tensor& positions,
    const torch::Tensor& cache_slots,
    const torch::Device& expected_device,
    int64_t expected_hidden_size,
    torch::ScalarType expected_hidden_dtype) {
  CHECK_GT(expected_hidden_size, 0);
  CHECK(context_hidden.defined()) << "DFlash context hidden must be defined";
  CHECK(positions.defined()) << "DFlash context positions must be defined";
  CHECK(cache_slots.defined()) << "DFlash context cache slots must be defined";
  CHECK_EQ(context_hidden.dim(), 2)
      << "DFlash context hidden must be two-dimensional";
  CHECK_EQ(context_hidden.size(/*dim=*/1), expected_hidden_size)
      << "DFlash context hidden width must match the configured context "
         "hidden size";
  CHECK_EQ(positions.dim(), 1)
      << "DFlash context positions must be one-dimensional";
  CHECK_EQ(cache_slots.dim(), 1)
      << "DFlash context cache slots must be one-dimensional";
  const int64_t row_count = context_hidden.size(/*dim=*/0);
  CHECK_EQ(positions.numel(), row_count)
      << "DFlash context positions must match the context hidden rows";
  CHECK_EQ(cache_slots.numel(), row_count)
      << "DFlash context cache slots must match the context hidden rows";
  CHECK_EQ(context_hidden.device(), expected_device)
      << "DFlash context hidden must share the contract device";
  CHECK_EQ(positions.device(), expected_device)
      << "DFlash context positions must share the contract device";
  CHECK_EQ(cache_slots.device(), expected_device)
      << "DFlash context cache slots must share the contract device";
  CHECK_EQ(context_hidden.scalar_type(), expected_hidden_dtype)
      << "DFlash context hidden has an incompatible dtype";
  CHECK_EQ(positions.scalar_type(), torch::kInt)
      << "DFlash context positions must use int32 dtype";
  CHECK_EQ(cache_slots.scalar_type(), torch::kInt)
      << "DFlash context cache slots must use int32 dtype";
}

}  // namespace dflash_detail
namespace {

void trace_block_spec_step_state(std::string_view algorithm,
                                 const std::vector<int32_t>& embedding_ids,
                                 const std::vector<std::string>& request_ids,
                                 const torch::Tensor& accepted_tokens,
                                 const torch::Tensor& base_positions,
                                 const torch::Tensor& base_kv_seq_lens) {
  if (!VerboseTraceLogger::get_instance().enabled()) {
    return;
  }

  CHECK(accepted_tokens.defined());
  CHECK_EQ(accepted_tokens.dim(), 2);
  const int64_t batch_size = accepted_tokens.size(0);
  const int64_t token_width = accepted_tokens.size(1);
  CHECK_EQ(embedding_ids.size(), static_cast<size_t>(batch_size));
  CHECK(request_ids.empty() ||
        request_ids.size() == static_cast<size_t>(batch_size));
  CHECK(base_positions.defined());
  CHECK_EQ(base_positions.numel(), batch_size);
  CHECK(base_kv_seq_lens.defined());
  CHECK_EQ(base_kv_seq_lens.numel(), batch_size);

  const torch::Tensor accepted_tokens_cpu =
      accepted_tokens.to(torch::kCPU, torch::kInt64).contiguous();
  const torch::Tensor base_positions_cpu =
      base_positions.to(torch::kCPU, torch::kInt64).contiguous().view({-1});
  const torch::Tensor base_kv_seq_lens_cpu =
      base_kv_seq_lens.to(torch::kCPU, torch::kInt64).contiguous().view({-1});
  const int64_t* token_data = accepted_tokens_cpu.const_data_ptr<int64_t>();
  const int64_t* position_data = base_positions_cpu.const_data_ptr<int64_t>();
  const int64_t* kv_seq_len_data =
      base_kv_seq_lens_cpu.const_data_ptr<int64_t>();
  for (int64_t row = 0; row < batch_size; ++row) {
    int64_t committed_length = 0;
    bool saw_padding = false;
    std::string token_list;
    token_list.reserve(static_cast<size_t>(token_width) * 12);
    for (int64_t column = 0; column < token_width; ++column) {
      const int64_t token = token_data[row * token_width + column];
      if (column > 0) {
        token_list.push_back(',');
      }
      token_list.append(std::to_string(token));
      if (token < 0) {
        saw_padding = true;
        continue;
      }
      CHECK(!saw_padding)
          << "Block-Spec accepted-token padding must be a contiguous suffix";
      ++committed_length;
    }
    const std::string_view request_id =
        request_ids.empty()
            ? std::string_view()
            : std::string_view(request_ids[static_cast<size_t>(row)]);
    XLLM_VERBOSE_TRACE()
        << "event=speculative_step_state algorithm=" << algorithm
        << " request_id="
        << (request_id.empty() ? std::string_view("-") : request_id)
        << " embedding_id=" << embedding_ids[static_cast<size_t>(row)]
        << " base_position=" << position_data[row]
        << " base_kv_seq_len=" << kv_seq_len_data[row]
        << " committed_length=" << committed_length << " accepted_draft_length="
        << std::max<int64_t>(committed_length - 1, 0)
        << " tokens=" << token_list;
  }
}

void trace_block_spec_step_state_from_host_input(
    std::string_view algorithm,
    const ForwardInput& input,
    const torch::Tensor& accepted_tokens) {
  if (!VerboseTraceLogger::get_instance().enabled()) {
    return;
  }

  const int64_t batch_size = accepted_tokens.size(0);
  const torch::Tensor& positions =
      input.positions_host.defined() ? input.positions_host : input.positions;
  CHECK(positions.defined());
  CHECK_EQ(positions.numel(), batch_size);
  const Slice<int32_t> kv_seq_lens =
      input.input_params.attention.host.kv_seq_lens;
  CHECK_EQ(kv_seq_lens.size(), static_cast<size_t>(batch_size));
  std::vector<int64_t> base_kv_seq_lens;
  base_kv_seq_lens.reserve(static_cast<size_t>(batch_size));
  for (int64_t row = 0; row < batch_size; ++row) {
    base_kv_seq_lens.emplace_back(kv_seq_lens[static_cast<size_t>(row)]);
  }
  trace_block_spec_step_state(
      algorithm,
      input.input_params.embedding.embedding_ids,
      input.input_params.embedding.request_ids,
      accepted_tokens,
      positions,
      torch::tensor(base_kv_seq_lens, torch::dtype(torch::kLong)));
}

void trace_block_spec_step_state_from_target_input(
    std::string_view algorithm,
    const ForwardInput& input,
    const torch::Tensor& accepted_tokens,
    const torch::Tensor& base_positions,
    int64_t target_width) {
  if (!VerboseTraceLogger::get_instance().enabled()) {
    return;
  }

  const int64_t batch_size = accepted_tokens.size(0);
  const torch::Tensor& target_kv_seq_lens =
      input.input_params.attention.device.kv_seq_lens;
  CHECK(target_kv_seq_lens.defined());
  torch::Tensor base_kv_seq_lens;
  if (target_kv_seq_lens.numel() == batch_size) {
    base_kv_seq_lens = target_kv_seq_lens - (target_width - 1);
  } else {
    CHECK_EQ(target_kv_seq_lens.numel(), batch_size * target_width)
        << "Block-Spec trace requires chunked or tokenwise Target KV metadata";
    base_kv_seq_lens = target_kv_seq_lens.view({batch_size, target_width})
                           .select(/*dim=*/1, /*index=*/0);
  }
  trace_block_spec_step_state(algorithm,
                              input.input_params.embedding.embedding_ids,
                              input.input_params.embedding.request_ids,
                              accepted_tokens,
                              base_positions,
                              base_kv_seq_lens);
}

// Per-rank sampling RNG can diverge across the tensor-parallel group.
// Broadcasting the sampled draft/accepted tokens to the group's rank 0 keeps
// every rank's cached draft probs and accepted prefixes identical. No-op for a
// single rank (world_size <= 1).
ProcessGroup* spec_broadcast_group(const ParallelArgs& parallel_args) {
  return parallel_args.tp_group_ != nullptr ? parallel_args.tp_group_
                                            : parallel_args.process_group_;
}

void broadcast_spec_tokens(torch::Tensor& tokens,
                           ProcessGroup* pg,
                           int32_t root_rank = 0) {
  if (pg == nullptr || pg->world_size() <= 1 || !tokens.defined()) {
    return;
  }
  tokens = tokens.contiguous();
  pg->broadcast(tokens, root_rank);
}

runtime::Options target_options(const runtime::Options& options) {
  runtime::Options opts = options;
  opts.enable_schedule_overlap(false)
      .is_draft_engine(false)
      .enable_graph_aux_hidden_states(true);
  return opts;
}

runtime::Options draft_options(const runtime::Options& options) {
  // DSpark sizes its attention window from num_speculative_tokens; other
  // DFlash-style drafts still run one step at a time.
  const int32_t draft_num_speculative_tokens =
      options.speculative_algorithm() == "DSpark"
          ? options.num_speculative_tokens()
          : 0;
  runtime::Options opts = options;
  opts.enable_schedule_overlap(false)
      .is_draft_engine(true)
      .num_decoding_tokens(1)
      .num_speculative_tokens(draft_num_speculative_tokens)
      .enable_graph_aux_hidden_states(false);
  return opts;
}

void expand_block_parallel_sequence_rows(ModelInputParams& input_params,
                                         int32_t query_width) {
  input_params.meta.num_sequences *= query_width;
  if (input_params.meta.actual_num_sequences > 0) {
    input_params.meta.actual_num_sequences *= query_width;
  }
}

// Stage a host int32 vector to `device` on the caller's active stream.
torch::Tensor cpu_int_vec_to_device(const std::vector<int32_t>& values,
                                    const Device& device) {
  return safe_to(
      specBuilder::make_cpu_int_tensor(values),
      torch::TensorOptions().dtype(torch::kInt).device(device.unwrap()),
      /*non_blocking=*/true);
}

void repeat_sampling_tensor(torch::Tensor& tensor, int32_t repeats) {
  if (tensor.defined()) {
    tensor = tensor.repeat_interleave(/*repeats=*/repeats, /*dim=*/0);
  }
}

void repeat_sampling_params(
    SamplingParameters& sampling_params,
    int32_t repeats,
    const torch::Tensor& fixed_do_sample_host,
    const torch::Tensor& fixed_repeated_sampling_storage) {
  if (fixed_repeated_sampling_storage.defined()) {
    repeat_speculative_sampling_metadata_out(
        sampling_params, repeats, fixed_repeated_sampling_storage);
  } else {
    repeat_sampling_tensor(sampling_params.frequency_penalties, repeats);
    repeat_sampling_tensor(sampling_params.presence_penalties, repeats);
    repeat_sampling_tensor(sampling_params.repetition_penalties, repeats);
    repeat_sampling_tensor(sampling_params.temperatures, repeats);
    repeat_sampling_tensor(sampling_params.top_p, repeats);
    repeat_sampling_tensor(sampling_params.top_k, repeats);
    repeat_sampling_tensor(sampling_params.unique_token_ids, repeats);
    repeat_sampling_tensor(sampling_params.unique_token_counts, repeats);
    repeat_sampling_tensor(sampling_params.unique_token_ids_lens, repeats);
  }
  if (fixed_do_sample_host.defined()) {
    CHECK(sampling_params.all_greedy_sample)
        << "Fixed zero do_sample staging requires greedy sampling";
    sampling_params.do_sample = specBuilder::fill_cpu_bool_out(
        /*value=*/false,
        sampling_params.selected_token_idxes.numel(),
        fixed_do_sample_host);
  } else {
    repeat_sampling_tensor(sampling_params.do_sample, repeats);
  }
}

SpeculativePreparedHostInputWorkspace allocate_prepared_host_input_workspace(
    int64_t capacity,
    const torch::Tensor& repeated_sampling_storage = torch::Tensor()) {
  CHECK_GT(capacity, 0);
  const torch::TensorOptions int_options = torch::TensorOptions()
                                               .dtype(torch::kInt)
                                               .device(torch::kCPU)
                                               .pinned_memory(true);
  const torch::TensorOptions bool_options = torch::TensorOptions()
                                                .dtype(torch::kBool)
                                                .device(torch::kCPU)
                                                .pinned_memory(true);
  return SpeculativePreparedHostInputWorkspace{
      torch::empty({capacity}, int_options),
      torch::empty({capacity}, int_options),
      torch::empty({capacity}, int_options),
      torch::empty({capacity}, int_options),
      torch::empty({capacity}, bool_options),
      repeated_sampling_storage};
}

void clear_selected_embeddings(ForwardOutput& output) {
  output.sample_output.selected_embeddings = torch::Tensor();
}

void clear_all_output_embeddings(ForwardOutput& output) {
  output.sample_output.embeddings = torch::Tensor();
  clear_selected_embeddings(output);
}

void release_prepared_model_intermediates(ForwardOutput& output) {
  output.logits = torch::Tensor();
  output.selected_hidden = torch::Tensor();
  output.mtp_topk_state.reset();
}

void record_metadata_ready_event(Stream& stream, ForwardInput& input) {
  input.metadata_ready_event = stream.record_event_or_sync();
}

void wait_metadata_ready_event(const ForwardInput& input, Stream& stream) {
  CHECK(stream.wait_event(input.metadata_ready_event))
      << "failed to wait DFlash metadata ready event";
}

std::optional<ForwardOutput> run_llm_no_sync_impl(
    LLMWorkerImpl& worker,
    const ForwardInput& input,
    Stream& prepare_stream,
    Stream& compute_stream,
    ForwardInput* processed_output = nullptr) {
  ForwardInput processed_input;
  worker.prepare_work_before_execute_on_stream(
      input, processed_input, prepare_stream);
  std::optional<ForwardOutput> output =
      worker.execute_no_sync_on_stream(processed_input, compute_stream);
  if (processed_output != nullptr) {
    *processed_output = std::move(processed_input);
  }
  return output;
}

void build_query_rows(const ForwardInput& input,
                      int32_t mask_token_id,
                      int32_t num_speculative_tokens,
                      int32_t block_size,
                      bool sample_from_anchor,
                      bool use_block_parallel_rows,
                      specBuilder::DecodeBuildBuffers& buf,
                      std::vector<int32_t>& selected_idxes) {
  const int32_t num_sequences = input.input_params.meta.num_sequences;
  // DFlash: (1 + N) block — slot 0 is the un-selected anchor (real token), the
  // N mask positions are sampled. DSpark: N-wide block — every position is a
  // prediction; slot 0 still carries the real token but is itself sampled
  // (predicts the first draft token), positions 1..N-1 are masks.
  const int32_t query_width = dflash_detail::decode_draft_width(
      num_speculative_tokens, sample_from_anchor);
  specBuilder::DecodeRowContext row_ctx =
      specBuilder::make_decode_row_context(input);
  if (use_block_parallel_rows) {
    CHECK(sample_from_anchor)
        << "block-parallel rows require DSpark anchor sampling";
    CHECK(row_ctx.model_managed_multiblock)
        << "DSV4 block-parallel rows require grouped KV tables";
  }
  Slice<int32_t> token_ids = {
      input.token_ids_host.data_ptr<int32_t>(),
      static_cast<size_t>(input.token_ids_host.numel())};
  CHECK_GE(static_cast<int32_t>(token_ids.size()), num_sequences)
      << "DFlash input token_ids size is smaller than num_sequences.";

  buf.out_token_ids.reserve(num_sequences * query_width);
  buf.out_positions.reserve(num_sequences * query_width);
  buf.out_new_cache_slots.reserve(num_sequences * query_width);
  const int32_t metadata_rows =
      use_block_parallel_rows ? num_sequences * query_width : num_sequences;
  buf.out_kv_seq_lens.reserve(metadata_rows);
  buf.out_q_seq_lens.reserve(metadata_rows);
  buf.out_q_cu_seq_lens.reserve(metadata_rows + 1);
  buf.out_q_cu_seq_lens.emplace_back(0);

  selected_idxes.reserve(num_sequences * query_width);

  specBuilder::RowSpec row_template;
  row_template.append_kv_len = use_block_parallel_rows;
  row_template.kv_len_offset =
      use_block_parallel_rows ? std::make_optional<int32_t>(query_width - 1)
                              : std::nullopt;
  row_template.append_q_len_one = use_block_parallel_rows;
  row_template.append_block_table = use_block_parallel_rows;

  for (int32_t seq_id = 0; seq_id < num_sequences; ++seq_id) {
    for (int32_t query_idx = 0; query_idx < query_width; ++query_idx) {
      specBuilder::RowSpec row = row_template;
      row.seq_id = seq_id;
      row.token_id = query_idx == 0 ? token_ids[seq_id] : mask_token_id;
      row.position_offset = query_idx;
      specBuilder::append_decode_row(row_ctx, row, block_size, buf);
      // DFlash skips slot 0 (anchor, not sampled); DSpark samples every slot.
      if (sample_from_anchor || query_idx > 0) {
        selected_idxes.emplace_back(seq_id * query_width + query_idx);
      }
    }

    if (use_block_parallel_rows) {
      continue;
    }
    specBuilder::append_q_seq_len(
        buf.out_q_seq_lens, buf.out_q_cu_seq_lens, query_width);
    // kv_len must cover exactly this block's max absolute position (anchor +
    // query_width - 1), not a fixed anchor + num_speculative_tokens: DFlash's
    // (1+N)-wide block and DSpark's N-wide block (sample_from_anchor) advance
    // the max position by different amounts, and using num_speculative_tokens
    // unconditionally overshoots by one slot for DSpark, exposing an
    // uninitialized cache slot to this block's non-causal attention.
    const int32_t kv_len =
        specBuilder::calc_kv_len(input.input_params.attention.host.kv_seq_lens,
                                 seq_id,
                                 /*offset=*/0) +
        (query_width - 1);
    specBuilder::update_kv_seq_lens_and_max(
        buf.out_kv_seq_lens, kv_len, buf.meta.kv_max_seq_len);
  }
}

std::vector<int64_t> build_accepted_context_rows(
    const ForwardInput& input,
    const torch::Tensor& accepted_tokens_cpu,
    int32_t block_size,
    specBuilder::DecodeBuildBuffers& buf) {
  const int32_t batch_size = static_cast<int32_t>(accepted_tokens_cpu.size(0));
  const int32_t token_width = static_cast<int32_t>(accepted_tokens_cpu.size(1));
  CHECK_EQ(input.input_params.meta.num_sequences, batch_size)
      << "DFlash accepted token batch mismatch.";

  specBuilder::DecodeRowContext row_ctx =
      specBuilder::make_decode_row_context(input);
  std::vector<int64_t> accepted_idxes;
  accepted_idxes.reserve(static_cast<size_t>(accepted_tokens_cpu.numel()));
  buf.out_positions.reserve(buf.out_positions.size() +
                            static_cast<size_t>(accepted_tokens_cpu.numel()));
  buf.out_new_cache_slots.reserve(
      buf.out_new_cache_slots.size() +
      static_cast<size_t>(accepted_tokens_cpu.numel()));

  const int64_t* accepted_tokens_data =
      accepted_tokens_cpu.const_data_ptr<int64_t>();
  for (int32_t seq_id = 0; seq_id < batch_size; ++seq_id) {
    const int64_t row_offset = static_cast<int64_t>(seq_id) * token_width;
    for (int32_t token_idx = 0; token_idx < token_width; ++token_idx) {
      if (accepted_tokens_data[row_offset + token_idx] < 0) {
        break;
      }

      specBuilder::RowSpec row;
      row.seq_id = seq_id;
      row.position_offset = token_idx;
      row.append_token = false;
      row.append_kv_len = false;
      specBuilder::append_decode_row(row_ctx, row, block_size, buf);
      accepted_idxes.emplace_back(row_offset + token_idx);
    }
  }

  CHECK(!accepted_idxes.empty())
      << "DFlash accepted context must not be empty.";
  CHECK_EQ(buf.out_new_cache_slots.size(), buf.out_positions.size())
      << "DFlash accepted context slots/positions mismatch.";
  return accepted_idxes;
}

}  // namespace

DFlashWorkerImpl::DFlashWorkerImpl(const ParallelArgs& parallel_args,
                                   const torch::Device& device,
                                   const runtime::Options& options)
    : SpeculativeWorkerImpl(parallel_args,
                            device,
                            options,
                            target_options(options)) {
  // DFlash feeds the target's captured intermediate-layer aux hidden states
  // into the draft's context K/V. Under context parallelism the worker only
  // exposes the lm_head-gathered final hidden (see llm_worker_impl.cpp), not
  // the aux hidden, so the draft would silently receive the wrong tensor.
  // Reject cp_size > 1 until aux-hidden plumbing under CP is implemented.
  CHECK_LE(parallel_args.cp_size(), 1)
      << "Block-diffusion speculative decoding does not support context "
         "parallelism (cp_size > 1).";
  draft_impl_ = std::make_unique<LLMWorkerImpl>(
      parallel_args, device, draft_options(options));

  // Adaptive per-seq validate pruning. DP is supported: the worker gathers
  // each rank's true validate token count over the DP group before the target
  // forward (see sync_dp_global_token_nums_after_prune).
  const bool enable_adaptive = options.enable_adaptive_speculative_decode() &&
                               options.num_speculative_tokens() > 1;
  if (enable_adaptive) {
    adaptive_spec_controller_ =
        std::make_unique<AdaptiveSpeculativeController>(options);
  }
}

int64_t DFlashWorkerImpl::prepared_dspark_markov_rank() const {
  LOG(FATAL) << "DFlash does not have a DSpark Markov head";
  return 0;
}

int64_t DFlashWorkerImpl::prepared_dspark_vocab_size() const {
  LOG(FATAL) << "DFlash does not have a DSpark vocabulary";
  return 0;
}

void DFlashWorkerImpl::launch_prepared_dspark_markov_sample(
    const torch::Tensor& /*base_logits*/,
    const torch::Tensor& /*anchor_token_ids*/,
    const SamplingParameters& /*sampling_params*/,
    int64_t /*row_count*/,
    int32_t /*block_step*/,
    dspark_detail::PreparedSamplingWorkspace& /*workspace*/) const {
  LOG(FATAL) << "DFlash cannot launch a DSpark Markov sample";
}

void DFlashWorkerImpl::launch_prepared_dspark_token_broadcast(
    const SamplingParameters& /*sampling_params*/,
    int64_t /*row_count*/,
    int32_t /*block_step*/,
    dspark_detail::PreparedSamplingWorkspace& /*workspace*/) const {
  LOG(FATAL) << "DFlash cannot launch a DSpark token broadcast";
}

class DFlashWorkerImpl::PreparedTaskBackend final
    : public BlockSpecPreparedTaskBackendBase {
 public:
  PreparedTaskBackend(DFlashWorkerImpl* worker,
                      const BlockSpecPreparedTaskBufferConfig& config)
      : BlockSpecPreparedTaskBackendBase(config),
        worker_(CHECK_NOTNULL(worker)),
        algorithm_(worker_->options_.speculative_algorithm() == "DSpark"
                       ? BlockSpecAlgorithm::DSPARK
                       : BlockSpecAlgorithm::DFLASH),
        max_rows_(config.max_rows),
        speculative_width_(worker_->options_.num_speculative_tokens()),
        query_width_(dflash_detail::decode_draft_width(
            worker_->options_.num_speculative_tokens(),
            worker_->sample_from_anchor())),
        accepted_token_capacity_(config.accepted_token_capacity),
        context_hidden_size_(config.context_hidden_size),
        draft_hidden_size_(
            worker_->draft_impl_->context_.get_model_args().hidden_size()) {
    CHECK(worker_->impl_ != nullptr);
    CHECK(worker_->draft_impl_ != nullptr);
    CHECK(worker_->options_.speculative_algorithm() == "DFlash" ||
          worker_->options_.speculative_algorithm() == "DSpark");
    CHECK_GT(speculative_width_, 0);
    CHECK_GT(query_width_, 0);
    CHECK_EQ(accepted_token_capacity_, speculative_width_ + 1);
    CHECK_GT(context_hidden_size_, 0);
    CHECK_GT(draft_hidden_size_, 0);

    const ModelArgs& target_args = worker_->impl_->context_.get_model_args();
    if (util::is_deepseek_v4_model_type(target_args.model_type())) {
      const int64_t max_model_tokens =
          std::max<int64_t>({target_args.max_seq_len(),
                             target_args.max_position_embeddings(),
                             target_args.window_size()});
      CHECK_GT(max_model_tokens, 0)
          << "DeepSeek-V4 Prepared cache-slot mapping requires a positive "
             "model sequence capacity";
      max_swa_block_table_width_ =
          (max_model_tokens + config.block_size - 1) / config.block_size;
    }

    const torch::TensorOptions token_options =
        torch::TensorOptions().dtype(config.token_dtype).device(config.device);
    const torch::TensorOptions hidden_options =
        torch::TensorOptions().dtype(config.hidden_dtype).device(config.device);
    const torch::TensorOptions position_options =
        torch::TensorOptions()
            .dtype(config.position_dtype)
            .device(config.device);
    const torch::TensorOptions index_options =
        torch::TensorOptions().dtype(torch::kLong).device(config.device);
    const torch::TensorOptions mask_options =
        torch::TensorOptions().dtype(torch::kBool).device(config.device);
    const torch::TensorOptions host_byte_options = torch::TensorOptions()
                                                       .dtype(torch::kUInt8)
                                                       .device(torch::kCPU)
                                                       .pinned_memory(true);
    CHECK_GT(worker_->options_.max_tokens_per_batch(), 0);
    const int64_t dsa_geometry_token_capacity =
        std::max<int64_t>(worker_->options_.max_tokens_per_batch(),
                          max_rows_ * accepted_token_capacity_);
    const int64_t dsa_compact_storage_capacity =
        dsa_geometry_token_capacity * 2;

    slots_.reserve(static_cast<size_t>(config.slot_count));
    for (int32_t slot_id = 0; slot_id < config.slot_count; ++slot_id) {
      SlotResources resources;
      resources.prefill_next_tokens = torch::empty({max_rows_}, token_options);
      resources.prefill_selected_hidden =
          torch::empty({max_rows_, context_hidden_size_}, hidden_options);
      resources.draft_next_tokens =
          torch::empty({max_rows_ * speculative_width_}, token_options);
      resources.draft_selected_hidden = torch::empty(
          {max_rows_ * speculative_width_, draft_hidden_size_}, hidden_options);
      resources.target_next_tokens =
          torch::empty({max_rows_ * accepted_token_capacity_}, token_options);
      resources.target_context_hidden = torch::empty(
          {max_rows_ * accepted_token_capacity_, context_hidden_size_},
          hidden_options);
      resources.continuation_context_hidden =
          torch::zeros({max_rows_, context_hidden_size_}, hidden_options);
      if (max_swa_block_table_width_ > 0) {
        resources.dsa_device_geometry_workspace =
            std::make_shared<DSADeviceGeometryWorkspace>();
        resources.dsa_device_geometry_workspace->actual_seq_lengths_query =
            torch::empty({max_rows_ + 1}, position_options);
        resources.dsa_device_geometry_workspace->kv_cu_seq_lens =
            torch::empty({max_rows_ + 1}, position_options);
        resources.dsa_device_geometry_workspace->max_seqlen_q =
            torch::empty({1}, position_options);
        resources.dsa_device_geometry_workspace->max_seqlen_kv =
            torch::empty({1}, position_options);
        resources.dsa_device_geometry_workspace->start_pos =
            torch::empty({max_rows_}, position_options);
        resources.dsa_device_geometry_workspace->c4_compact_positions =
            torch::empty({dsa_compact_storage_capacity}, position_options);
        resources.dsa_device_geometry_workspace->c128_compact_positions =
            torch::empty({dsa_compact_storage_capacity}, position_options);
        resources.dsa_device_geometry_workspace->c4_compact_slots =
            torch::empty({dsa_compact_storage_capacity}, position_options);
        resources.dsa_device_geometry_workspace->c128_compact_slots =
            torch::empty({dsa_compact_storage_capacity}, position_options);
        resources.dsa_device_geometry_workspace->swa_block_table = torch::empty(
            {max_rows_ * max_swa_block_table_width_}, position_options);
        resources.dsa_device_geometry_workspace->token_indices =
            torch::empty({dsa_geometry_token_capacity}, index_options);
        resources.dsa_device_geometry_workspace->position_values =
            torch::empty({dsa_geometry_token_capacity}, position_options);
        resources.dsa_device_geometry_workspace->position_remainders =
            torch::empty({dsa_geometry_token_capacity}, position_options);
        resources.dsa_device_geometry_workspace->boundary_mask =
            torch::empty({dsa_geometry_token_capacity}, mask_options);
        resources.dsa_device_geometry_workspace->boundary_ranks =
            torch::empty({dsa_geometry_token_capacity}, index_options);
        resources.dsa_device_geometry_workspace->sentinel_indices =
            torch::empty({dsa_geometry_token_capacity}, index_options);
        resources.dsa_device_geometry_workspace->destination_indices =
            torch::empty({dsa_geometry_token_capacity}, index_options);
        resources.dsa_device_geometry_workspace->token_offsets =
            torch::empty({dsa_geometry_token_capacity}, position_options);
        resources.dsa_device_geometry_workspace->token_candidates =
            torch::empty({dsa_geometry_token_capacity}, position_options);
        resources.dsa_device_geometry_workspace->mapping_valid =
            torch::empty({dsa_geometry_token_capacity}, mask_options);
        resources.dsa_device_geometry_workspace->mapping_valid_aux =
            torch::empty({dsa_geometry_token_capacity}, mask_options);
        resources.dsa_device_geometry_workspace->swa_logical_column_indices =
            torch::empty({max_swa_block_table_width_}, index_options);
        const int64_t swa_matrix_capacity =
            max_rows_ * max_swa_block_table_width_;
        resources.dsa_device_geometry_workspace->swa_physical_columns =
            torch::empty({swa_matrix_capacity}, index_options);
        resources.dsa_device_geometry_workspace->swa_gathered =
            torch::empty({swa_matrix_capacity}, position_options);
        resources.dsa_device_geometry_workspace->swa_valid =
            torch::empty({swa_matrix_capacity}, mask_options);
        resources.dsa_device_geometry_workspace->swa_valid_aux =
            torch::empty({swa_matrix_capacity}, mask_options);
        resources.dsa_device_geometry_workspace->manager_row_indices =
            torch::empty({max_rows_}, index_options);
        resources.dsa_device_geometry_workspace->manager_expanded_block_tables
            .reserve(kMaxDeepseekV4CacheManagers);
        resources.dsa_group_block_table_storage.reserve(
            kMaxDeepseekV4CacheManagers);
        for (int32_t manager_id = 0; manager_id < kMaxDeepseekV4CacheManagers;
             ++manager_id) {
          resources.dsa_device_geometry_workspace->manager_expanded_block_tables
              .emplace_back(
                  torch::empty({swa_matrix_capacity}, position_options));
          resources.dsa_group_block_table_storage.emplace_back(torch::empty(
              {max_rows_, max_swa_block_table_width_}, position_options));
        }
        resources.active_dsa_group_block_tables.reserve(
            kMaxDeepseekV4CacheManagers);
      }
      resources.prefill_swa_slots.reserve(
          static_cast<size_t>(worker_->options_.max_tokens_per_batch()));
      resources.decode_states.reserve(static_cast<size_t>(max_rows_));
      specBuilder::reserve_decode_build_workspace(
          resources.decode_build_workspace,
          max_rows_ * accepted_token_capacity_);
      resources.anchor_host_workspace =
          allocate_prepared_host_input_workspace(max_rows_);
      const int64_t generated_row_capacity =
          max_rows_ * accepted_token_capacity_;
      torch::Tensor repeated_sampling_storage = torch::empty(
          {static_cast<int64_t>(config.input_arena_capacity_bytes)},
          host_byte_options);
      resources.query_host_workspace = allocate_prepared_host_input_workspace(
          generated_row_capacity, repeated_sampling_storage);
      resources.target_host_workspace = allocate_prepared_host_input_workspace(
          generated_row_capacity, repeated_sampling_storage);
      resources.rejection_workspace = GreedyTokenIdRejectionWorkspace{
          torch::empty({max_rows_, accepted_token_capacity_}, token_options),
          torch::empty({max_rows_, speculative_width_}, mask_options),
          torch::empty({max_rows_, accepted_token_capacity_}, mask_options),
          torch::full({max_rows_, accepted_token_capacity_}, -1, token_options),
          torch::empty({max_rows_, accepted_token_capacity_}, token_options)};
      resources.decode_patch_workspace =
          block_spec_async::allocate_block_spec_decode_input_patch_workspace(
              max_rows_,
              query_width_,
              accepted_token_capacity_,
              position_options,
              position_options);
      if (algorithm_ == BlockSpecAlgorithm::DSPARK) {
        resources.dspark_sampling_workspace =
            dspark_detail::allocate_prepared_sampling_workspace(
                max_rows_,
                static_cast<int32_t>(speculative_width_),
                worker_->prepared_dspark_markov_rank(),
                worker_->prepared_dspark_vocab_size(),
                token_options,
                hidden_options);
      }
      slots_.emplace_back(std::move(resources));
    }
  }

  void initialize_state_thread() override {
    worker_->initialize_prepared_task_thread();
  }

  void initialize_launch_thread() override {
    worker_->initialize_prepared_task_thread();
  }

 protected:
  void prepare_staged_input(const ForwardInput& input,
                            ExecutionSlot& slot,
                            const BlockSpecPreparedTaskPlan& plan) override {
    CHECK(plan.algorithm == algorithm_)
        << "Block-Spec Prepared Backend received the wrong algorithm plan";
    CHECK_EQ(plan.speculative_width, speculative_width_);
    CHECK(input.json_object_states.empty() &&
          input.json_object_state_snapshots.empty())
        << "Prepared Block-Spec does not support JSON grammar";
    CHECK(worker_->adaptive_spec_controller_ == nullptr ||
          !worker_->adaptive_spec_controller_->enabled())
        << "Prepared Block-Spec does not support adaptive speculative "
           "decoding";
    CHECK_EQ(worker_->parallel_args_.dp_size(), 1)
        << "Prepared Block-Spec Worker binding currently requires DP=1";
    CHECK_EQ(worker_->parallel_args_.cp_size(), 1)
        << "Prepared Block-Spec Worker binding currently requires CP=1";
    validate_model_managed_block_tables(input);
    CHECK(input.sampling_params.all_greedy_sample)
        << "Prepared Block-Spec Worker binding supports greedy sampling";
    CHECK(!input.sampling_params.logprobs)
        << "Prepared Block-Spec Worker binding does not support logprobs";
    CHECK_EQ(input.sampling_params.max_top_logprobs, 0)
        << "Prepared Block-Spec Worker binding does not support top logprobs";
    CHECK(input.transfer_kv_infos.empty())
        << "Prepared Block-Spec Worker binding does not support PD transfer";

    SlotResources& resources = mutable_slot(slot.slot_id);
    reset_slot(resources);
    resources.task_kind = plan.task_kind;
    c10::StreamGuard stream_guard =
        worker_->prepare_stream_->set_stream_guard();
    ForwardInput prefill_source;
    const ForwardInput* primary_source = &input;
    if (plan.task_kind == PreparedTaskKind::PREFILL_LIKE &&
        !input.input_params.multi_block_tables.empty()) {
      CHECK_LE(input.positions_host.numel(),
               worker_->options_.max_tokens_per_batch())
          << "Prepared grouped prefill exceeds the fixed SWA slot workspace";
      prefill_source = input;
      specBuilder::build_grouped_prefill_swa_slots_out(
          input, worker_->options_.block_size(), resources.prefill_swa_slots);
      prefill_source.input_params.attention.host.new_cache_slots =
          std::move(resources.prefill_swa_slots);
      primary_source = &prefill_source;
    }
    CHECK(
        stage_primary_input(slot.slot_id, *primary_source, slot.prepared_input))
        << "DFlash ForwardInput is not supported by the fixed Prepared Arena";
    if (primary_source == &prefill_source) {
      resources.prefill_swa_slots =
          std::move(prefill_source.input_params.attention.host.new_cache_slots);
    }
    switch (plan.task_kind) {
      case PreparedTaskKind::PREFILL_LIKE:
        prepare_prefill(slot.prepared_input, slot.slot_id, resources);
        return;
      case PreparedTaskKind::DECODE:
        prepare_decode(input, slot, plan, resources);
        return;
      case PreparedTaskKind::EMPTY:
        prepare_empty(slot.prepared_input, slot.slot_id, resources);
        return;
    }
    LOG(FATAL) << "Unsupported Prepared Block-Spec Task kind";
  }

  void launch_predecessor_patch(
      const block_spec_async::BlockSpecDeviceStepState& predecessor_state,
      ExecutionSlot& slot) override {
    SlotResources& resources = mutable_slot(slot.slot_id);
    CHECK(resources.task_kind == PreparedTaskKind::DECODE);
    patch_continuation_state(
        slot.slot_id,
        predecessor_state,
        slot.prepared_input.input_params.embedding.predecessor_rows,
        resources.continuation);
  }

  void launch_model_invocation(const BlockSpecPreparedInvocation& invocation,
                               ExecutionSlot& slot) override {
    SlotResources& resources = mutable_slot(slot.slot_id);
    switch (invocation.kind) {
      case BlockSpecPreparedInvocationKind::TARGET_PREFILL:
        launch_target_prefill(resources);
        return;
      case BlockSpecPreparedInvocationKind::TASK_CONTINUATION_PATCH:
        launch_decode_geometry_patch(resources);
        return;
      case BlockSpecPreparedInvocationKind::BLOCK_DRAFT:
        launch_block_draft(resources);
        return;
      case BlockSpecPreparedInvocationKind::TARGET_VALIDATE_PATCH:
        launch_target_validate_patch(resources);
        return;
      case BlockSpecPreparedInvocationKind::TARGET_VALIDATE:
        launch_target_validate(resources);
        return;
      case BlockSpecPreparedInvocationKind::REJECTION_SAMPLE:
        launch_rejection_sample(slot, resources);
        return;
      case BlockSpecPreparedInvocationKind::PUBLISH_CONTEXT_KV_STATE:
        return;
      case BlockSpecPreparedInvocationKind::WRITE_CONTEXT_KV:
        launch_context_kv_write(slot, resources);
        return;
      case BlockSpecPreparedInvocationKind::EMPTY_COLLECTIVE:
        launch_empty(slot, resources);
        return;
      case BlockSpecPreparedInvocationKind::DSPARK_MARKOV_SAMPLE:
        launch_dspark_markov_sample(invocation.block_step, resources);
        return;
      case BlockSpecPreparedInvocationKind::DSPARK_TOKEN_BROADCAST:
        launch_dspark_token_broadcast(invocation.block_step, resources);
        return;
    }
    LOG(FATAL) << "Unsupported Prepared Block-Spec invocation";
  }

  StreamEventPtr record_task_stream_event() const override {
    CHECK(worker_->compute_stream_ != nullptr);
    return worker_->compute_stream_->record_event_or_sync();
  }

  c10::Stream task_stream() const override {
    CHECK(worker_->compute_stream_ != nullptr);
    return worker_->compute_stream_->get_stream()->unwrap();
  }

  bool wait_prepare_stream_event(const StreamEventPtr& event) const override {
    CHECK(worker_->prepare_stream_ != nullptr);
    return worker_->prepare_stream_->wait_event(event);
  }

  void consume_ready(ExecutionSlot& slot) override {
    SlotResources& resources = mutable_slot(slot.slot_id);
    if (!slot.output.has_value()) {
      return;
    }
    c10::StreamGuard stream_guard =
        worker_->prepare_stream_->set_stream_guard();
    switch (resources.task_kind) {
      case PreparedTaskKind::PREFILL_LIKE:
        consume_prefill(slot, resources);
        return;
      case PreparedTaskKind::DECODE:
        consume_decode(slot, resources);
        return;
      case PreparedTaskKind::EMPTY:
        clear_all_output_embeddings(*slot.output);
        return;
    }
    LOG(FATAL) << "Unsupported Prepared Block-Spec consume kind";
  }

 private:
  struct LeafInvocation {
    ForwardInput input;
    std::optional<PreparedSlotBinding> binding;
    PreparedModelOutputWorkspace output_workspace;
  };

  struct SlotResources {
    LeafInvocation target_prefill;
    LeafInvocation block_draft;
    LeafInvocation target_validate;
    LeafInvocation empty_target;
    std::optional<ForwardOutput> target_output;
    std::optional<ForwardOutput> draft_output;
    block_spec_async::BlockSpecContinuationState continuation;
    block_spec_async::BlockSpecDecodeInputPatchTarget decode_patch_target;
    block_spec_async::BlockSpecDecodeInputPatchWorkspace decode_patch_workspace;
    dspark_detail::PreparedSamplingWorkspace dspark_sampling_workspace;
    SamplingParameters dspark_sampling_params;
    GreedyTokenIdRejectionWorkspace rejection_workspace;
    torch::Tensor prefill_next_tokens;
    torch::Tensor prefill_selected_hidden;
    torch::Tensor draft_next_tokens;
    torch::Tensor draft_selected_hidden;
    torch::Tensor target_next_tokens;
    torch::Tensor target_context_hidden;
    torch::Tensor continuation_context_hidden;
    DSADeviceGeometryWorkspacePtr dsa_device_geometry_workspace;
    std::vector<torch::Tensor> dsa_group_block_table_storage;
    std::vector<torch::Tensor> active_dsa_group_block_tables;
    torch::Tensor active_cache_slot_block_tables;
    SpeculativePreparedHostInputWorkspace anchor_host_workspace;
    SpeculativePreparedHostInputWorkspace query_host_workspace;
    SpeculativePreparedHostInputWorkspace target_host_workspace;
    std::vector<EmbeddingCache::DecodeState> decode_states;
    std::vector<int32_t> prefill_swa_slots;
    specBuilder::DecodeBuildWorkspace decode_build_workspace;
    torch::Tensor draft_token_ids;
    torch::Tensor dspark_base_logits;
    block_spec_async::CacheSlotMappingMode cache_slot_mapping_mode =
        block_spec_async::CacheSlotMappingMode::LINEAR;
    int64_t batch_size = 0;
    PreparedTaskKind task_kind = PreparedTaskKind::EMPTY;
  };

  SlotResources& mutable_slot(int32_t slot_id) {
    CHECK_GE(slot_id, 0);
    CHECK_LT(static_cast<size_t>(slot_id), slots_.size());
    return slots_[static_cast<size_t>(slot_id)];
  }

  void validate_model_managed_block_tables(const ForwardInput& input) const {
    const std::vector<torch::Tensor>& block_tables =
        input.input_params.multi_block_tables;
    if (block_tables.empty()) {
      return;
    }
    const ModelArgs& target_args = worker_->impl_->context_.get_model_args();
    CHECK(util::is_deepseek_v4_model_type(target_args.model_type()))
        << "Prepared Block-Spec multi-block input requires DeepSeek-V4";
    CHECK_GT(input.input_params.meta.num_sequences, 0);
    const std::vector<int32_t>& compress_ratios = target_args.compress_ratios();
    size_t expected_manager_count = 1;
    if (std::find(compress_ratios.begin(), compress_ratios.end(), 4) !=
        compress_ratios.end()) {
      ++expected_manager_count;
    }
    if (std::find(compress_ratios.begin(), compress_ratios.end(), 128) !=
        compress_ratios.end()) {
      ++expected_manager_count;
    }
    CHECK_EQ(block_tables.size(), expected_manager_count)
        << "Prepared DeepSeek-V4 manager tables must follow the complete "
           "SWA/C4/C128 model-role set";
    for (size_t manager_index = 0; manager_index < block_tables.size();
         ++manager_index) {
      const torch::Tensor& block_table = block_tables[manager_index];
      CHECK(block_table.defined()) << "Prepared multi_block_tables["
                                   << manager_index << "] is undefined";
      CHECK(block_table.device().is_cpu())
          << "Prepared multi_block_tables must remain on Host";
      CHECK_EQ(block_table.scalar_type(), torch::kInt)
          << "Prepared multi_block_tables must use int32";
      CHECK_EQ(block_table.dim(), 2)
          << "Prepared multi_block_tables must be two-dimensional";
      CHECK_GE(block_table.size(0), input.input_params.meta.num_sequences)
          << "Prepared multi_block_tables row count is too small";
      CHECK_GT(block_table.size(1), 0)
          << "Prepared multi_block_tables must have at least one column";
    }
  }

  void stage_cache_slot_block_tables(ForwardInput& staged_input,
                                     SlotResources& resources) {
    resources.cache_slot_mapping_mode =
        block_spec_async::CacheSlotMappingMode::LINEAR;
    resources.active_cache_slot_block_tables = torch::Tensor();
    resources.active_dsa_group_block_tables.clear();
    if (staged_input.input_params.multi_block_tables.empty()) {
      return;
    }

    const std::vector<torch::Tensor>& host_group_tables =
        staged_input.input_params.multi_block_tables;
    CHECK_LE(host_group_tables.size(),
             resources.dsa_group_block_table_storage.size())
        << "Prepared DeepSeek-V4 manager count exceeds fixed capacity";
    for (size_t manager_id = 0; manager_id < host_group_tables.size();
         ++manager_id) {
      const torch::Tensor& host_block_table = host_group_tables[manager_id];
      CHECK(host_block_table.device().is_cpu());
      CHECK_EQ(host_block_table.scalar_type(), torch::kInt);
      CHECK_EQ(host_block_table.dim(), 2);
      CHECK_GE(host_block_table.size(0), resources.batch_size);
      const torch::Tensor& storage =
          resources.dsa_group_block_table_storage[manager_id];
      CHECK_LE(host_block_table.size(1), storage.size(1))
          << "Prepared DeepSeek-V4 manager block table exceeds fixed capacity";
      torch::Tensor source = host_block_table.narrow(
          /*dim=*/0, /*start=*/0, resources.batch_size);
      torch::Tensor destination =
          storage.narrow(/*dim=*/0, /*start=*/0, resources.batch_size)
              .narrow(/*dim=*/1,
                      /*start=*/0,
                      host_block_table.size(1));
      destination.copy_(source, /*non_blocking=*/true);
      resources.active_dsa_group_block_tables.emplace_back(destination);
      const uint64_t transfer_bytes =
          static_cast<uint64_t>(source.numel()) * source.element_size();
      staged_input.prepared_arena_h2d_bytes += transfer_bytes;
      ++staged_input.prepared_arena_h2d_copies;
      COUNTER_ADD(prepared_task_staging_h2d_bytes_total, transfer_bytes);
      COUNTER_INC(prepared_task_staging_h2d_copies_total);
    }
    CHECK(!resources.active_dsa_group_block_tables.empty());
    resources.active_cache_slot_block_tables =
        resources.active_dsa_group_block_tables.front();
    resources.cache_slot_mapping_mode =
        block_spec_async::CacheSlotMappingMode::CIRCULAR;
  }

  void reset_leaf_invocation(LeafInvocation& invocation) {
    invocation.binding.reset();
    invocation.output_workspace = PreparedModelOutputWorkspace();
    invocation.input.metadata_ready_event.reset();
    invocation.input.retained_device_tensors.clear();
  }

  void reset_slot(SlotResources& resources) {
    reset_leaf_invocation(resources.target_prefill);
    reset_leaf_invocation(resources.block_draft);
    reset_leaf_invocation(resources.target_validate);
    reset_leaf_invocation(resources.empty_target);
    resources.target_output.reset();
    resources.draft_output.reset();
    resources.continuation = block_spec_async::BlockSpecContinuationState();
    resources.decode_patch_target =
        block_spec_async::BlockSpecDecodeInputPatchTarget();
    resources.active_cache_slot_block_tables = torch::Tensor();
    resources.active_dsa_group_block_tables.clear();
    resources.cache_slot_mapping_mode =
        block_spec_async::CacheSlotMappingMode::LINEAR;
    resources.dspark_sampling_params = SamplingParameters();
    resources.draft_token_ids = torch::Tensor();
    resources.dspark_base_logits = torch::Tensor();
    if (algorithm_ == BlockSpecAlgorithm::DSPARK) {
      dspark_detail::reset_prepared_sampling_workspace(
          resources.dspark_sampling_workspace);
    }
    resources.batch_size = 0;
  }

  const BlockSpecPreparedInvocation& find_invocation(
      const BlockSpecPreparedTaskPlan& plan,
      BlockSpecPreparedInvocationKind kind) const {
    for (const BlockSpecPreparedInvocation& invocation : plan.invocations) {
      if (invocation.kind == kind) {
        return invocation;
      }
    }
    LOG(FATAL) << "Prepared DFlash plan is missing an invocation";
    return plan.invocations.front();
  }

  void bind_leaf(LLMWorkerImpl& leaf,
                 int32_t slot_id,
                 LeafInvocation& invocation) {
    if (leaf.prepared_graph_enabled()) {
      invocation.binding = leaf.bind_prepared_task(slot_id, invocation.input);
    } else {
      invocation.binding.reset();
    }
    invocation.input.metadata_ready_event =
        worker_->prepare_stream_->record_event_or_sync();
    CHECK(invocation.input.metadata_ready_event != nullptr)
        << "Failed to record Prepared DFlash input-ready event";
  }

  void stage_leaf(LLMWorkerImpl& leaf,
                  ForwardInput& source,
                  ExecutionSlot& slot,
                  SlotResources& resources,
                  LeafInvocation& destination) {
    if (leaf.prepared_graph_enabled()) {
      leaf.prepare_prepared_graph_input(slot.slot_id, source);
    }
    CHECK(stage_invocation_input(slot.slot_id, source, destination.input))
        << "DFlash model invocation is not supported by the fixed Arena";
    if (resources.cache_slot_mapping_mode ==
        block_spec_async::CacheSlotMappingMode::CIRCULAR) {
      ModelInputParams& params = destination.input.input_params;
      params.device_multi_block_tables.clear();
      params.device_multi_block_tables.reserve(
          resources.active_dsa_group_block_tables.size());
      for (const torch::Tensor& block_table :
           resources.active_dsa_group_block_tables) {
        params.device_multi_block_tables.emplace_back(block_table);
      }
      params.dsa_device_geometry_authoritative = true;
      params.dsa_device_geometry_kv_headroom =
          static_cast<int32_t>(accepted_token_capacity_);
      CHECK(resources.dsa_device_geometry_workspace != nullptr);
      params.dsa_device_geometry_workspace =
          resources.dsa_device_geometry_workspace;
    }
    bind_leaf(leaf, slot.slot_id, destination);
  }

  std::optional<ForwardOutput> launch_leaf(LLMWorkerImpl& leaf,
                                           const LeafInvocation& invocation) {
    return leaf.execute_prepared_on_stream(invocation.input,
                                           *worker_->compute_stream_,
                                           invocation.binding,
                                           &invocation.output_workspace);
  }

  void prepare_prefill(const ForwardInput& staged_input,
                       int32_t slot_id,
                       SlotResources& resources) {
    resources.batch_size = staged_input.input_params.meta.num_sequences;
    CHECK_GT(resources.batch_size, 0);
    CHECK_LE(resources.batch_size, max_rows_);
    resources.target_prefill.input = staged_input;
    resources.target_prefill.input.sampling_params.return_probs = false;
    if (staged_input.sampling_params.selected_token_idxes.defined()) {
      const int64_t selected_count =
          staged_input.sampling_params.selected_token_idxes.numel();
      CHECK_EQ(selected_count, resources.batch_size)
          << "Prepared Block-Spec Target Prefill requires one selected token "
             "per sequence";
      CHECK_LE(selected_count, max_rows_);
      resources.target_prefill.output_workspace.next_tokens =
          resources.prefill_next_tokens.narrow(
              /*dim=*/0, /*start=*/0, selected_count);
      resources.target_prefill.output_workspace.selected_embeddings =
          resources.prefill_selected_hidden.narrow(
              /*dim=*/0, /*start=*/0, selected_count);
    }
    bind_leaf(*worker_->impl_, slot_id, resources.target_prefill);
  }

  void prepare_decode(const ForwardInput& input,
                      ExecutionSlot& slot,
                      const BlockSpecPreparedTaskPlan& plan,
                      SlotResources& resources) {
    CHECK_GE(worker_->mask_token_id_, 0)
        << "Prepared DFlash Backend was created before the draft model loaded";
    resources.batch_size = input.input_params.meta.num_sequences;
    CHECK_GT(resources.batch_size, 0);
    CHECK_LE(resources.batch_size, max_rows_);
    stage_cache_slot_block_tables(slot.prepared_input, resources);

    ForwardInput updated_input;
    const ForwardInput* working_input = &input;
    if (!slot.predecessor_slot_id.has_value()) {
      updated_input = input;
      worker_->embedding_cache_->read_decode_states_out(
          input.input_params.embedding.embedding_ids,
          input.input_params.embedding.request_ids,
          resources.decode_states);
      worker_->update_decode_step_input(updated_input,
                                        resources.decode_states,
                                        &resources.anchor_host_workspace);
      working_input = &updated_input;
    }

    ForwardInput query_source;
    worker_->prepare_query_inputs(*working_input,
                                  query_source,
                                  /*stage_sampling_on_host=*/true,
                                  &resources.query_host_workspace,
                                  &resources.decode_build_workspace);
    query_source.sampling_params.return_probs = false;
    if (algorithm_ == BlockSpecAlgorithm::DSPARK) {
      query_source.skip_sampling_for_logits_only = true;
      query_source.return_selected_hidden = false;
    }
    CHECK(find_invocation(plan, BlockSpecPreparedInvocationKind::BLOCK_DRAFT)
              .input_partition == 1);
    stage_leaf(*worker_->draft_impl_,
               query_source,
               slot,
               resources,
               resources.block_draft);
    specBuilder::reclaim_decode_build_workspace(
        query_source.input_params, resources.decode_build_workspace);
    const int64_t draft_row_count = resources.batch_size * speculative_width_;
    if (algorithm_ == BlockSpecAlgorithm::DFLASH) {
      resources.block_draft.output_workspace.next_tokens =
          resources.draft_next_tokens.narrow(
              /*dim=*/0, /*start=*/0, draft_row_count);
      resources.block_draft.output_workspace.selected_embeddings =
          resources.draft_selected_hidden.narrow(
              /*dim=*/0, /*start=*/0, draft_row_count);
    }

    ForwardInput target_source;
    worker_->prepare_validate_inputs(*working_input,
                                     target_source,
                                     /*stage_sampling_on_host=*/true,
                                     &resources.target_host_workspace,
                                     &resources.decode_build_workspace);
    target_source.sampling_params.return_probs = false;
    CHECK(
        find_invocation(plan, BlockSpecPreparedInvocationKind::TARGET_VALIDATE)
            .input_partition == 2);
    stage_leaf(*worker_->impl_,
               target_source,
               slot,
               resources,
               resources.target_validate);
    specBuilder::reclaim_decode_build_workspace(
        target_source.input_params, resources.decode_build_workspace);
    const int64_t target_row_count =
        resources.batch_size * accepted_token_capacity_;
    resources.target_validate.output_workspace.next_tokens =
        resources.target_next_tokens.narrow(
            /*dim=*/0, /*start=*/0, target_row_count);
    resources.target_validate.output_workspace.selected_embeddings =
        resources.target_context_hidden.narrow(
            /*dim=*/0, /*start=*/0, target_row_count);

    ForwardInput& query_input = resources.block_draft.input;
    ForwardInput& target_input = resources.target_validate.input;
    const int64_t query_row_count = resources.batch_size * query_width_;
    CHECK_EQ(query_input.token_ids.numel(), query_row_count);
    CHECK_EQ(query_input.positions.numel(), query_row_count);
    CHECK_EQ(target_input.token_ids.numel(), target_row_count);
    CHECK_EQ(target_input.positions.numel(), target_row_count);
    torch::Tensor query_token_rows =
        query_input.token_ids.view({resources.batch_size, query_width_});
    torch::Tensor query_position_rows =
        query_input.positions.view({resources.batch_size, query_width_});
    torch::Tensor target_token_rows = target_input.token_ids.view(
        {resources.batch_size, accepted_token_capacity_});
    torch::Tensor target_position_rows = target_input.positions.view(
        {resources.batch_size, accepted_token_capacity_});
    resources.continuation = block_spec_async::BlockSpecContinuationState{
        query_token_rows.select(/*dim=*/1, /*index=*/0),
        resources.continuation_context_hidden.narrow(
            /*dim=*/0, /*start=*/0, resources.batch_size),
        query_position_rows.select(/*dim=*/1, /*index=*/0)};
    resources.continuation.anchor_context_hidden.zero_();

    if (algorithm_ == BlockSpecAlgorithm::DSPARK) {
      resources.draft_token_ids =
          resources.dspark_sampling_workspace.token_ids.narrow(
              /*dim=*/0, /*start=*/0, resources.batch_size);
      resources.dspark_sampling_params = slot.prepared_input.sampling_params;
      resources.dspark_sampling_params.selected_token_idxes = torch::Tensor();
      resources.dspark_sampling_params.sample_idxes = torch::Tensor();
      resources.dspark_sampling_params.return_probs = false;
      resources.dspark_sampling_params.logprobs = false;
      resources.dspark_sampling_params.max_top_logprobs = 0;
      resources.dspark_sampling_params.use_beam_search = false;
    }

    const torch::Tensor& source_block_tables =
        resources.cache_slot_mapping_mode ==
                block_spec_async::CacheSlotMappingMode::CIRCULAR
            ? resources.active_cache_slot_block_tables
            : slot.prepared_input.input_params.attention.device.block_tables;
    CHECK(source_block_tables.defined());
    CHECK_EQ(source_block_tables.dim(), 2);
    CHECK_EQ(source_block_tables.size(0), resources.batch_size);
    const dflash_detail::PreparedDecodeCacheSlotViews cache_slot_views =
        dflash_detail::view_prepared_decode_cache_slots(
            query_input.input_params.attention.device.new_cache_slots,
            target_input.input_params.attention.device.new_cache_slots,
            resources.batch_size,
            worker_->options_.num_speculative_tokens(),
            worker_->sample_from_anchor());
    resources.decode_patch_target =
        block_spec_async::BlockSpecDecodeInputPatchTarget{
            query_token_rows,
            query_position_rows,
            query_input.input_params.attention.device.kv_seq_lens,
            cache_slot_views.query,
            target_token_rows,
            target_position_rows,
            target_input.input_params.attention.device.kv_seq_lens,
            cache_slot_views.target,
            source_block_tables,
            resources.cache_slot_mapping_mode};
  }

  void prepare_empty(const ForwardInput& staged_input,
                     int32_t slot_id,
                     SlotResources& resources) {
    resources.empty_target.input = staged_input;
    bind_leaf(*worker_->impl_, slot_id, resources.empty_target);
  }

  void launch_target_prefill(SlotResources& resources) {
    resources.target_output =
        launch_leaf(*worker_->impl_, resources.target_prefill);
    CHECK(resources.target_output.has_value())
        << "Prepared DFlash Target Prefill returned no output";
    release_prepared_model_intermediates(*resources.target_output);
  }

  void launch_decode_geometry_patch(SlotResources& resources) {
    CHECK(resources.task_kind == PreparedTaskKind::DECODE);
    CHECK(resources.block_draft.input.metadata_ready_event != nullptr);
    CHECK(worker_->compute_stream_->wait_event(
        resources.block_draft.input.metadata_ready_event))
        << "Prepared Block-Spec geometry patch failed to wait for input";
    block_spec_async::patch_block_spec_decode_input_geometry(
        resources.continuation,
        worker_->options_.block_size(),
        resources.decode_patch_target,
        resources.decode_patch_workspace);
  }

  void launch_block_draft(SlotResources& resources) {
    resources.draft_output =
        launch_leaf(*worker_->draft_impl_, resources.block_draft);
    CHECK(resources.draft_output.has_value())
        << "Prepared Block-Spec Draft returned no output";
    if (algorithm_ == BlockSpecAlgorithm::DSPARK) {
      CHECK(resources.draft_output->logits.defined())
          << "Prepared DSpark Draft must return logits";
      const int64_t expected_rows = resources.batch_size * speculative_width_;
      CHECK_EQ(resources.draft_output->logits.size(0), expected_rows);
      CHECK_EQ(resources.draft_output->logits.size(1),
               worker_->prepared_dspark_vocab_size());
      resources.dspark_base_logits = resources.draft_output->logits.view(
          {resources.batch_size,
           speculative_width_,
           resources.draft_output->logits.size(1)});
      release_prepared_model_intermediates(*resources.draft_output);
      clear_all_output_embeddings(*resources.draft_output);
      return;
    }

    SampleOutput& draft_sample = resources.draft_output->sample_output;
    const int64_t expected_draft_token_count =
        resources.batch_size * speculative_width_;
    dflash_detail::check_fixed_prepared_output_binding(
        draft_sample.next_tokens,
        resources.block_draft.output_workspace.next_tokens,
        worker_->device_.unwrap(),
        torch::kLong,
        expected_draft_token_count,
        "Draft token output");
    worker_->maybe_broadcast_spec_tokens(draft_sample.next_tokens);
    dflash_detail::check_fixed_prepared_output_binding(
        draft_sample.next_tokens,
        resources.block_draft.output_workspace.next_tokens,
        worker_->device_.unwrap(),
        torch::kLong,
        expected_draft_token_count,
        "Draft token output after broadcast");
    resources.draft_token_ids = draft_sample.next_tokens.view(
        {resources.batch_size, speculative_width_});
    release_prepared_model_intermediates(*resources.draft_output);
    clear_all_output_embeddings(*resources.draft_output);
  }

  void launch_dspark_markov_sample(int32_t block_step,
                                   SlotResources& resources) {
    CHECK(algorithm_ == BlockSpecAlgorithm::DSPARK);
    CHECK(resources.dspark_base_logits.defined());
    worker_->launch_prepared_dspark_markov_sample(
        resources.dspark_base_logits,
        resources.continuation.anchor_tokens,
        resources.dspark_sampling_params,
        resources.batch_size,
        block_step,
        resources.dspark_sampling_workspace);
  }

  void launch_dspark_token_broadcast(int32_t block_step,
                                     SlotResources& resources) {
    CHECK(algorithm_ == BlockSpecAlgorithm::DSPARK);
    worker_->launch_prepared_dspark_token_broadcast(
        resources.dspark_sampling_params,
        resources.batch_size,
        block_step,
        resources.dspark_sampling_workspace);
    if (block_step == speculative_width_ - 1) {
      resources.dspark_base_logits = torch::Tensor();
      resources.draft_output.reset();
    }
  }

  void launch_target_validate_patch(SlotResources& resources) {
    CHECK(resources.draft_token_ids.defined());
    block_spec_async::patch_block_spec_target_token_ids(
        resources.draft_token_ids,
        resources.decode_patch_target.target_token_ids);
  }

  void launch_target_validate(SlotResources& resources) {
    resources.target_output =
        launch_leaf(*worker_->impl_, resources.target_validate);
    CHECK(resources.target_output.has_value())
        << "Prepared DFlash Target Validate returned no output";
    release_prepared_model_intermediates(*resources.target_output);
  }

  void launch_rejection_sample(ExecutionSlot& slot, SlotResources& resources) {
    CHECK(resources.target_output.has_value());
    CHECK(resources.draft_token_ids.defined());
    const SampleOutput& target_sample = resources.target_output->sample_output;
    CHECK(!target_sample.logprobs.defined())
        << "Prepared DFlash greedy rejection does not support logprobs";
    const int64_t target_token_count =
        resources.batch_size * accepted_token_capacity_;
    dflash_detail::check_fixed_prepared_output_binding(
        target_sample.next_tokens,
        resources.target_validate.output_workspace.next_tokens,
        worker_->device_.unwrap(),
        torch::kLong,
        target_token_count,
        "Target token output");
    dflash_detail::check_fixed_prepared_output_binding(
        target_sample.embeddings,
        resources.target_validate.output_workspace.selected_embeddings,
        worker_->device_.unwrap(),
        worker_->dtype_,
        target_token_count * context_hidden_size_,
        "Target context hidden output");
    torch::Tensor target_token_ids = target_sample.next_tokens.view(
        {resources.batch_size, accepted_token_capacity_});
    torch::Tensor target_draft_token_ids = target_token_ids.narrow(
        /*dim=*/1, /*start=*/0, speculative_width_);
    torch::Tensor bonus_token_ids = target_token_ids.narrow(
        /*dim=*/1, /*start=*/speculative_width_, /*length=*/1);
    GreedyTokenIdRejectionWorkspace rejection_workspace{
        resources.rejection_workspace.candidate_token_ids.narrow(
            /*dim=*/0, /*start=*/0, resources.batch_size),
        resources.rejection_workspace.draft_matches.narrow(
            /*dim=*/0, /*start=*/0, resources.batch_size),
        resources.rejection_workspace.accepted_prefix_mask.narrow(
            /*dim=*/0, /*start=*/0, resources.batch_size),
        resources.rejection_workspace.rejected_token_ids.narrow(
            /*dim=*/0, /*start=*/0, resources.batch_size),
        resources.rejection_workspace.masked_accepted_token_ids.narrow(
            /*dim=*/0, /*start=*/0, resources.batch_size)};
    RejectionSampler::greedy_masked_sample_from_token_ids_out(
        resources.draft_token_ids,
        target_draft_token_ids,
        bonus_token_ids,
        rejection_workspace);

    SampleOutput accepted_output;
    accepted_output.next_tokens = rejection_workspace.masked_accepted_token_ids;
    accepted_output.embeddings = target_sample.embeddings.view(
        {resources.batch_size, accepted_token_capacity_, context_hidden_size_});
    bind_publish_source(
        slot.slot_id,
        BlockSpecPreparedPublishSource{
            accepted_output.next_tokens,
            accepted_output.embeddings,
            resources.continuation.base_positions,
            resources.decode_patch_target.source_block_tables,
            resources.decode_patch_target.cache_slot_mapping_mode});
    resources.target_output->sample_output = std::move(accepted_output);
    slot.output = std::move(resources.target_output);
  }

  void launch_context_kv_write(ExecutionSlot& slot, SlotResources& resources) {
    if (resources.task_kind == PreparedTaskKind::PREFILL_LIKE) {
      CHECK(resources.target_output.has_value());
      const torch::Tensor& context_hidden =
          resources.target_output->sample_output.embeddings;
      if (context_hidden.defined()) {
        worker_->write_context_kv(slot.prepared_input,
                                  context_hidden,
                                  resources.target_prefill.input.positions,
                                  resources.target_prefill.input.input_params
                                      .attention.device.new_cache_slots,
                                  /*synchronize_completion=*/false);
      }
      slot.output = std::move(resources.target_output);
      return;
    }

    CHECK(resources.task_kind == PreparedTaskKind::DECODE);
    const block_spec_async::BlockSpecDeviceStepState& state =
        device_step_state(slot.slot_id);
    torch::Tensor context_hidden =
        state.context_hidden
            .narrow(/*dim=*/0, /*start=*/0, resources.batch_size)
            .view({resources.batch_size * accepted_token_capacity_,
                   context_hidden_size_});
    torch::Tensor context_positions =
        state.context_positions
            .narrow(/*dim=*/0, /*start=*/0, resources.batch_size)
            .view({resources.batch_size * accepted_token_capacity_});
    torch::Tensor context_cache_slots =
        state.context_cache_slots
            .narrow(/*dim=*/0, /*start=*/0, resources.batch_size)
            .view({resources.batch_size * accepted_token_capacity_});
    worker_->write_context_kv(slot.prepared_input,
                              context_hidden,
                              context_positions,
                              context_cache_slots,
                              /*synchronize_completion=*/false);
  }

  void launch_empty(ExecutionSlot& slot, SlotResources& resources) {
    std::optional<ForwardOutput> output =
        launch_leaf(*worker_->impl_, resources.empty_target);
    if (output.has_value()) {
      release_prepared_model_intermediates(*output);
      clear_all_output_embeddings(*output);
    }
    slot.output = std::move(output);
  }

  void consume_prefill(ExecutionSlot& slot, SlotResources& resources) {
    SampleOutput& sample = slot.output->sample_output;
    const ForwardInput& input = resources.target_prefill.input;
    if (!input.sampling_params.selected_token_idxes.defined()) {
      clear_all_output_embeddings(*slot.output);
      return;
    }
    CHECK(worker_->embedding_cache_ != nullptr);
    worker_->embedding_cache_->write_prefill_target_context(
        input.input_params.embedding.embedding_ids,
        input.input_params.embedding.request_ids,
        sample.next_tokens,
        sample.embeddings,
        input.sampling_params.selected_token_idxes);

    torch::Tensor bootstrap_embeddings = sample.selected_embeddings.defined()
                                             ? sample.selected_embeddings
                                             : sample.embeddings;
    const int64_t sequence_count =
        static_cast<int64_t>(input.input_params.embedding.embedding_ids.size());
    CHECK_EQ(bootstrap_embeddings.size(0), sequence_count)
        << "Prepared Block-Spec Target Prefill must publish one fixed selected "
           "embedding per sequence";
    sample.embeddings = bootstrap_embeddings.detach();
    clear_selected_embeddings(*slot.output);
    worker_->prepare_stream_->synchronize();
  }

  void consume_decode(ExecutionSlot& slot, SlotResources& resources) {
    SampleOutput& sample = slot.output->sample_output;
    torch::Tensor accepted_tokens =
        safe_to(sample.next_tokens, torch::kCPU).contiguous();
    if (accepted_tokens.scalar_type() != torch::kLong) {
      accepted_tokens = accepted_tokens.to(torch::kLong);
    }
    sample.next_tokens = accepted_tokens;
    const ForwardInput& input = resources.target_validate.input;
    trace_block_spec_step_state_from_target_input(
        worker_->options_.speculative_algorithm(),
        input,
        accepted_tokens,
        resources.continuation.base_positions,
        accepted_token_capacity_);
    worker_->record_validate_metrics(
        sample, /*per_seq_val_tokens=*/std::vector<int32_t>{});
    CHECK(worker_->embedding_cache_ != nullptr);
    worker_->embedding_cache_->write_target_context(
        input.input_params.embedding.embedding_ids,
        input.input_params.embedding.request_ids,
        accepted_tokens,
        sample.embeddings,
        speculative_width_);
    clear_all_output_embeddings(*slot.output);
    worker_->prepare_stream_->synchronize();
  }

  DFlashWorkerImpl* worker_ = nullptr;
  BlockSpecAlgorithm algorithm_ = BlockSpecAlgorithm::DFLASH;
  int64_t max_rows_ = 0;
  int64_t speculative_width_ = 0;
  int64_t query_width_ = 0;
  int64_t accepted_token_capacity_ = 0;
  int64_t context_hidden_size_ = 0;
  int64_t draft_hidden_size_ = 0;
  int64_t max_swa_block_table_width_ = 0;
  static constexpr int32_t kMaxDeepseekV4CacheManagers = 3;
  std::vector<SlotResources> slots_;
};

std::unique_ptr<BlockSpecPreparedTaskBackend>
DFlashWorkerImpl::create_prepared_task_backend(
    uint64_t input_arena_capacity_bytes,
    int32_t slot_count) {
  CHECK(options_.speculative_algorithm() == "DFlash" ||
        options_.speculative_algorithm() == "DSpark")
      << "Prepared Block-Spec Backend requires DFlash or DSpark";
  CHECK(impl_ != nullptr && draft_impl_ != nullptr);
  CHECK(embedding_cache_ != nullptr)
      << "Prepared DFlash Backend requires allocated KV/embedding caches";
  CHECK_GE(mask_token_id_, 0);
  CHECK_GT(expected_context_hidden_size_, 0);
  BlockSpecPreparedTaskBufferConfig config;
  config.device = device_.unwrap();
  config.input_arena_capacity_bytes = input_arena_capacity_bytes;
  config.slot_count = slot_count;
  config.block_size = options_.block_size();
  config.max_rows = options_.max_seqs_per_batch();
  config.accepted_token_capacity = options_.num_speculative_tokens() + 1;
  config.context_hidden_size = expected_context_hidden_size_;
  config.token_dtype = torch::kLong;
  config.hidden_dtype = dtype_;
  config.position_dtype = torch::kInt;
  config.cache_slot_dtype = torch::kInt;
  return std::make_unique<PreparedTaskBackend>(this, config);
}

bool DFlashWorkerImpl::init_model(const std::string& model_weights_path,
                                  int32_t random_seed,
                                  MasterStatus master_status) {
  // DFlash draft attends each block non-causally, which the shared QWen3 model
  // only wires up on the chunked-prefill mask path. Without it the draft falls
  // back to a causal mask and proposal quality silently degrades, so require
  // the flag rather than accept a misconfigured run.
  CHECK(::xllm::SchedulerConfig::get_instance().enable_chunked_prefill())
      << "Block-diffusion speculative decoding requires "
         "--enable_chunked_prefill=true.";
  bool result = true;
  const bool loading_target =
      impl_->get_status() == WorkerImpl::Status::UNINITIALIZED;
  if (loading_target) {
    result = SpeculativeWorkerImpl::init_model(
        model_weights_path, random_seed, master_status);
  } else {
    CHECK_EQ(draft_impl_->get_status(), WorkerImpl::Status::UNINITIALIZED);
    // Draft config's use_sliding_window / sliding_window is intentionally
    // ignored: xLLM NPU FIA hard-codes pre_tokens=INT_MAX, next_tokens=0
    // and never plumbs sliding_window into the aclnn call. Attended kv_len
    // on the draft path is bounded by the target sequence length, not by
    // block_size, so enforcing a block_size-vs-window relationship here
    // would compare the wrong quantities.
    result = draft_impl_->WorkerImpl::init_model(
        model_weights_path, random_seed, master_status);
  }

  if (impl_->get_status() == WorkerImpl::Status::LOADED) {
    context_ = impl_->context_;
  }

  if (draft_impl_->get_status() == WorkerImpl::Status::LOADED) {
    const ModelArgs& draft_args = draft_impl_->context_.get_model_args();
    // DeepSeek-V4 DSpark checkpoints carry trained mtp.0.embed /
    // mtp.<last>.head tensors (with the draft-side QuaRot transform); replacing
    // them after load destroys proposal quality.
    const bool uses_own_head_and_embedding =
        util::is_deepseek_v4_dspark_model_type(draft_args.model_type());
    if (uses_own_head_and_embedding) {
      CHECK_EQ(parallel_args_.cp_size(), 1)
          << "DeepSeek-V4 DSpark does not support context parallelism yet.";
      LOG(INFO) << "Configured DeepSeek-V4 DSpark draft block size: "
                << draft_args.dspark_block_size();
      LOG(INFO) << "Configured DeepSeek-V4 DSpark SAS mode: "
                << (draft_args.dspark_use_native_sas()
                        ? "native explicit indices"
                        : "CANN 9.0-compatible q_len=1 fallback");
#if defined(USE_NPU)
      if (draft_args.dspark_use_native_sas()) {
        LOG(WARNING)
            << "Native DeepSeek-V4 DSpark SAS requires an operator that "
               "accepts non-empty ori_sparse_indices and ori_win_left="
            << layer::deepseek_v4_ori_window_left(
                   draft_args.window_size(),
                   draft_args.dspark_block_size(),
                   /*use_native_dspark_sas=*/true)
            << ".";
      }
#endif
      // Keep the trained mtp.0.embed and mtp.<last>.head modules loaded by
      // DeepseekV4DSparkForCausalLMImpl. Sharing the target modules here makes
      // the draft backbone/Markov head project through the wrong vocabulary
      // basis and reduces acceptance to near-random levels.
    } else {
#if defined(USE_NPU)
      auto head = impl_->get_npu_lm_head();
      draft_impl_->set_npu_lm_head(head);
      auto word_embedding = impl_->get_npu_word_embedding();
      draft_impl_->set_npu_word_embedding(word_embedding);
#else
      auto head = impl_->get_lm_head();
      draft_impl_->set_lm_head(head);
      auto word_embedding = impl_->get_word_embedding();
      draft_impl_->set_word_embedding(word_embedding);
#endif
    }

    JsonReader reader;
    const std::string config_path = model_weights_path + "/config.json";
    CHECK(reader.parse(config_path))
        << "Failed to parse block-diffusion draft config: " << config_path;
    mask_token_id_ = reader.value_or<int32_t>({"dflash_config.mask_token_id",
                                               "mask_token_id",
                                               "dspark_noise_token_id"},
                                              /*default=*/-1);
    CHECK_GE(mask_token_id_, 0)
        << "Block-diffusion draft config requires mask_token_id, "
           "dflash_config.mask_token_id, or dspark_noise_token_id.";

    const int64_t draft_vocab_size = draft_args.vocab_size();
    CHECK_GT(draft_vocab_size, 0)
        << "Block-diffusion draft vocab_size must be set.";
    CHECK_GE(mask_token_id_, 0)
        << "Block-diffusion mask_token_id (" << mask_token_id_
        << ") must be a valid embedding index (>= 0).";
    CHECK_LT(mask_token_id_, draft_vocab_size)
        << "Block-diffusion mask_token_id (" << mask_token_id_
        << ") must be < draft vocab_size (" << draft_vocab_size << ").";
    // Context hidden comes from the target.
    const ModelArgs& target_args = impl_->context_.get_model_args();
    const int64_t num_target_layers =
        static_cast<int64_t>(target_args.layers_to_capture().size());
    CHECK_GT(num_target_layers, 0)
        << "Block-diffusion draft config requires dspark_target_layer_ids, "
           "target_layer_ids, or dflash_config.target_layer_ids.";
    expected_context_hidden_size_ =
        static_cast<int64_t>(target_args.hidden_size()) * num_target_layers;
    draft_sas_mode_ = dflash_detail::classify_dspark_sas_mode(
        draft_args, sample_from_anchor());
  }
  return result;
}

std::tuple<int64_t, int64_t> DFlashWorkerImpl::estimate_kv_cache_capacity() {
  CHECK(impl_ != nullptr);
  CHECK(draft_impl_ != nullptr);
  return estimate_kv_cache_capacity_with_draft(
      *draft_impl_, target_options(options_), draft_options(options_));
}

bool DFlashWorkerImpl::allocate_kv_cache(const KVCacheShape& kv_cache_shape) {
  const int64_t num_blocks = kv_cache_shape.key_cache_shape()[0];
  embedding_cache_ = std::make_shared<EmbeddingCache>(num_blocks);

  bool target_allocated = true;
  const WorkerImpl::Status target_status = impl_->get_status();
  if (target_status == WorkerImpl::Status::LOADED) {
    target_allocated = impl_->allocate_kv_cache(kv_cache_shape);
  } else {
    CHECK_EQ(target_status, WorkerImpl::Status::READY);
  }

  bool draft_allocated = true;
  const WorkerImpl::Status draft_status = draft_impl_->get_status();
  if (draft_status == WorkerImpl::Status::LOADED) {
    draft_allocated = draft_impl_->allocate_kv_cache(kv_cache_shape);
  } else {
    CHECK_EQ(draft_status, WorkerImpl::Status::READY);
  }

  return target_allocated && draft_allocated;
}

#if defined(USE_NPU) || defined(USE_MLU)
bool DFlashWorkerImpl::allocate_kv_cache_with_transfer(
    const KVCacheShape& kv_cache_shape) {
  const int64_t num_blocks = kv_cache_shape.key_cache_shape()[0];

  if (kv_cache_transfer_ == nullptr) {
    kv_cache_transfer_ = std::make_shared<MooncakeKVCacheTransferDefault>(
        device_.index(),
        options_.transfer_listen_port(),
        device_,
        context_.get_model_args().model_type());

    const int32_t device_id = device_.index();
    kv_cache_transfer_->initialize(device_id);
  }

  bool target_allocated = true;
  const WorkerImpl::Status target_status = impl_->get_status();
  if (target_status == WorkerImpl::Status::LOADED) {
    target_allocated = impl_->allocate_kv_cache_with_transfer(
        kv_cache_transfer_, kv_cache_shape);
  } else {
    CHECK_EQ(target_status, WorkerImpl::Status::READY);
  }

  bool draft_allocated = true;
  const WorkerImpl::Status draft_status = draft_impl_->get_status();
  if (draft_status == WorkerImpl::Status::LOADED) {
    draft_allocated = draft_impl_->allocate_kv_cache_with_transfer(
        kv_cache_transfer_, kv_cache_shape);
  } else {
    CHECK_EQ(draft_status, WorkerImpl::Status::READY);
  }

  embedding_cache_ = std::make_shared<EmbeddingCache>(num_blocks);
  return target_allocated && draft_allocated;
}
#endif

ForwardInput DFlashWorkerImpl::update_input_by_last_step_output(
    ForwardInput& inputs) {
  return inputs;
}

std::optional<ForwardOutput> DFlashWorkerImpl::step_empty(
    const ForwardInput& input) {
  if (!input.input_params.meta.batch_forward_type.is_decode()) {
    std::optional<ForwardOutput> output =
        run_llm_no_sync_impl(*impl_, input, *prepare_stream_, *compute_stream_);
    // Active prefill ranks write the draft context KV without a draft forward.
    // Keep idle ranks symmetric: a draft MoE forward here would enter EP
    // collectives that active ranks never join and deadlock the whole group.
    // Sync the target forward before its staged input can be reused.
    compute_stream_->synchronize();
    if (output.has_value()) {
      clear_all_output_embeddings(output.value());
    }
    return output;
  }

  // Mirror prepare_query_inputs' metadata geometry: DSV4 DSpark represents
  // every block row as a q_len=1 sequence, while Qwen/DFlash keeps one
  // query_width-wide sequence.
  const int32_t draft_width = dflash_detail::decode_draft_width(
      options_.num_speculative_tokens(), sample_from_anchor());
  const bool use_block_parallel_rows = draft_use_block_parallel_rows();
  ForwardInput query_input = input;
  dflash_detail::invalidate_draft_model_geometry(query_input.input_params);
  query_input.input_params.meta.batch_forward_type = draft_batch_forward_type();
  query_input.input_params.meta.q_max_seq_len =
      use_block_parallel_rows ? 1 : draft_width;
  if (use_block_parallel_rows) {
    expand_block_parallel_sequence_rows(query_input.input_params, draft_width);
  }
  scale_speculative_parallel_token_counts(query_input.input_params,
                                          draft_width);
  // Warmup only: prime the draft; its output is unused. Keep it alive until the
  // sync below so the no-sync draft input is not freed while the target forward
  // launched next can reuse the buffer.
  std::optional<ForwardOutput> draft_output = run_llm_no_sync_impl(
      *draft_impl_, query_input, *prepare_stream_, *compute_stream_);

  ForwardInput validate_input = input;
  // DSpark's N-wide draft geometry must be rescaled to (N+1) for the target's
  // anchor + drafts forward.
  scale_speculative_parallel_token_counts(
      validate_input.input_params, options_.num_speculative_tokens() + 1);
  // Deadlock-safety under DP: when all ranks decode but this rank's shard is
  // empty (fake input), busy peers reach run_validate and allgather their
  // pruned validate counts before the target forward. This idle rank runs the
  // same target forward and must join that allgather in lockstep, contributing
  // its own uniform (unpruned) count. No-op unless adaptive + dp_size>1.
  sync_dp_global_token_nums_for_idle_rank(validate_input.input_params);
  ForwardOutput output =
      run_llm_no_sync_impl(
          *impl_, validate_input, *prepare_stream_, *compute_stream_)
          .value();
  // See above: sync the no-sync draft and target forwards before returning.
  compute_stream_->synchronize();
  clear_all_output_embeddings(output);
  return output;
}

std::optional<ForwardOutput> DFlashWorkerImpl::step_prefill(
    const ForwardInput& input) {
  Timer timer;
  ForwardInput processed_target_input;
  ForwardOutput output = run_llm_no_sync_impl(*impl_,
                                              input,
                                              *prepare_stream_,
                                              *compute_stream_,
                                              &processed_target_input)
                             .value();
  COUNTER_ADD(speculative_execution_latency_seconds_target,
              timer.elapsed_seconds());

  const torch::Tensor& embeddings = output.sample_output.embeddings;
  if (embeddings.defined()) {
    CHECK(processed_target_input.positions_host.defined())
        << "DFlash prefill requires processed positions_host.";
    Slice<int32_t> positions = {
        processed_target_input.positions_host.data_ptr<int32_t>(),
        static_cast<size_t>(processed_target_input.positions_host.numel())};
    CHECK_EQ(positions.size(), static_cast<size_t>(embeddings.size(0)))
        << "DFlash prefill hidden/position count mismatch.";
    torch::Tensor context_cache_slots =
        processed_target_input.input_params.attention.device.new_cache_slots;
    if (!processed_target_input.input_params.multi_block_tables.empty()) {
      const std::vector<int32_t> grouped_swa_slots =
          specBuilder::build_grouped_prefill_swa_slots(processed_target_input,
                                                       options_.block_size());
      c10::StreamGuard stream_guard = compute_stream_->set_stream_guard();
      context_cache_slots = cpu_int_vec_to_device(grouped_swa_slots, device_);
    }
    CHECK(context_cache_slots.defined())
        << "DFlash prefill requires context cache slots.";
    CHECK_EQ(static_cast<size_t>(context_cache_slots.numel()), positions.size())
        << "DFlash prefill hidden/cache slot count mismatch.";

    timer.reset();
    write_context_kv(processed_target_input,
                     embeddings,
                     processed_target_input.positions,
                     context_cache_slots);
    COUNTER_ADD(speculative_execution_latency_seconds_draft,
                timer.elapsed_seconds());
  }

  if (input.sampling_params.selected_token_idxes.defined()) {
    embedding_cache_->write_prefill_target_context(
        input.input_params.embedding.embedding_ids,
        input.input_params.embedding.request_ids,
        output.sample_output.next_tokens,
        embeddings,
        input.sampling_params.selected_token_idxes);
    // PD handoff: the decode instance requires get_mtp_bootstrap_embedding()
    // defined before it accepts the request (disagg_pd_scheduler). Compress the
    // full prefill hidden to one row per sequence, as
    // write_prefill_target_context stores it.
    torch::Tensor bootstrap_embeddings = embeddings;
    if (bootstrap_embeddings.size(0) !=
        static_cast<int64_t>(
            input.input_params.embedding.embedding_ids.size())) {
      torch::Tensor bootstrap_idxes =
          input.sampling_params.selected_token_idxes.to(
              torch::dtype(torch::kLong).device(bootstrap_embeddings.device()));
      bootstrap_embeddings =
          bootstrap_embeddings.index_select(/*dim=*/0, bootstrap_idxes);
    }
    output.sample_output.embeddings = bootstrap_embeddings.detach();
    clear_selected_embeddings(output);
  } else {
    clear_all_output_embeddings(output);
  }

  if (!enable_schedule_overlap() && !driver_ && !dp_driver_) {
    return std::nullopt;
  }
  return output;
}

std::optional<ForwardOutput> DFlashWorkerImpl::step_decode(
    const ForwardInput& raw_input) {
  ForwardInput input = raw_input;
  ForwardInput validate_input;

  CHECK(embedding_cache_ != nullptr)
      << "DFlash embedding cache is not allocated";

  const auto& embedding = input.input_params.embedding;
  if (embedding.mtp_bootstrap_embeddings.defined()) {
    CHECK(input.token_ids_host.defined())
        << "DFlash bootstrap requires host token ids";
    CHECK(input.token_ids_host.device().is_cpu())
        << "DFlash bootstrap host token ids must be on CPU";
    CHECK_EQ(input.token_ids_host.scalar_type(), torch::kInt)
        << "DFlash bootstrap host token ids must be int32";

    torch::Tensor bootstrap_embeddings =
        safe_to(embedding.mtp_bootstrap_embeddings,
                torch::dtype(dtype_).device(device_));
    CHECK_EQ(bootstrap_embeddings.size(0),
             static_cast<int64_t>(embedding.mtp_bootstrap_row_idxes.size()))
        << "DFlash bootstrap row count mismatch";

    Slice<int32_t> token_ids = {
        input.token_ids_host.data_ptr<int32_t>(),
        static_cast<size_t>(input.token_ids_host.numel())};
    for (int32_t i = 0;
         i < static_cast<int32_t>(embedding.mtp_bootstrap_row_idxes.size());
         ++i) {
      const int32_t row_idx = embedding.mtp_bootstrap_row_idxes[i];
      CHECK_GE(row_idx, 0) << "DFlash bootstrap row index should be valid";
      CHECK_LT(row_idx, static_cast<int32_t>(embedding.embedding_ids.size()))
          << "DFlash bootstrap row index exceeds embedding ids";
      CHECK_LT(row_idx, static_cast<int32_t>(embedding.request_ids.size()))
          << "DFlash bootstrap row index exceeds request ids";
      CHECK_LT(static_cast<int64_t>(row_idx), input.token_ids_host.numel())
          << "DFlash bootstrap row index exceeds token ids";
      embedding_cache_->write_mtp_bootstrap_context(
          embedding.embedding_ids[row_idx],
          embedding.request_ids[row_idx],
          token_ids[row_idx],
          bootstrap_embeddings[i]);
    }
  }

  std::vector<EmbeddingCache::DecodeState> last_states =
      embedding_cache_->read_decode_states(
          input.input_params.embedding.embedding_ids,
          input.input_params.embedding.request_ids);
  CHECK_EQ(last_states.size(),
           input.input_params.embedding.embedding_ids.size())
      << "DFlash decode target state count mismatch";

  update_decode_step_input(input, last_states);
  DraftBlock draft_block = run_decode_draft(input, validate_input);
  return run_validate(input, draft_block, validate_input);
}

DFlashWorkerImpl::DraftBlock DFlashWorkerImpl::run_decode_draft(
    const ForwardInput& input,
    ForwardInput& validate_input) {
  Timer timer;

  ForwardInput query_input;
  prepare_query_inputs(input, query_input);

  ForwardOutput draft_output =
      run_llm_no_sync_impl(
          *draft_impl_, query_input, *prepare_stream_, *compute_stream_)
          .value();
  // Overlap validate input preparation with the async draft forward: the draft
  // launch above returns immediately, so building validate_input here (it only
  // reads the original input; draft tokens are injected later in
  // fill_validate_input_from_draft_outputs) runs on the host while the draft
  // computes on device, instead of delaying the draft launch.
  prepare_validate_inputs(input, validate_input);
  // Unify the draft next_tokens across the tensor-parallel group before
  // process_draft_sample_output() compresses the probs into the cache, so every
  // rank caches the same selected draft prob under schedule-overlap. No-op for
  // a single rank.
  maybe_broadcast_spec_tokens(draft_output.sample_output.next_tokens);
  process_draft_sample_output(draft_output.sample_output);
  COUNTER_ADD(speculative_execution_latency_seconds_draft,
              timer.elapsed_seconds());

  // Draft emits the whole block in one forward; reshape the flat outputs into
  // [batch, num_speculative_tokens] instead of splitting into per-step outputs.
  const int32_t num_speculative_tokens = options_.num_speculative_tokens();
  const int32_t num_draft_tokens =
      static_cast<int32_t>(draft_output.sample_output.next_tokens.numel());
  CHECK_EQ(num_draft_tokens % num_speculative_tokens, 0)
      << "DFlash draft token count mismatch.";
  const int32_t batch_size = num_draft_tokens / num_speculative_tokens;
  CHECK_EQ(draft_output.sample_output.probs.numel(), num_draft_tokens)
      << "DFlash draft output requires selected draft probs.";

  DraftBlock draft_block;
  draft_block.token_ids = draft_output.sample_output.next_tokens.view(
      {batch_size, num_speculative_tokens});
  draft_block.probs = draft_output.sample_output.probs.view(
      {batch_size, num_speculative_tokens});
  draft_block.retained_inputs = take_retained_inputs(draft_output);
  return draft_block;
}

void DFlashWorkerImpl::fill_validate_input_from_draft_outputs(
    const DraftBlock& draft_block,
    ForwardInput& validate_input,
    Stream& compute_stream,
    int32_t effective_val_tokens) {
  const int32_t num_speculative_tokens = options_.num_speculative_tokens();
  const int32_t num_val_tokens = effective_val_tokens;
  const int32_t effective_speculative_tokens = effective_val_tokens - 1;
  CHECK_GE(effective_speculative_tokens, 0);
  CHECK_LE(effective_speculative_tokens, num_speculative_tokens);
  CHECK(draft_block.token_ids.defined())
      << "DFlash draft token_ids must be defined for validate token fill";
  CHECK_EQ(draft_block.token_ids.dim(), 2)
      << "DFlash draft token_ids must be [batch, num_speculative_tokens]";
  CHECK_EQ(draft_block.token_ids.size(1), num_speculative_tokens)
      << "DFlash draft token_ids width mismatch";
  CHECK(validate_input.token_ids.defined())
      << "DFlash validate token_ids must be prepared before draft token fill";
  CHECK_EQ(validate_input.token_ids.dim(), 1)
      << "DFlash validate token_ids must be flat";
  CHECK_EQ(validate_input.token_ids.numel() % num_val_tokens, 0)
      << "DFlash validate token_ids size must be divisible by validation width";

  const int64_t total_num_val_tokens = validate_input.token_ids.numel();
  const int64_t num_sequences = total_num_val_tokens / num_val_tokens;
  CHECK_EQ(draft_block.token_ids.size(0), num_sequences)
      << "DFlash draft batch must match validate sequence count";
  const torch::TensorOptions token_options = validate_input.token_ids.options();
  c10::StreamGuard stream_guard = compute_stream.set_stream_guard();
  wait_metadata_ready_event(validate_input, compute_stream);
  torch::Tensor validate_token_rows =
      validate_input.token_ids.view({num_sequences, num_val_tokens});

  validate_input.device_tensors_ready = false;
  if (effective_speculative_tokens == 0) {
    // Controller pruned every seq's speculation down to zero; nothing to fill
    // beyond the anchor column that already holds the real token.
    validate_input.device_tensors_ready = true;
    // still need to publish the compute-stream write below.
  } else {
    using ISlice = torch::indexing::Slice;
    torch::Tensor draft_slice =
        draft_block.token_ids
            .index({ISlice(),
                    ISlice(/*start=*/0, /*end=*/effective_speculative_tokens)})
            .contiguous();
    torch::Tensor draft_tokens =
        safe_to(draft_slice, token_options, /*non_blocking=*/true);
    validate_token_rows.index({ISlice(), ISlice(1, num_val_tokens)})
        .copy_(draft_tokens, /*non_blocking=*/true);
    validate_input.device_tensors_ready = true;
  }
  // Publish this compute-stream write so the target's prepare stage (which
  // consumes validate_input.token_ids under ACL-graph double buffering) waits
  // for the copy to complete before staging into the graph's persistent
  // buffer. Without this, prepare could read stale/placeholder token ids.
  record_metadata_ready_event(compute_stream, validate_input);
}

void DFlashWorkerImpl::fill_validate_input_from_draft_outputs_varlen(
    const DraftBlock& draft_block,
    ForwardInput& validate_input,
    Stream& compute_stream,
    const std::vector<int32_t>& per_seq_val_tokens) {
  const int32_t num_speculative_tokens = options_.num_speculative_tokens();
  const int64_t num_sequences = static_cast<int64_t>(per_seq_val_tokens.size());
  CHECK(draft_block.token_ids.defined())
      << "DFlash draft token_ids must be defined for varlen validate fill";
  CHECK_EQ(draft_block.token_ids.dim(), 2);
  CHECK_EQ(draft_block.token_ids.size(0), num_sequences);
  CHECK_EQ(draft_block.token_ids.size(1), num_speculative_tokens);
  CHECK(validate_input.token_ids.defined());
  CHECK_EQ(validate_input.token_ids.dim(), 1);

  const torch::TensorOptions token_options = validate_input.token_ids.options();
  c10::StreamGuard stream_guard = compute_stream.set_stream_guard();
  wait_metadata_ready_event(validate_input, compute_stream);

  validate_input.device_tensors_ready = false;

  // Compute destination offsets: seq i's draft tokens go at
  // [cu_offset[i] + 1, cu_offset[i] + per_seq_val_tokens[i]).
  std::vector<int64_t> dst_idx_vec;
  std::vector<int64_t> src_idx_vec;
  // Upper bound: each seq contributes at most num_speculative_tokens draft
  // rows (seq_val_tokens - 1 <= num_speculative_tokens).
  const size_t max_draft_rows =
      static_cast<size_t>(num_sequences) * num_speculative_tokens;
  dst_idx_vec.reserve(max_draft_rows);
  src_idx_vec.reserve(max_draft_rows);
  int64_t cu_offset = 0;
  for (int64_t seq_id = 0; seq_id < num_sequences; ++seq_id) {
    const int32_t seq_val_tokens =
        per_seq_val_tokens[static_cast<size_t>(seq_id)];
    for (int32_t j = 0; j < seq_val_tokens - 1; ++j) {
      dst_idx_vec.push_back(cu_offset + 1 + j);
      src_idx_vec.push_back(seq_id * num_speculative_tokens + j);
    }
    cu_offset += seq_val_tokens;
  }

  if (!dst_idx_vec.empty()) {
    const torch::TensorOptions long_dev_opts =
        torch::TensorOptions()
            .dtype(torch::kLong)
            .device(validate_input.token_ids.device());
    torch::Tensor dst_idx = safe_to(
        torch::tensor(dst_idx_vec, torch::TensorOptions().dtype(torch::kLong)),
        long_dev_opts,
        /*non_blocking=*/true);
    torch::Tensor src_idx = safe_to(
        torch::tensor(src_idx_vec, torch::TensorOptions().dtype(torch::kLong)),
        long_dev_opts,
        /*non_blocking=*/true);
    // Flatten [B, N] -> [B*N] and gather via src_idx.
    torch::Tensor draft_flat = draft_block.token_ids.view({-1});
    torch::Tensor draft_selected = draft_flat.index_select(/*dim=*/0, src_idx);
    torch::Tensor draft_tokens =
        safe_to(draft_selected, token_options, /*non_blocking=*/true);
    validate_input.token_ids.index_copy_(/*dim=*/0, dst_idx, draft_tokens);
  }
  validate_input.device_tensors_ready = true;
  record_metadata_ready_event(compute_stream, validate_input);
}

std::optional<ForwardOutput> DFlashWorkerImpl::run_validate(
    const ForwardInput& input,
    const DraftBlock& draft_block_in,
    ForwardInput& validate_input) {
  Timer timer;
  // Adaptive-speculative per-seq varlen validate:
  // 1. controller decides per-seq prefix_lengths from confidence/proposal
  //    probs.
  // 2. we rebuild validate_input as a *true varlen* [Σ (prefix_i+1), ...]
  //    batch — target forward runs only Σ (prefix_i+1) tokens, so batches with
  //    even a single high-confidence seq do not force the whole batch to full
  //    N+1 width (the batch-max regression from v1). MTP already scatters
  //    varlen target output back into padded dense before rejection sampling
  //    — we do the same here.
  // 3. rejection sampler is dense-only; scatter varlen target output into
  //    [B, N+1, ...] with -inf at padded positions.
  // 4. apply_pruned_prefix_lengths clips the sampler output at each seq's
  //    prefix_len so the accepted_token_ids beyond prefix_i become -1.
  const int32_t num_speculative_tokens = options_.num_speculative_tokens();
  const int32_t default_val_tokens = num_speculative_tokens + 1;
  const int32_t batch_size = input.input_params.meta.num_sequences;

  DraftBlock draft_block = draft_block_in;
  std::vector<int32_t> prefix_lengths =
      compute_adaptive_prefix_lengths(draft_block, input);
  std::vector<int32_t> per_seq_val_tokens;
  bool did_prune = false;
  int32_t max_val_tokens = default_val_tokens;
  if (!prefix_lengths.empty()) {
    // Note: we intentionally do NOT mask draft_block.probs beyond each seq's
    // prefix_len. The varlen validate path only sends prefix_lengths[i] draft
    // tokens per seq to target; and apply_pruned_prefix_lengths downstream
    // overwrites all pruned rejection-sampler outputs (via cut_mask + drop
    // mask). So the sampler's decision on pruned draft slots is irrelevant
    // to the emitted tokens — no need to touch draft_probs on the hot path.
    per_seq_val_tokens.resize(batch_size);
    max_val_tokens = 0;
    for (int32_t i = 0; i < batch_size; ++i) {
      int32_t p = std::clamp(prefix_lengths[static_cast<size_t>(i)],
                             /*min=*/0,
                             /*max=*/num_speculative_tokens);
      // Per-seq validate width = accepted-draft-count + 1 bonus. When the
      // controller decides prefix=0 (don't speculate this step), the seq
      // still must verify its bonus token, so the minimum is 1 slot — not
      // 2. A previous floor to 2 forced a phantom "draft slot" at position
      // 0 that leaked whatever the sampler emitted there past the intended
      // prefix, showing up as duplicate/garbage tokens in adaptive output.
      const int32_t width = p + 1;
      per_seq_val_tokens[static_cast<size_t>(i)] = width;
      max_val_tokens = std::max(max_val_tokens, width);
      if (width < default_val_tokens) {
        did_prune = true;
      }
    }
  }

  if (did_prune) {
    apply_per_seq_varlen_prune(input, validate_input, per_seq_val_tokens);
    fill_validate_input_from_draft_outputs_varlen(
        draft_block, validate_input, *compute_stream_, per_seq_val_tokens);
  } else {
    fill_validate_input_from_draft_outputs(
        draft_block, validate_input, *compute_stream_, max_val_tokens);
  }
  // Under DP, publish this rank's true validate token count to all DP peers so
  // DpEpPadding computes matching MoE all-to-all pads. Runs on both branches
  // (pruned and dense) and on every DP rank so the collective stays in
  // lockstep. No-op when the DP group spans a single rank.
  int32_t local_total_val_tokens = 0;
  if (did_prune) {
    for (int32_t v : per_seq_val_tokens) {
      local_total_val_tokens += v;
    }
  } else {
    local_total_val_tokens = batch_size * max_val_tokens;
  }
  sync_dp_global_token_nums_after_prune(validate_input.input_params,
                                        local_total_val_tokens);
  ForwardOutput target_output =
      run_llm_no_sync_impl(
          *impl_, validate_input, *prepare_stream_, *compute_stream_)
          .value();
  COUNTER_ADD(speculative_execution_latency_seconds_target,
              timer.elapsed_seconds());

  // Scatter varlen target output back to dense [B, max_val_tokens] layout so
  // the rejection sampler (dense API) can consume it. Pad next_tokens with -1
  // (reject marker), not 0: Qwen id 0 is "!", and a padded slot must not
  // surface a real token if any downstream consumer reads it before
  // apply_pruned_prefix_lengths masks trailing positions to -1.
  if (did_prune) {
    adaptive_pruning::scatter_varlen_target_output_to_dense(
        target_output,
        per_seq_val_tokens,
        batch_size,
        max_val_tokens,
        /*next_token_pad_value=*/-1);
  }

  timer.reset();
  SampleOutput val_output =
      validate(input.sampling_params,
               draft_block,
               target_output,
               max_val_tokens,
               did_prune ? per_seq_val_tokens : std::vector<int32_t>{});
  // Post-process: mask sampler output beyond each seq's prefix_len so seq
  // accepted counts respect the per-seq decision even under batch-max
  // dense rejection sampling.
  if (did_prune) {
    // effective_prefix[i] mirrors the controller's decision (0-based, clamped
    // to [0, num_speculative_tokens]). Since per_seq_val_tokens[i] is exactly
    // prefix_lengths[i] + 1 now (bonus slot only when prefix=0), we could
    // equivalently write per_seq_val_tokens[i] - 1; keeping the raw
    // prefix_lengths read here documents the semantic and stays robust if the
    // width calculation grows another guard later.
    std::vector<int32_t> effective_prefix(batch_size);
    for (int32_t i = 0; i < batch_size; ++i) {
      int32_t p = prefix_lengths[static_cast<size_t>(i)];
      effective_prefix[static_cast<size_t>(i)] =
          std::clamp(p, 0, options_.num_speculative_tokens());
    }
    adaptive_pruning::PrunedPrefixMasks masks =
        adaptive_pruning::build_pruned_prefix_masks(
            effective_prefix,
            max_val_tokens - 1,
            val_output.next_tokens.device());
    // Sync logprob/top-logprob at each seq's cut position with the target's
    // resampled token (paper Section 3.2: cut-position token switches from
    // the rejected draft to a target resample; its logprob has to switch
    // too). Skipping this leaves logprobs pointing at the draft token when
    // logprobs=true in the sampling request.
    adaptive_pruning::sync_pruned_boundary_outputs(
        val_output, target_output, batch_size, max_val_tokens, masks);
    adaptive_pruning::apply_pruned_prefix_lengths(
        val_output,
        target_output.sample_output.next_tokens,
        max_val_tokens - 1,
        masks);
  }
  COUNTER_ADD(speculative_execution_latency_seconds_validation,
              timer.elapsed_seconds());

  // target forward and validate()'s reads share compute_stream_, so they
  // serialize without a cross-stream wait; the sync below only makes the
  // accepted tokens host-visible for the D2H copy and context-cache write.
  maybe_broadcast_spec_tokens(val_output.next_tokens);
  compute_stream_->synchronize();
  val_output.next_tokens = val_output.next_tokens.to(torch::kCPU);
  trace_block_spec_step_state_from_host_input(
      options_.speculative_algorithm(), input, val_output.next_tokens);
  // Precise adaptive-aware metrics on the already-CPU tensor: static path
  // passes an empty per_seq_val_tokens and every row counts full width;
  // adaptive passes the per-seq widths so padded tail slots aren't counted
  // as rejections. Zero extra device sync — we're already on CPU.
  record_validate_metrics(
      val_output, did_prune ? per_seq_val_tokens : std::vector<int32_t>{});
  write_target_context_to_cache(input, val_output);

  if (!enable_schedule_overlap() && !driver_ && !dp_driver_) {
    return std::nullopt;
  }
  val_output.embeddings = torch::Tensor();
  target_output.sample_output = val_output;
  return target_output;
}

SampleOutput DFlashWorkerImpl::validate(
    const SamplingParameters& sampling_params,
    const DraftBlock& draft_block,
    const ForwardOutput& target_output,
    int32_t effective_val_tokens,
    const std::vector<int32_t>& per_seq_val_tokens) {
  // Draft already emits the whole block [batch, num_speculative_tokens]; feed
  // it straight to the verifier without the per-step select/view/cat round
  // trip. The shared rejection sampler uses MTP's dense contract, so
  // reconstruct dense draft probs unless the selected-only optimization is on.
  const int32_t vocab_size =
      static_cast<int32_t>(target_output.logits.size(/*dim=*/-1));
  const bool enable_opt_validate_probs =
      ::xllm::SpeculativeConfig::get_instance().enable_opt_validate_probs();
  // Slice draft block down to the effective validate width; the rejection
  // sampler only sees the tokens the target actually validated.
  const int32_t effective_speculative_tokens = effective_val_tokens - 1;
  using ISlice = torch::indexing::Slice;
  torch::Tensor pruned_token_ids =
      draft_block.token_ids
          .index({ISlice(),
                  ISlice(/*start=*/0, /*end=*/effective_speculative_tokens)})
          .contiguous();
  torch::Tensor pruned_probs =
      draft_block.probs
          .index({ISlice(),
                  ISlice(/*start=*/0, /*end=*/effective_speculative_tokens)})
          .contiguous();
  auto [draft_token_ids, draft_probs] =
      specBuilder::draftProbs::build_validate_tensors_from_block(
          pruned_token_ids,
          pruned_probs,
          vocab_size,
          enable_opt_validate_probs);
  return validate(sampling_params,
                  draft_token_ids,
                  draft_probs,
                  target_output,
                  effective_val_tokens,
                  per_seq_val_tokens);
}

SampleOutput DFlashWorkerImpl::validate(
    const SamplingParameters& sampling_params,
    const torch::Tensor& draft_token_ids,
    const torch::Tensor& draft_probs,
    const ForwardOutput& target_output,
    int32_t effective_val_tokens,
    const std::vector<int32_t>& per_seq_val_tokens) {
  const int32_t num_val_tokens = effective_val_tokens;
  // Derive batch_size from the target logits rows rather than next_tokens so
  // the reshape stays valid regardless of how the target was sampled.
  const int32_t num_logits_rows =
      static_cast<int32_t>(target_output.logits.size(/*dim=*/0));
  CHECK_EQ(num_logits_rows % num_val_tokens, 0)
      << "DFlash validate target logits rows must be divisible by validation "
         "width";
  const int32_t batch_size = num_logits_rows / num_val_tokens;

  using torch::indexing::None;
  using ISlice = torch::indexing::Slice;
  torch::Tensor target_next_tokens_2d =
      target_output.sample_output.next_tokens.view(
          {batch_size, num_val_tokens});
  torch::Tensor bonus_token_ids;
  if (per_seq_val_tokens.empty()) {
    // Uniform batch-max width: bonus is at the fixed last column.
    bonus_token_ids = target_next_tokens_2d
                          .index({ISlice(), ISlice(num_val_tokens - 1, None)})
                          .view({-1, 1});
  } else {
    // Per-seq varlen: seq i's bonus lives at dense col
    // (per_seq_val_tokens[i] - 1) because the varlen->dense scatter placed
    // the seq's rows order-preserved at cols [0, per_seq_val_tokens[i]).
    CHECK_EQ(static_cast<int32_t>(per_seq_val_tokens.size()), batch_size)
        << "per_seq_val_tokens size must match validate batch";
    std::vector<int64_t> bonus_cols(static_cast<size_t>(batch_size));
    for (int32_t i = 0; i < batch_size; ++i) {
      const int32_t w = per_seq_val_tokens[static_cast<size_t>(i)];
      bonus_cols[static_cast<size_t>(i)] = std::max(w - 1, 0);
    }
    torch::Tensor bonus_idx =
        torch::tensor(bonus_cols,
                      torch::TensorOptions()
                          .dtype(torch::kLong)
                          .device(target_next_tokens_2d.device()))
            .view({batch_size, 1});
    bonus_token_ids =
        target_next_tokens_2d.gather(/*dim=*/1, bonus_idx).view({-1, 1});
  }

  torch::Tensor target_logits = target_output.logits.view(
      {batch_size, num_val_tokens, target_output.logits.size(/*dim=*/-1)});
  return spec_verify::run_rejection_sampling(
      {.do_sample = sampling_params.do_sample,
       .all_random_sample = sampling_params.all_random_sample,
       .all_greedy_sample = sampling_params.all_greedy_sample},
      draft_token_ids,
      draft_probs,
      target_logits,
      target_output,
      bonus_token_ids,
      enable_fused_kernel_);
}

void DFlashWorkerImpl::process_draft_sample_output(
    SampleOutput& sample_output) {
  specBuilder::draftProbs::compress_sample_output_for_cache(sample_output);
}

void DFlashWorkerImpl::maybe_broadcast_spec_tokens(torch::Tensor& tokens) {
  if (get_optimization_config().enable_spec_token_broadcast) {
    c10::StreamGuard stream_guard = compute_stream_->set_stream_guard();
    broadcast_spec_tokens(tokens, spec_broadcast_group(parallel_args_));
  }
}

void DFlashWorkerImpl::update_decode_step_input(
    ForwardInput& input,
    const std::vector<EmbeddingCache::DecodeState>& last_states,
    const SpeculativePreparedHostInputWorkspace* fixed_host_workspace) const {
  const int32_t num_sequences = input.input_params.meta.num_sequences;
  CHECK_EQ(last_states.size(), static_cast<size_t>(num_sequences))
      << "DFlash decode context state count mismatch";
  const bool enable_cache_correction = enable_schedule_overlap();

  std::vector<int32_t> token_ids_vec;
  std::vector<int32_t> positions_vec;
  std::vector<int32_t> kv_seq_lens_vec;
  torch::Tensor fixed_token_ids;
  torch::Tensor fixed_positions;
  int32_t* fixed_token_ids_data = nullptr;
  int32_t* fixed_positions_data = nullptr;
  if (fixed_host_workspace == nullptr) {
    token_ids_vec.reserve(num_sequences);
    positions_vec.reserve(num_sequences);
#if defined(USE_NPU)
    kv_seq_lens_vec.reserve(num_sequences);
#else
    kv_seq_lens_vec.reserve(num_sequences + 1);
#endif
  } else {
    CHECK(fixed_host_workspace->token_ids.defined());
    CHECK(fixed_host_workspace->positions.defined());
    CHECK(fixed_host_workspace->token_ids.device().is_cpu());
    CHECK(fixed_host_workspace->positions.device().is_cpu());
    CHECK_EQ(fixed_host_workspace->token_ids.scalar_type(), torch::kInt);
    CHECK_EQ(fixed_host_workspace->positions.scalar_type(), torch::kInt);
    CHECK(fixed_host_workspace->token_ids.is_contiguous());
    CHECK(fixed_host_workspace->positions.is_contiguous());
    CHECK_GE(fixed_host_workspace->token_ids.numel(), num_sequences);
    CHECK_GE(fixed_host_workspace->positions.numel(), num_sequences);
    CHECK_EQ(input.input_params.attention.host.kv_seq_lens.size(),
             static_cast<size_t>(num_sequences))
        << "Prepared Block-Spec fixed Host staging requires one KV length per "
           "sequence";
    fixed_token_ids = fixed_host_workspace->token_ids.narrow(
        /*dim=*/0, /*start=*/0, num_sequences);
    fixed_positions = fixed_host_workspace->positions.narrow(
        /*dim=*/0, /*start=*/0, num_sequences);
    fixed_token_ids_data = fixed_token_ids.data_ptr<int32_t>();
    fixed_positions_data = fixed_positions.data_ptr<int32_t>();
  }

  const torch::Tensor& token_ids_cpu = input.token_ids_host;
  const torch::Tensor& positions_cpu = input.positions_host;
  Slice<int32_t> input_token_ids = {token_ids_cpu.data_ptr<int32_t>(),
                                    static_cast<size_t>(token_ids_cpu.numel())};
  Slice<int32_t> input_positions = {positions_cpu.data_ptr<int32_t>(),
                                    static_cast<size_t>(positions_cpu.numel())};

  for (int32_t seq_id = 0; seq_id < num_sequences; ++seq_id) {
    CHECK_LT(static_cast<size_t>(seq_id), input_token_ids.size())
        << "DFlash decode context token seq_id out of range, seq_id=" << seq_id;
    CHECK_LT(static_cast<size_t>(seq_id), input_positions.size())
        << "DFlash decode context position seq_id out of range, seq_id="
        << seq_id;
    const EmbeddingCache::DecodeState& state = last_states[seq_id];
    const int32_t input_token_id = input_token_ids[seq_id];
    const bool input_is_fake_token = input_token_id < 0;
    // Rewrite fake input tokens to the last committed real token so KV cache
    // scatter has a valid id; only apply the recorded position offset when the
    // cached state is still valid.
    const bool rewrite_fake_token =
        enable_cache_correction && input_is_fake_token;
    const bool use_cache_correction = rewrite_fake_token && state.valid;
    const int32_t position_offset =
        use_cache_correction ? state.position_offset : 0;
    const int32_t current_position = input_positions[seq_id] + position_offset;
    const int32_t current_kv_len = specBuilder::calc_kv_len(
        input.input_params.attention.host.kv_seq_lens, seq_id, position_offset);
    const int32_t expected_kv_len = current_position + 1;

    CHECK_EQ(expected_kv_len, current_kv_len)
        << "DFlash decode context position/kv_len mismatch, seq_id=" << seq_id
        << ", current_position=" << current_position
        << ", current_kv_len=" << current_kv_len;

    const int32_t output_token_id =
        rewrite_fake_token ? state.token_id : input_token_id;
    if (fixed_host_workspace == nullptr) {
      token_ids_vec.emplace_back(output_token_id);
      positions_vec.emplace_back(current_position);
      specBuilder::append_seq_len_by_layout(kv_seq_lens_vec, current_kv_len);
    } else {
      fixed_token_ids_data[seq_id] = output_token_id;
      fixed_positions_data[seq_id] = current_position;
      input.input_params.attention.host.kv_seq_lens[seq_id] = current_kv_len;
    }
  }

  if (fixed_host_workspace == nullptr) {
    input.token_ids_host = specBuilder::make_cpu_int_tensor(token_ids_vec);
    input.positions_host = specBuilder::make_cpu_int_tensor(positions_vec);
    input.input_params.attention.host.kv_seq_lens = std::move(kv_seq_lens_vec);
  } else {
    input.token_ids_host = fixed_token_ids;
    input.positions_host = fixed_positions;
  }
  input.device_tensors_ready = false;
}

void DFlashWorkerImpl::prepare_validate_inputs(
    const ForwardInput& input,
    ForwardInput& validate_input,
    bool stage_sampling_on_host,
    const SpeculativePreparedHostInputWorkspace* fixed_host_workspace,
    specBuilder::DecodeBuildWorkspace* decode_build_workspace) {
  c10::StreamGuard stream_guard = prepare_stream_->set_stream_guard();
  SpeculativeWorkerImpl::prepare_validate_inputs(input,
                                                 validate_input,
                                                 stage_sampling_on_host,
                                                 fixed_host_workspace,
                                                 decode_build_workspace);
  validate_input.metadata_ready_event.reset();
  validate_input.input_params.embedding.input_embedding = torch::Tensor();
  record_metadata_ready_event(*prepare_stream_, validate_input);
}

void DFlashWorkerImpl::prepare_query_inputs(
    const ForwardInput& input,
    ForwardInput& query_input,
    bool stage_sampling_on_host,
    const SpeculativePreparedHostInputWorkspace* fixed_host_workspace,
    specBuilder::DecodeBuildWorkspace* decode_build_workspace) {
  CHECK(fixed_host_workspace == nullptr || stage_sampling_on_host)
      << "Fixed DFlash Host workspace requires Host staging";
  c10::StreamGuard stream_guard = prepare_stream_->set_stream_guard();
  query_input = input;
  query_input.device_tensors_ready = false;
  ModelInputParams& input_params = query_input.input_params;
  input_params.embedding.input_embedding = torch::Tensor();
  dflash_detail::invalidate_draft_model_geometry(input_params);

  specBuilder::DecodeBuildWorkspace local_build_workspace;
  specBuilder::DecodeBuildWorkspace& build_workspace =
      decode_build_workspace == nullptr ? local_build_workspace
                                        : *decode_build_workspace;
  specBuilder::reset_decode_build_workspace(build_workspace);
  specBuilder::DecodeBuildBuffers& buf = build_workspace.buffers;
  std::vector<int32_t>& selected_idxes = build_workspace.selected_indices;
  const bool use_block_parallel_rows = draft_use_block_parallel_rows();
  build_query_rows(input,
                   mask_token_id_,
                   options_.num_speculative_tokens(),
                   options_.block_size(),
                   sample_from_anchor(),
                   use_block_parallel_rows,
                   buf,
                   selected_idxes);
  // DFlash: (1 + N) rows per seq; DSpark (sample_from_anchor): N rows.
  const int32_t query_width = dflash_detail::decode_draft_width(
      options_.num_speculative_tokens(), sample_from_anchor());
  // DFlash emits query_width rows per seq unconditionally, so DP shape
  // symmetry holds by construction. Catch scheduler regressions that break
  // the invariant (see MTP dp_enabled idle-rank branch).
  const int32_t num_sequences_query = input.input_params.meta.num_sequences;
  CHECK_EQ(static_cast<int32_t>(buf.out_positions.size()),
           num_sequences_query * query_width)
      << "DFlash per-seq row count must be uniform (query_width=" << query_width
      << ", num_sequences=" << num_sequences_query << ")";

  torch::TensorOptions token_options = input.token_ids.options();
  torch::TensorOptions position_options = input.positions.options();
  if (stage_sampling_on_host) {
    token_options = token_options.device(torch::kCPU);
    position_options = position_options.device(torch::kCPU);
  }
  if (fixed_host_workspace == nullptr) {
    specBuilder::set_token_position_tensors(query_input,
                                            buf.out_token_ids,
                                            buf.out_positions,
                                            token_options,
                                            position_options);
  } else {
    query_input.token_ids_host = specBuilder::copy_cpu_int_values_out(
        buf.out_token_ids, fixed_host_workspace->token_ids);
    query_input.positions_host = specBuilder::copy_cpu_int_values_out(
        buf.out_positions, fixed_host_workspace->positions);
    query_input.token_ids = query_input.token_ids_host;
    query_input.positions = query_input.positions_host;
  }
  input_params.meta.batch_forward_type = draft_batch_forward_type();
  if (use_block_parallel_rows) {
    expand_block_parallel_sequence_rows(input_params, query_width);
  }
  specBuilder::update_input_params(input_params,
                                   buf,
                                   use_block_parallel_rows ? 1 : query_width,
                                   std::move(buf.out_q_seq_lens),
                                   std::move(buf.out_q_cu_seq_lens),
                                   buf.meta.kv_max_seq_len,
                                   std::move(buf.out_kv_seq_lens),
                                   /*update_block_tables=*/
                                   use_block_parallel_rows);
  build_workspace.owns_kv_seq_lens = true;
  build_workspace.owns_q_seq_lens = true;
  build_workspace.owns_q_cu_seq_lens = true;
  scale_speculative_parallel_token_counts(input_params, query_width);
  if (!stage_sampling_on_host) {
    input_params.attention.rebuild_device_buffer(device_);
  }

  torch::TensorOptions idx_options =
      torch::TensorOptions()
          .dtype(torch::kInt)
          .device(stage_sampling_on_host ? torch::Device(torch::kCPU)
                                         : device_.unwrap());
  // Prepared keeps controls on Host for the fixed Arena's direct H2D. Legacy
  // retains the existing asynchronous upload on the prepare stream.
  if (fixed_host_workspace == nullptr) {
    torch::Tensor selected_token_idxes =
        specBuilder::make_cpu_int_tensor(selected_idxes);
    query_input.sampling_params.selected_token_idxes =
        stage_sampling_on_host ? selected_token_idxes
                               : safe_to(selected_token_idxes,
                                         idx_options,
                                         /*non_blocking=*/true);
    query_input.sampling_params.sample_idxes =
        torch::arange(static_cast<int64_t>(selected_idxes.size()), idx_options);
  } else {
    query_input.sampling_params.selected_token_idxes =
        specBuilder::copy_cpu_int_values_out(
            selected_idxes, fixed_host_workspace->selected_token_idxes);
    query_input.sampling_params.sample_idxes =
        specBuilder::fill_cpu_int_range_out(
            /*start=*/0,
            /*step=*/1,
            static_cast<int64_t>(selected_idxes.size()),
            fixed_host_workspace->sample_idxes);
  }
  // Force the draft sampler to emit selected-token probabilities even on the
  // greedy path (temperature=0); the rejection sampler needs them to verify
  // the block. Without this the greedy sampler skips probs entirely.
  query_input.sampling_params.return_probs = true;
  repeat_sampling_params(query_input.sampling_params,
                         options_.num_speculative_tokens(),
                         fixed_host_workspace == nullptr
                             ? torch::Tensor()
                             : fixed_host_workspace->do_sample,
                         fixed_host_workspace == nullptr
                             ? torch::Tensor()
                             : fixed_host_workspace->repeated_sampling_storage);
  query_input.device_tensors_ready = true;
}

void DFlashWorkerImpl::write_context_kv(
    const ForwardInput& input,
    const torch::Tensor& context_hidden,
    const torch::Tensor& positions_device,
    const torch::Tensor& new_cache_slots_device,
    bool synchronize_completion) {
  dflash_detail::check_context_kv_write_tensor_contract(
      context_hidden,
      positions_device,
      new_cache_slots_device,
      device_.unwrap(),
      expected_context_hidden_size_,
      dtype_);

  // Both the target forward that produced context_hidden and this pass run
  // on compute_stream_, so no explicit event dance is needed — the stream
  // orders them. Model methods below use torch ops on the same stream.
  c10::StreamGuard stream_guard = compute_stream_->set_stream_guard();

#if defined(USE_NPU)
  // PD PUSH: the draft context-KV scattered below is not covered by the target
  // push, so wire the draft push here. Overwrite layer_synchronizer with a
  // draft-sized one (one event per draft layer); the target's was already
  // waited on before this runs, so the overwrite is safe. No-op when
  // transfer_kv_infos is empty. NPU-only: the scatter's record_event loop is
  // NPU-gated, so a non-NPU push would stall on events that never record.
  KVTransferCompletion kv_transfers;
  if (options_.kv_cache_transfer_mode() == "PUSH" &&
      !input.transfer_kv_infos.empty()) {
    CHECK(synchronize_completion)
        << "Prepared DFlash does not support asynchronous PD-PUSH";
    std::shared_ptr<NPULayerSynchronizerImpl> layer_synchronizer =
        std::make_shared<NPULayerSynchronizerImpl>(
            draft_impl_->context_.get_model_args().n_layers());
    const_cast<ModelInputParams*>(&(input.input_params))
        ->parallel.layer_synchronizer = layer_synchronizer;
    kv_transfers.add(kv_cache_transfer_->push_kv_blocks_async(
        input.transfer_kv_infos,
        draft_impl_->context_.get_parallel_args(),
        layer_synchronizer,
        /*is_spec_draft=*/true));
  }
#endif

  ModelOutput scatter_output =
      draft_impl_->write_context_kv(context_hidden,
                                    positions_device,
                                    new_cache_slots_device,
                                    input.input_params);
  // Model returns an empty ModelOutput (hidden_states undefined) when the
  // per-layer NPULayerSynchronizer::record_event fails mid-scatter under
  // PD-PUSH transfer. Fail fast instead of silently continuing: subsequent
  // layers won't have been scattered and the PD transfer side would block
  // forever waiting on the missing per-layer event.
  CHECK(scatter_output.hidden_states.defined())
      << "DFlash context-KV scatter failed (layer_synchronizer record_event "
         "returned false); PD-PUSH transfer would deadlock.";

  // Legacy execution returns directly to the scheduler and therefore waits
  // here before the next step may read the draft cache. Prepared execution
  // records its Slot completion Event after this scatter and waits only at
  // Consume/reuse, keeping the launch path free of a Host synchronization.
  if (synchronize_completion) {
    compute_stream_->synchronize();
  }

#if defined(USE_NPU)
  // Wait for the draft KV push (if any) so the source draft cache is not
  // overwritten by the next step while the transfer is still reading it
  // (mirrors step_internal's wait_kv_push()). No-op when no push was issued.
  if (synchronize_completion) {
    CHECK(kv_transfers.wait()) << "DFlash draft context-KV push failed";
  }
#endif
}

void DFlashWorkerImpl::write_target_context_to_cache(
    const ForwardInput& input,
    const SampleOutput& validate_output) {
  const torch::Tensor& accepted_embeddings = validate_output.embeddings;
  CHECK(accepted_embeddings.defined())
      << "DFlash validate target embeddings are undefined.";
  CHECK_EQ(accepted_embeddings.dim(), 3)
      << "DFlash validate target embeddings must be [batch,width,hidden].";

  torch::Tensor accepted_tokens = validate_output.next_tokens;
  CHECK(accepted_tokens.defined()) << "DFlash accepted tokens are undefined.";
  if (accepted_tokens.scalar_type() != torch::kInt64) {
    accepted_tokens = accepted_tokens.to(torch::kInt64);
  }
  DCHECK(accepted_tokens.is_contiguous())
      << "DFlash accepted tokens must be contiguous (guaranteed by upstream "
         ".to(kCPU) and .to(kInt64) branches); check for stride changes if "
         "this fires.";

  CHECK_EQ(accepted_tokens.dim(), 2)
      << "DFlash accepted tokens must be [batch,width].";
  const int64_t batch_size = accepted_tokens.size(0);
  const int64_t token_width = accepted_tokens.size(1);
  CHECK_EQ(accepted_embeddings.size(0), batch_size)
      << "DFlash accepted token/embedding batch mismatch.";
  CHECK_EQ(accepted_embeddings.size(1), token_width)
      << "DFlash accepted token/embedding width mismatch.";

  specBuilder::DecodeBuildBuffers buf;
  std::vector<int64_t> accepted_idxes = build_accepted_context_rows(
      input, accepted_tokens, options_.block_size(), buf);
  torch::TensorOptions host_index_options = torch::TensorOptions()
                                                .dtype(torch::kLong)
                                                .device(torch::kCPU)
                                                .pinned_memory(true);
  torch::TensorOptions device_index_options =
      torch::TensorOptions()
          .dtype(torch::kLong)
          .device(accepted_embeddings.device());
  c10::StreamGuard stream_guard = prepare_stream_->set_stream_guard();
  torch::Tensor accepted_index =
      safe_to(torch::tensor(accepted_idxes, host_index_options),
              device_index_options,
              /*non_blocking=*/true);
  torch::Tensor flat_embeddings = accepted_embeddings.reshape(
      {batch_size * token_width, accepted_embeddings.size(/*dim=*/2)});
  torch::Tensor context_hidden =
      flat_embeddings.index_select(/*dim=*/0, accepted_index);
  torch::Tensor positions_device =
      cpu_int_vec_to_device(buf.out_positions, device_);
  torch::Tensor new_cache_slots_device =
      cpu_int_vec_to_device(buf.out_new_cache_slots, device_);
  // Publish the prepare_stream_ work (index_select producing context_hidden +
  // pinned H2D copies for positions/slots) so compute_stream_ waits for it
  // before the model reads these tensors. Without this, torch does not
  // enforce cross-stream ordering: the write_context_kv scatter could launch
  // before index_select finishes, producing corrupt KV cache. The prefill
  // caller runs its producer on compute_stream_, so no event is needed there.
  StreamEventPtr context_hidden_ready_event =
      prepare_stream_->record_event_or_sync();
  if (context_hidden_ready_event != nullptr) {
    CHECK(compute_stream_->wait_event(context_hidden_ready_event))
        << "failed to wait DFlash context hidden ready event";
  }
  write_context_kv(
      input, context_hidden, positions_device, new_cache_slots_device);
  CHECK(!input.input_params.embedding.embedding_ids.empty())
      << "DFlash target context cache write requires embedding ids";
  embedding_cache_->write_target_context(
      input.input_params.embedding.embedding_ids,
      input.input_params.embedding.request_ids,
      validate_output.next_tokens,
      validate_output.embeddings,
      options_.num_speculative_tokens());
}

// -----------------------------------------------------------------------------
// Adaptive-speculative helpers (DFlash + DSpark).
// -----------------------------------------------------------------------------

std::vector<int32_t> DFlashWorkerImpl::compute_adaptive_prefix_lengths(
    const DraftBlock& draft_block,
    const ForwardInput& input) {
  const int32_t num_speculative_tokens = options_.num_speculative_tokens();
  if (adaptive_spec_controller_ == nullptr ||
      !adaptive_spec_controller_->enabled()) {
    return {};
  }
  // Prefer the trained ConfidenceHead output when present (DSpark); otherwise
  // fall back to sampler-gathered proposal probs (DFlash / DSpark without a
  // confidence head).
  //
  // Both signals are per-step conditional accept probabilities: c_k =
  // P(step k accepted | prefix accepted). ConfidenceHead simply replaces
  // proposal probs as a better-trained estimator of the same quantity. The
  // controller chain-rule multiplies them to obtain path probs
  // a_{r,j} = ∏ c_i (paper Section 3.2.2 Algorithm 1). Same code path for
  // both signal sources.
  //
  // Note (DSpark v1): the ConfidenceHead is loaded from the released
  // checkpoint but is *not* STS-calibrated yet (paper Section 3.2.1 "Post-hoc
  // Calibration"). Raw sigmoid confidence is overconfident, so the cumulative
  // product decays incorrectly at longer block sizes and can over-prune.
  // DSPARK_CONFIDENCE_TEMPERATURE (see qwen3_dspark.h) offers a single
  // temperature knob to approximate STS until we ship a proper offline
  // per-position calibration table.
  torch::Tensor probs_for_controller = draft_block.confidence_probs.defined()
                                           ? draft_block.confidence_probs
                                           : draft_block.probs;
  if (!probs_for_controller.defined()) {
    return {};
  }
  if (probs_for_controller.dim() != 2 ||
      probs_for_controller.size(1) != num_speculative_tokens) {
    LOG(WARNING) << "Adaptive: unexpected probs shape "
                 << probs_for_controller.sizes()
                 << " — falling back to full width.";
    return {};
  }

  const int32_t batch_size = input.input_params.meta.num_sequences;
  std::vector<double> per_seq_kv_lens(static_cast<size_t>(batch_size), 0.0);
  const Slice<int32_t> kv_seq_lens =
      input.input_params.attention.host.kv_seq_lens;
  for (int32_t i = 0; i < batch_size; ++i) {
    per_seq_kv_lens[static_cast<size_t>(i)] = static_cast<double>(
        specBuilder::calc_kv_len(kv_seq_lens, i, /*offset=*/0));
  }

  std::vector<int32_t> prefix_lengths =
      adaptive_spec_controller_->select_pruned_prefix_lengths(
          probs_for_controller,
          /*full_draft_time_ms=*/0.0,
          per_seq_kv_lens);
  return prefix_lengths;
}

void DFlashWorkerImpl::apply_per_seq_varlen_prune(
    const ForwardInput& input,
    ForwardInput& validate_input,
    const std::vector<int32_t>& per_seq_val_tokens) {
  const int32_t num_sequences = input.input_params.meta.num_sequences;
  CHECK_EQ(static_cast<int32_t>(per_seq_val_tokens.size()), num_sequences);
  c10::StreamGuard stream_guard = prepare_stream_->set_stream_guard();
  ForwardInput prepared_input = input;
  prepared_input.metadata_ready_event.reset();
  ForwardInput new_validate;
  SpeculativeWorkerImpl::prepare_validate_inputs(
      prepared_input, new_validate, per_seq_val_tokens);
  new_validate.input_params.embedding.input_embedding = torch::Tensor();
  record_metadata_ready_event(*prepare_stream_, new_validate);
  validate_input = std::move(new_validate);
}

void DFlashWorkerImpl::record_validate_metrics(
    const SampleOutput& val_output,
    const std::vector<int32_t>& per_seq_val_tokens) const {
  if (!val_output.next_tokens.defined() || val_output.next_tokens.dim() != 2 ||
      val_output.next_tokens.numel() == 0) {
    return;
  }
  const int32_t batch_size =
      static_cast<int32_t>(val_output.next_tokens.size(0));
  const int32_t width = static_cast<int32_t>(val_output.next_tokens.size(1));
  const int32_t num_speculative_tokens = options_.num_speculative_tokens();
  if (num_speculative_tokens <= 0 || width < 2) {
    return;
  }
  CHECK(val_output.next_tokens.device().is_cpu())
      << "record_validate_metrics expects next_tokens already on CPU to avoid "
         "a blocking device sync on the hot path";
  const bool have_per_seq = !per_seq_val_tokens.empty();
  if (have_per_seq) {
    CHECK_EQ(per_seq_val_tokens.size(), static_cast<size_t>(batch_size))
        << "per_seq_val_tokens size mismatch with next_tokens batch";
  }

  torch::Tensor next_tokens_cpu =
      val_output.next_tokens.to(torch::kInt64).contiguous();
  const int64_t* token_data = next_tokens_cpu.const_data_ptr<int64_t>();
  int64_t num_draft_tokens = 0;
  int64_t accepted_count = 0;
  for (int32_t seq_id = 0; seq_id < batch_size; ++seq_id) {
    // seq_width = target-side validate width for this seq (anchor + drafts).
    // Under adaptive per-seq varlen prune it is per_seq_val_tokens[i], else
    // the full dense width.
    int32_t seq_width = width;
    if (have_per_seq) {
      // lo=1: a controller prefix=0 decision yields per_seq_val_tokens[i]==1
      // (bonus only, zero drafts). Clamping to 1 gives prefix_len=0 so a
      // fully-pruned seq contributes no draft/accept counts; clamping to 2
      // would fabricate one phantom draft + one phantom accept.
      seq_width = std::clamp(per_seq_val_tokens[static_cast<size_t>(seq_id)],
                             /*lo=*/1,
                             /*hi=*/width);
    }
    // Drafts attempted for this seq = seq_width - 1 (bonus column excluded).
    const int32_t prefix_len = seq_width - 1;
    num_draft_tokens += prefix_len;

    // Count accepted drafts by walking columns [0, prefix_len) — the first
    // -1 marks the boundary where the sampler rejected. Padding tail past
    // prefix_len is ignored so it never counts as rejection.
    const int64_t row_offset =
        static_cast<int64_t>(seq_id) * static_cast<int64_t>(width);
    int32_t emitted = 0;
    for (int32_t token_idx = 0; token_idx < prefix_len; ++token_idx) {
      if (token_data[row_offset + token_idx] < 0) {
        break;
      }
      ++emitted;
    }
    accepted_count += emitted;
  }
  COUNTER_ADD(speculative_num_draft_tokens_total, num_draft_tokens);
  COUNTER_ADD(speculative_num_accepted_tokens_total, accepted_count);
}

}  // namespace xllm
