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

#include "runtime/prepared_task/prepared_task_pipeline.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "common/metrics.h"
#include "core/framework/config/speculative_config.h"
#include "core/framework/speculative/block_spec_async_state.h"
#include "core/framework/speculative/spec_input_builder.h"
#include "core/layers/common/dsa_metadata.h"
#include "core/layers/common/dsa_metadata_builder.h"
#include "core/layers/common/expanded_decode_metadata_builder.h"
#include "runtime/dflash_worker_impl.h"
#include "runtime/llm_worker_impl.h"
#include "runtime/mtp_worker_impl.h"
#include "runtime/params_utils.h"
#include "runtime/prepared_task/block_spec_prepared_task_adapter.h"
#include "runtime/prepared_task/dspark_prepared_sampling.h"
#include "runtime/prepared_task/mtp_prepared_task_adapter.h"
#include "runtime/prepared_task/mtp_prepared_task_backend.h"
#include "runtime/prepared_task/prepared_input_arena.h"
#include "runtime/prepared_task/prepared_pipeline_activator.h"

namespace xllm {
namespace {

TEST(DSAManagerOrderContractTest, AcceptsCanonicalPresentRoleSubsets) {
  EXPECT_NO_FATAL_FAILURE(layer::validate_dsa_group_order({
      {DSACacheType::SLIDING_WINDOW, 1, 4096},
      {DSACacheType::TOKEN, 4, 128},
      {DSACacheType::TOKEN, 128, 128},
  }));
  EXPECT_NO_FATAL_FAILURE(layer::validate_dsa_group_order({
      {DSACacheType::SLIDING_WINDOW, 1, 4096},
      {DSACacheType::TOKEN, 128, 128},
  }));
}

TEST(DSAManagerOrderContractTest, RejectsC128BeforeC4) {
  EXPECT_DEATH(layer::validate_dsa_group_order({
                   {DSACacheType::SLIDING_WINDOW, 1, 4096},
                   {DSACacheType::TOKEN, 128, 128},
                   {DSACacheType::TOKEN, 4, 128},
               }),
               "canonical C4/C128 order");
}

TEST(DSAManagerOrderContractTest,
     CanonicalRatiosIgnoreCompressionConfigurationOrder) {
  const std::vector<int32_t> manager_ratios =
      canonical_dsa_token_manager_ratios({128, 4, 1, 128});

  EXPECT_EQ(manager_ratios, (std::vector<int32_t>{4, 128}));
}

TEST(PreparedModelOutputWorkspaceTest,
     GathersSelectedEmbeddingsIntoFixedDestination) {
  const torch::Tensor embeddings =
      torch::tensor({{1.0F, 2.0F}, {3.0F, 4.0F}, {5.0F, 6.0F}});
  const torch::Tensor selected_token_idxes =
      torch::tensor({2, 0}, torch::dtype(torch::kInt));
  torch::Tensor destination = torch::empty({2, 2}, torch::kFloat);
  const void* destination_address = destination.data_ptr();

  torch::Tensor selected_embeddings = gather_prepared_selected_embeddings(
      embeddings, selected_token_idxes, destination);

  EXPECT_EQ(selected_embeddings.data_ptr(), destination_address);
  EXPECT_EQ(destination.data_ptr(), destination_address);
  EXPECT_TRUE(torch::equal(selected_embeddings,
                           torch::tensor({{5.0F, 6.0F}, {1.0F, 2.0F}})));
}

TEST(PreparedModelOutputWorkspaceTest,
     ValidatesFixedTokenAndEmbeddingBindings) {
  PreparedModelOutputWorkspace workspace;
  workspace.next_tokens = torch::empty({2}, torch::kLong);
  workspace.selected_embeddings = torch::empty({2, 3}, torch::kFloat);
  SampleOutput sample_output;
  sample_output.next_tokens = workspace.next_tokens;
  sample_output.selected_embeddings = workspace.selected_embeddings;

  EXPECT_NO_FATAL_FAILURE(
      check_prepared_model_output_binding(sample_output, workspace));

  sample_output.next_tokens = torch::empty({2}, torch::kLong);
  EXPECT_DEATH(check_prepared_model_output_binding(sample_output, workspace),
               "token output.*replaced the fixed output storage");
}

TEST(PreparedModelOutputWorkspaceTest,
     ValidatesDecodeEmbeddingBindingThroughEmbeddingsField) {
  PreparedModelOutputWorkspace workspace;
  workspace.selected_embeddings = torch::empty({2, 3}, torch::kFloat);
  SampleOutput sample_output;
  sample_output.embeddings = workspace.selected_embeddings;

  EXPECT_NO_FATAL_FAILURE(
      check_prepared_model_output_binding(sample_output, workspace));

  sample_output.embeddings = torch::empty({2, 3}, torch::kDouble);
  EXPECT_DEATH(check_prepared_model_output_binding(sample_output, workspace),
               "selected embedding output.*changed dtype");
}

TEST(MtpPreparedEmbeddingWorkspaceTest, StacksRowsIntoFixedDestination) {
  const std::vector<torch::Tensor> embedding_rows = {
      torch::tensor({1.0F, 2.0F}), torch::tensor({3.0F, 4.0F})};
  torch::Tensor destination = torch::empty({2, 2}, torch::kFloat);
  const void* destination_address = destination.data_ptr();

  torch::Tensor stacked_embeddings =
      stack_prepared_embedding_rows_out(embedding_rows, destination);

  EXPECT_EQ(stacked_embeddings.data_ptr(), destination_address);
  EXPECT_EQ(destination.data_ptr(), destination_address);
  EXPECT_TRUE(torch::equal(stacked_embeddings,
                           torch::tensor({{1.0F, 2.0F}, {3.0F, 4.0F}})));
}

TEST(MtpPreparedHostInputWorkspaceTest,
     WritesValuesAndRangesIntoFixedDestination) {
  torch::Tensor destination = torch::empty({6}, torch::kInt);
  const void* destination_address = destination.data_ptr();

  torch::Tensor copied_values = specBuilder::copy_cpu_int_values_out(
      std::vector<int32_t>{4, 7, 9}, destination);

  EXPECT_EQ(copied_values.data_ptr(), destination_address);
  EXPECT_TRUE(
      torch::equal(copied_values, torch::tensor({4, 7, 9}, torch::kInt)));

  torch::Tensor range_values = specBuilder::fill_cpu_int_range_out(
      /*start=*/1,
      /*step=*/2,
      /*count=*/3,
      destination);

  EXPECT_EQ(range_values.data_ptr(), destination_address);
  EXPECT_EQ(destination.data_ptr(), destination_address);
  EXPECT_TRUE(
      torch::equal(range_values, torch::tensor({1, 3, 5}, torch::kInt)));
}

TEST(PreparedDecodeBuildWorkspaceTest,
     ReclaimsMovedVectorsWithoutReplacingStorage) {
  specBuilder::DecodeBuildWorkspace workspace;
  specBuilder::reserve_decode_build_workspace(workspace, /*row_capacity=*/8);
  specBuilder::reset_decode_build_workspace(workspace);

  workspace.buffers.out_token_ids.emplace_back(7);
  workspace.buffers.out_positions.emplace_back(11);
  workspace.buffers.out_new_cache_slots.emplace_back(13);
  workspace.auxiliary_kv_seq_lens.emplace_back(17);
  workspace.auxiliary_q_seq_lens.emplace_back(1);
  workspace.auxiliary_q_cu_seq_lens.emplace_back(1);
  const int32_t* token_address = workspace.buffers.out_token_ids.data();
  const int32_t* position_address = workspace.buffers.out_positions.data();
  const int32_t* cache_slot_address =
      workspace.buffers.out_new_cache_slots.data();
  const int32_t* kv_seq_lens_address = workspace.auxiliary_kv_seq_lens.data();
  const int32_t* q_seq_lens_address = workspace.auxiliary_q_seq_lens.data();
  const int32_t* q_cu_seq_lens_address =
      workspace.auxiliary_q_cu_seq_lens.data();

  ModelInputParams input_params;
  input_params.attention.host.new_cache_slots =
      std::move(workspace.buffers.out_new_cache_slots);
  input_params.attention.host.kv_seq_lens =
      std::move(workspace.auxiliary_kv_seq_lens);
  input_params.attention.host.q_seq_lens =
      std::move(workspace.auxiliary_q_seq_lens);
  input_params.attention.host.q_cu_seq_lens =
      std::move(workspace.auxiliary_q_cu_seq_lens);
  workspace.owns_kv_seq_lens = true;
  workspace.owns_q_seq_lens = true;
  workspace.owns_q_cu_seq_lens = true;
  workspace.uses_auxiliary_kv_seq_lens = true;
  workspace.uses_auxiliary_q_seq_lens = true;
  workspace.uses_auxiliary_q_cu_seq_lens = true;

  specBuilder::reclaim_decode_build_workspace(input_params, workspace);
  specBuilder::reset_decode_build_workspace(workspace);
  workspace.buffers.out_token_ids.emplace_back(19);
  workspace.buffers.out_positions.emplace_back(23);
  workspace.buffers.out_new_cache_slots.emplace_back(29);
  workspace.auxiliary_kv_seq_lens.emplace_back(31);
  workspace.auxiliary_q_seq_lens.emplace_back(1);
  workspace.auxiliary_q_cu_seq_lens.emplace_back(1);

  EXPECT_EQ(workspace.buffers.out_token_ids.data(), token_address);
  EXPECT_EQ(workspace.buffers.out_positions.data(), position_address);
  EXPECT_EQ(workspace.buffers.out_new_cache_slots.data(), cache_slot_address);
  EXPECT_EQ(workspace.auxiliary_kv_seq_lens.data(), kv_seq_lens_address);
  EXPECT_EQ(workspace.auxiliary_q_seq_lens.data(), q_seq_lens_address);
  EXPECT_EQ(workspace.auxiliary_q_cu_seq_lens.data(), q_cu_seq_lens_address);
}

TEST(BlockSpecPreparedHostInputWorkspaceTest,
     KeepsQueryAndTargetControlsIndependent) {
  constexpr int64_t kCapacity = 8;
  const torch::TensorOptions int_options =
      torch::TensorOptions().dtype(torch::kInt).device(torch::kCPU);
  const torch::TensorOptions bool_options =
      torch::TensorOptions().dtype(torch::kBool).device(torch::kCPU);
  const torch::TensorOptions byte_options =
      torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU);
  SpeculativePreparedHostInputWorkspace query_workspace{
      torch::empty({kCapacity}, int_options),
      torch::empty({kCapacity}, int_options),
      torch::empty({kCapacity}, int_options),
      torch::empty({kCapacity}, int_options),
      torch::empty({kCapacity}, bool_options),
      torch::empty({128}, byte_options)};
  SpeculativePreparedHostInputWorkspace target_workspace{
      torch::empty({kCapacity}, int_options),
      torch::empty({kCapacity}, int_options),
      torch::empty({kCapacity}, int_options),
      torch::empty({kCapacity}, int_options),
      torch::empty({kCapacity}, bool_options),
      torch::empty({128}, byte_options)};
  const void* query_token_address = query_workspace.token_ids.data_ptr();
  const void* query_sample_address = query_workspace.sample_idxes.data_ptr();
  const void* query_do_sample_address = query_workspace.do_sample.data_ptr();
  const void* target_token_address = target_workspace.token_ids.data_ptr();

  torch::Tensor query_tokens = specBuilder::copy_cpu_int_values_out(
      std::vector<int32_t>{11, -1, 22, -1}, query_workspace.token_ids);
  torch::Tensor query_selected = specBuilder::fill_cpu_int_range_out(
      /*start=*/1,
      /*step=*/2,
      /*count=*/2,
      query_workspace.selected_token_idxes);
  torch::Tensor query_samples = specBuilder::fill_cpu_int_range_out(
      /*start=*/0,
      /*step=*/1,
      /*count=*/2,
      query_workspace.sample_idxes);
  torch::Tensor query_do_sample = specBuilder::fill_cpu_bool_out(
      /*value=*/false, /*count=*/2, query_workspace.do_sample);
  torch::Tensor target_tokens = specBuilder::copy_cpu_int_values_out(
      std::vector<int32_t>{11, -1, -2, 22, -1, -2}, target_workspace.token_ids);

  EXPECT_EQ(query_tokens.data_ptr(), query_token_address);
  EXPECT_EQ(query_samples.data_ptr(), query_sample_address);
  EXPECT_EQ(query_do_sample.data_ptr(), query_do_sample_address);
  EXPECT_EQ(target_tokens.data_ptr(), target_token_address);
  EXPECT_NE(query_tokens.data_ptr(), target_tokens.data_ptr());
  EXPECT_TRUE(torch::equal(query_selected, torch::tensor({1, 3}, torch::kInt)));
  EXPECT_TRUE(torch::equal(query_samples, torch::tensor({0, 1}, torch::kInt)));
  EXPECT_FALSE(query_do_sample.any().item<bool>());
  EXPECT_TRUE(torch::equal(
      target_tokens, torch::tensor({11, -1, -2, 22, -1, -2}, torch::kInt)));
}

TEST(SpeculativePreparedSamplingWorkspaceTest,
     RepeatsPenaltyAndTokenStatsIntoFixedStorage) {
  SamplingParameters source;
  source.frequency_penalties = torch::tensor({0.25F, 0.5F});
  source.presence_penalties = torch::tensor({0.1F, 0.2F});
  source.repetition_penalties = torch::tensor({1.1F, 1.2F});
  source.temperatures = torch::tensor({0.7F, 0.8F});
  source.top_p = torch::tensor({0.9F, 0.95F});
  source.top_k = torch::tensor({4, 8}, torch::kLong);
  source.unique_token_ids = torch::tensor({{11, 12}, {21, 22}}, torch::kLong);
  source.unique_token_counts = torch::tensor({{1, 2}, {3, 4}}, torch::kInt);
  source.unique_token_ids_lens = torch::tensor({2, 1}, torch::kInt);
  source.filter_mask =
      torch::tensor({{0.0F, -1.0F}, {-2.0F, 0.0F}}, torch::kFloat);
  source.filter_bitmask = torch::tensor({{3, 5}, {7, 9}}, torch::kInt);
  source.do_sample = torch::tensor({false, false}, torch::kBool);
  const void* do_sample_address = source.do_sample.data_ptr();
  SamplingParameters repeated = source;
  SamplingParameters repeated_again = source;
  torch::Tensor storage = torch::empty({512}, torch::kUInt8);

  repeat_speculative_sampling_metadata_out(repeated, /*repeats=*/3, storage);
  const void* frequency_address = repeated.frequency_penalties.data_ptr();
  repeat_speculative_sampling_metadata_out(
      repeated_again, /*repeats=*/3, storage);

  EXPECT_EQ(repeated_again.frequency_penalties.data_ptr(), frequency_address);
  EXPECT_EQ(repeated.do_sample.data_ptr(), do_sample_address);
  EXPECT_TRUE(
      torch::allclose(repeated.frequency_penalties,
                      torch::tensor({0.25F, 0.25F, 0.25F, 0.5F, 0.5F, 0.5F})));
  EXPECT_TRUE(torch::equal(repeated.top_k,
                           torch::tensor({4, 4, 4, 8, 8, 8}, torch::kLong)));
  EXPECT_TRUE(torch::equal(
      repeated.unique_token_ids,
      torch::tensor(
          {{11, 12}, {11, 12}, {11, 12}, {21, 22}, {21, 22}, {21, 22}},
          torch::kLong)));
  EXPECT_TRUE(torch::equal(
      repeated.unique_token_counts,
      torch::tensor({{1, 2}, {1, 2}, {1, 2}, {3, 4}, {3, 4}, {3, 4}},
                    torch::kInt)));
  EXPECT_TRUE(torch::equal(repeated.unique_token_ids_lens,
                           torch::tensor({2, 2, 2, 1, 1, 1}, torch::kInt)));
  EXPECT_TRUE(torch::equal(repeated.filter_mask,
                           torch::tensor({{0.0F, -1.0F},
                                          {0.0F, -1.0F},
                                          {0.0F, -1.0F},
                                          {-2.0F, 0.0F},
                                          {-2.0F, 0.0F},
                                          {-2.0F, 0.0F}},
                                         torch::kFloat)));
  EXPECT_TRUE(torch::equal(
      repeated.filter_bitmask,
      torch::tensor({{3, 5}, {3, 5}, {3, 5}, {7, 9}, {7, 9}, {7, 9}},
                    torch::kInt)));
}

TEST(SpeculativePreparedSamplingWorkspaceTest,
     RejectsInvalidStorageAndEmptyRows) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  SamplingParameters source;
  source.frequency_penalties = torch::tensor({0.25F, 0.5F});

  EXPECT_DEATH(repeat_speculative_sampling_metadata_out(
                   source,
                   /*repeats=*/3,
                   torch::empty({1}, torch::kUInt8)),
               "exceeds fixed Host workspace");
  EXPECT_DEATH(
      repeat_speculative_sampling_metadata_out(source,
                                               /*repeats=*/3,
                                               torch::empty({64}, torch::kInt)),
      "scalar_type");
  EXPECT_DEATH(repeat_speculative_sampling_metadata_out(
                   source,
                   /*repeats=*/3,
                   torch::empty({8, 8}, torch::kUInt8).transpose(0, 1)),
               "is_contiguous");

  SamplingParameters empty_source;
  empty_source.frequency_penalties = torch::empty({0}, torch::kFloat);
  EXPECT_DEATH(repeat_speculative_sampling_metadata_out(
                   empty_source,
                   /*repeats=*/3,
                   torch::empty({64}, torch::kUInt8)),
               "cannot repeat an empty row dimension");
}

struct AdapterOperation {
  std::string phase;
  uint64_t task_seq_no = 0;
  int32_t slot_id = 0;
  std::thread::id thread_id;
};

class FakePreparedTaskAdapter final : public PreparedTaskAdapter {
 public:
  PreparedTaskKind classify(const ForwardInput& input) const override {
    if (input.input_params.meta.batch_forward_type.is_empty()) {
      return PreparedTaskKind::EMPTY;
    }
    if (input.input_params.meta.batch_forward_type.is_decode()) {
      return PreparedTaskKind::DECODE;
    }
    return PreparedTaskKind::PREFILL_LIKE;
  }

  void prepare(const ForwardInput& input, ExecutionSlot& slot) override {
    slot.prepared_input = input;
    record("prepare", slot);
  }

  void launch(ExecutionSlot& slot) override {
    record("launch", slot);
    ForwardOutput output;
    output.prepared_token =
        static_cast<int64_t>(slot.prepared_input.input_params.meta.batch_id);
    slot.output = std::move(output);
  }

  std::optional<ForwardOutput> consume(ExecutionSlot& slot) override {
    record("consume", slot);
    return std::move(slot.output);
  }

  std::vector<AdapterOperation> operations() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return operations_;
  }

 private:
  void record(const std::string& phase, const ExecutionSlot& slot) {
    std::lock_guard<std::mutex> lock(mutex_);
    operations_.emplace_back(AdapterOperation{
        phase, slot.task_seq_no, slot.slot_id, std::this_thread::get_id()});
  }

  mutable std::mutex mutex_;
  std::vector<AdapterOperation> operations_;
};

class BlockingLaunchPreparedTaskAdapter final : public PreparedTaskAdapter {
 public:
  PreparedTaskKind classify(const ForwardInput&) const override {
    return PreparedTaskKind::DECODE;
  }

  void prepare(const ForwardInput& input, ExecutionSlot& slot) override {
    slot.prepared_input = input;
  }

  void launch(ExecutionSlot& slot) override {
    {
      std::unique_lock<std::mutex> lock(mutex_);
      launch_started_ = true;
      launch_started_cv_.notify_one();
      release_launch_cv_.wait(lock, [this]() { return release_launch_; });
    }
    ForwardOutput output;
    output.prepared_token =
        static_cast<int64_t>(slot.prepared_input.input_params.meta.batch_id);
    slot.output = std::move(output);
  }

  std::optional<ForwardOutput> consume(ExecutionSlot& slot) override {
    return std::move(slot.output);
  }

  void wait_until_launch_started() {
    std::unique_lock<std::mutex> lock(mutex_);
    launch_started_cv_.wait(lock, [this]() { return launch_started_; });
  }

  void release_launch() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      release_launch_ = true;
    }
    release_launch_cv_.notify_one();
  }

 private:
  std::mutex mutex_;
  std::condition_variable launch_started_cv_;
  std::condition_variable release_launch_cv_;
  bool launch_started_ = false;
  bool release_launch_ = false;
};

class ReuseFencePreparedTaskAdapter final : public PreparedTaskAdapter {
 public:
  PreparedTaskKind classify(const ForwardInput&) const override {
    return PreparedTaskKind::DECODE;
  }

  bool consumes_predecessor_state(const ForwardInput& input) const override {
    return input.input_params.embedding.predecessor_rows.defined();
  }

  void prepare_slot_for_reuse(ExecutionSlot& slot) override {
    if (!slot.successor_consumes_state) {
      return;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    reuse_wait_observed_ = slot.successor_read_enqueued;
  }

  void prepare(const ForwardInput& input, ExecutionSlot& slot) override {
    slot.prepared_input = input;
  }

  void launch_predecessor_continuation(
      ExecutionSlot& slot,
      ExecutionSlot& predecessor_slot) override {
    EXPECT_EQ(slot.task_seq_no, predecessor_slot.task_seq_no + 1);
    {
      std::unique_lock<std::mutex> lock(mutex_);
      predecessor_read_started_ = true;
      predecessor_read_started_cv_.notify_one();
      release_predecessor_read_cv_.wait(
          lock, [this]() { return release_predecessor_read_; });
    }
  }

  void launch(ExecutionSlot& slot) override {
    ForwardOutput output;
    output.prepared_token =
        static_cast<int64_t>(slot.prepared_input.input_params.meta.batch_id);
    slot.output = std::move(output);
  }

  std::optional<ForwardOutput> consume(ExecutionSlot& slot) override {
    if (slot.task_seq_no == 0) {
      {
        std::lock_guard<std::mutex> lock(mutex_);
        first_consume_started_ = true;
      }
      first_consume_started_cv_.notify_one();
    }
    return std::move(slot.output);
  }

  void wait_until_predecessor_read_started() {
    std::unique_lock<std::mutex> lock(mutex_);
    predecessor_read_started_cv_.wait(
        lock, [this]() { return predecessor_read_started_; });
  }

  void wait_until_first_consume_started() {
    std::unique_lock<std::mutex> lock(mutex_);
    first_consume_started_cv_.wait(lock,
                                   [this]() { return first_consume_started_; });
  }

  void release_predecessor_read() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      release_predecessor_read_ = true;
    }
    release_predecessor_read_cv_.notify_one();
  }

  bool reuse_wait_observed() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return reuse_wait_observed_;
  }

 private:
  mutable std::mutex mutex_;
  std::condition_variable predecessor_read_started_cv_;
  std::condition_variable release_predecessor_read_cv_;
  std::condition_variable first_consume_started_cv_;
  bool predecessor_read_started_ = false;
  bool release_predecessor_read_ = false;
  bool first_consume_started_ = false;
  bool reuse_wait_observed_ = false;
};

class TrackingPredecessorPreparedTaskAdapter final
    : public PreparedTaskAdapter {
 public:
  PreparedTaskKind classify(const ForwardInput& input) const override {
    if (input.input_params.meta.batch_forward_type.is_empty()) {
      return PreparedTaskKind::EMPTY;
    }
    if (input.input_params.meta.batch_forward_type.is_decode()) {
      return PreparedTaskKind::DECODE;
    }
    return PreparedTaskKind::PREFILL_LIKE;
  }

  bool consumes_predecessor_state(const ForwardInput& input) const override {
    return input.input_params.meta.batch_forward_type.is_decode() &&
           input.input_params.embedding.predecessor_rows.defined();
  }

  void prepare(const ForwardInput& input, ExecutionSlot& slot) override {
    slot.prepared_input = input;
  }

  void launch_predecessor_continuation(
      ExecutionSlot& slot,
      ExecutionSlot& predecessor_slot) override {
    std::lock_guard<std::mutex> lock(mutex_);
    predecessor_reads_.emplace_back(PredecessorRead{slot.task_seq_no,
                                                    slot.slot_id,
                                                    predecessor_slot.slot_id,
                                                    predecessor_slot.state});
  }

  void launch(ExecutionSlot& slot) override {
    ForwardOutput output;
    output.prepared_token =
        static_cast<int64_t>(slot.prepared_input.input_params.meta.batch_id);
    slot.output = std::move(output);
  }

  std::optional<ForwardOutput> consume(ExecutionSlot& slot) override {
    return std::move(slot.output);
  }

  struct PredecessorRead {
    uint64_t task_seq_no = 0;
    int32_t slot_id = 0;
    int32_t predecessor_slot_id = 0;
    ExecutionSlotState predecessor_state = ExecutionSlotState::FREE;
  };

  std::vector<PredecessorRead> predecessor_reads() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return predecessor_reads_;
  }

 private:
  mutable std::mutex mutex_;
  std::vector<PredecessorRead> predecessor_reads_;
};

struct MtpBackendLaunch {
  uint64_t task_seq_no = 0;
  int32_t slot_id = 0;
  int32_t predecessor_slot_id = -1;
  MtpPreparedInvocationKind kind = MtpPreparedInvocationKind::EMPTY_COLLECTIVE;
  int32_t draft_step = -1;
};

class RecordingMtpPreparedTaskBackend final : public MtpPreparedTaskBackend {
 public:
  void prepare(const ForwardInput& input,
               ExecutionSlot& slot,
               const MtpPreparedTaskPlan& plan) override {
    slot.prepared_input = input;
    std::lock_guard<std::mutex> lock(mutex_);
    prepared_plan_sizes_.emplace_back(plan.invocations.size());
  }

  void launch(const MtpPreparedInvocation& invocation,
              ExecutionSlot& slot,
              ExecutionSlot* predecessor_slot) override {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      launches_.emplace_back(MtpBackendLaunch{
          slot.task_seq_no,
          slot.slot_id,
          predecessor_slot == nullptr ? -1 : predecessor_slot->slot_id,
          invocation.kind,
          invocation.draft_step});
    }
    if (invocation.kind == MtpPreparedInvocationKind::PUBLISH_STEP_STATE ||
        invocation.kind == MtpPreparedInvocationKind::EMPTY_COLLECTIVE) {
      ForwardOutput output;
      output.prepared_token =
          static_cast<int64_t>(slot.prepared_input.input_params.meta.batch_id);
      slot.output = std::move(output);
    }
  }

  std::optional<ForwardOutput> consume(ExecutionSlot& slot) override {
    return std::move(slot.output);
  }

  std::vector<MtpBackendLaunch> launches() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return launches_;
  }

  std::vector<size_t> prepared_plan_sizes() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return prepared_plan_sizes_;
  }

 private:
  mutable std::mutex mutex_;
  std::vector<MtpBackendLaunch> launches_;
  std::vector<size_t> prepared_plan_sizes_;
};

struct BlockSpecBackendLaunch {
  uint64_t task_seq_no = 0;
  int32_t slot_id = 0;
  int32_t predecessor_slot_id = -1;
  BlockSpecPreparedInvocationKind kind =
      BlockSpecPreparedInvocationKind::EMPTY_COLLECTIVE;
  int32_t block_step = -1;
};

class RecordingBlockSpecPreparedTaskBackend final
    : public BlockSpecPreparedTaskBackend {
 public:
  void prepare(const ForwardInput& input,
               ExecutionSlot& slot,
               const BlockSpecPreparedTaskPlan& plan) override {
    slot.prepared_input = input;
    std::lock_guard<std::mutex> lock(mutex_);
    prepared_plan_sizes_.emplace_back(plan.invocations.size());
  }

  void launch(const BlockSpecPreparedInvocation& invocation,
              ExecutionSlot& slot,
              ExecutionSlot* predecessor_slot) override {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      launches_.emplace_back(BlockSpecBackendLaunch{
          slot.task_seq_no,
          slot.slot_id,
          predecessor_slot == nullptr ? -1 : predecessor_slot->slot_id,
          invocation.kind,
          invocation.block_step});
    }
    if (invocation.kind == BlockSpecPreparedInvocationKind::WRITE_CONTEXT_KV ||
        invocation.kind == BlockSpecPreparedInvocationKind::EMPTY_COLLECTIVE) {
      ForwardOutput output;
      output.prepared_token =
          static_cast<int64_t>(slot.prepared_input.input_params.meta.batch_id);
      slot.output = std::move(output);
    }
  }

  std::optional<ForwardOutput> consume(ExecutionSlot& slot) override {
    return std::move(slot.output);
  }

  std::vector<BlockSpecBackendLaunch> launches() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return launches_;
  }

  std::vector<size_t> prepared_plan_sizes() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return prepared_plan_sizes_;
  }

 private:
  mutable std::mutex mutex_;
  std::vector<BlockSpecBackendLaunch> launches_;
  std::vector<size_t> prepared_plan_sizes_;
};

StreamEventPtr make_unrecorded_test_event() {
#if defined(USE_NPU)
  return std::make_shared<StreamEvent>(static_cast<aclrtEvent>(nullptr));
#else
  return std::make_shared<StreamEvent>(c10::DeviceType::CPU);
#endif
}

MtpPreparedTaskBufferConfig make_mtp_backend_buffer_config() {
  MtpPreparedTaskBufferConfig config;
  config.device = torch::Device(torch::kCPU);
  config.input_arena_capacity_bytes = 4096;
  config.slot_count = 2;
  config.max_rows = 3;
  config.accepted_token_capacity = 4;
  config.max_verify_width = 4;
  config.max_block_table_width = 4;
  config.embedding_size = 2;
  return config;
}

BlockSpecPreparedTaskBufferConfig make_block_spec_backend_buffer_config() {
  BlockSpecPreparedTaskBufferConfig config;
  config.device = torch::Device(torch::kCPU);
  config.input_arena_capacity_bytes = 4096;
  config.slot_count = 2;
  config.block_size = 4;
  config.max_rows = 3;
  config.accepted_token_capacity = 4;
  config.context_hidden_size = 2;
  return config;
}

torch::TensorOptions cpu_tensor_options(torch::ScalarType dtype) {
  return torch::TensorOptions().dtype(dtype).device(torch::kCPU);
}

class FixedBufferMtpPreparedTaskBackend final
    : public MtpPreparedTaskBackendBase {
 public:
  FixedBufferMtpPreparedTaskBackend()
      : FixedBufferMtpPreparedTaskBackend(make_mtp_backend_buffer_config()) {}

  explicit FixedBufferMtpPreparedTaskBackend(
      const MtpPreparedTaskBufferConfig& config)
      : MtpPreparedTaskBackendBase(config) {
    slot_bindings_.reserve(2);
    for (int32_t slot_id = 0; slot_id < 2; ++slot_id) {
      SlotBindings bindings;
      bindings.continuation.tail_tokens =
          torch::empty({3}, cpu_tensor_options(torch::kLong));
      bindings.continuation.previous_tokens =
          torch::empty({3}, cpu_tensor_options(torch::kLong));
      bindings.continuation.tail_embeddings =
          torch::empty({3, 2}, cpu_tensor_options(torch::kFloat));
      bindings.continuation.previous_embeddings =
          torch::empty({3, 2}, cpu_tensor_options(torch::kFloat));
      bindings.continuation.base_positions =
          torch::empty({3}, cpu_tensor_options(torch::kInt));
      bindings.continuation.base_kv_seq_lens =
          torch::empty({3}, cpu_tensor_options(torch::kInt));
      bindings.publish_source.accepted_tokens =
          torch::empty({3, 4}, cpu_tensor_options(torch::kLong));
      bindings.publish_source.accepted_embeddings =
          torch::empty({3, 4, 2}, cpu_tensor_options(torch::kFloat));
      bindings.publish_source.embedding_placeholder =
          torch::tensor({-100.0F, -101.0F}, cpu_tensor_options(torch::kFloat));
      bindings.publish_source.base_positions =
          torch::empty({3}, cpu_tensor_options(torch::kInt));
      bindings.publish_source.base_kv_seq_lens =
          torch::empty({3}, cpu_tensor_options(torch::kInt));
      slot_bindings_.emplace_back(std::move(bindings));
    }
  }

  const mtp_async::MtpDeviceStepState& published_state(int32_t slot_id) const {
    return device_step_state(slot_id);
  }

  MtpPreparedPublishSource& publish_source(int32_t slot_id) {
    CHECK_GE(slot_id, 0);
    CHECK_LT(static_cast<size_t>(slot_id), slot_bindings_.size());
    return slot_bindings_[static_cast<size_t>(slot_id)].publish_source;
  }

  mtp_async::MtpContinuationState& continuation(int32_t slot_id) {
    CHECK_GE(slot_id, 0);
    CHECK_LT(static_cast<size_t>(slot_id), slot_bindings_.size());
    return slot_bindings_[static_cast<size_t>(slot_id)].continuation;
  }

  void bind_continuation_for_test(int32_t slot_id) {
    bind_continuation_state(slot_id, continuation(slot_id));
  }

  int32_t reuse_wait_count() const { return reuse_wait_count_; }

  std::vector<MtpPreparedInvocationKind> launches() const { return launches_; }

 protected:
  void prepare_staged_input(const ForwardInput& input,
                            ExecutionSlot& slot,
                            const MtpPreparedTaskPlan&) override {
    CHECK(stage_primary_input(slot.slot_id, input, slot.prepared_input));
    const size_t slot_index = static_cast<size_t>(slot.slot_id);
    bind_continuation_state(slot.slot_id,
                            slot_bindings_[slot_index].continuation);
    bind_publish_source(slot.slot_id,
                        slot_bindings_[slot_index].publish_source);
  }

  void launch_model_invocation(const MtpPreparedInvocation& invocation,
                               ExecutionSlot&) override {
    EXPECT_EQ(c10::impl::VirtualGuardImpl(c10::DeviceType::CPU)
                  .getStream(c10::Device(torch::kCPU)),
              task_stream());
    launches_.emplace_back(invocation.kind);
  }

  c10::Stream task_stream() const override {
    return c10::Stream(c10::Stream::DEFAULT, c10::Device(torch::kCPU));
  }

  StreamEventPtr record_task_stream_event() const override {
    return make_unrecorded_test_event();
  }

  bool wait_prepare_stream_event(const StreamEventPtr& event) const override {
    EXPECT_NE(event, nullptr);
    ++reuse_wait_count_;
    return true;
  }

 private:
  struct SlotBindings {
    mtp_async::MtpContinuationState continuation;
    MtpPreparedPublishSource publish_source;
  };

  std::vector<SlotBindings> slot_bindings_;
  mutable int32_t reuse_wait_count_ = 0;
  std::vector<MtpPreparedInvocationKind> launches_;
};

class FixedBufferBlockSpecPreparedTaskBackend final
    : public BlockSpecPreparedTaskBackendBase {
 public:
  FixedBufferBlockSpecPreparedTaskBackend()
      : BlockSpecPreparedTaskBackendBase(
            make_block_spec_backend_buffer_config()) {
    slot_sources_.reserve(2);
    for (int32_t slot_id = 0; slot_id < 2; ++slot_id) {
      SlotSource source;
      source.publish_source.accepted_tokens =
          torch::tensor({{10, 11, -1, -1}, {20, 21, 22, -1}},
                        cpu_tensor_options(torch::kLong));
      source.publish_source.accepted_context_hidden =
          torch::arange(/*start=*/0,
                        /*end=*/16,
                        cpu_tensor_options(torch::kFloat))
              .view({2, 4, 2})
              .contiguous();
      source.publish_source.base_positions =
          torch::tensor({5, 7}, cpu_tensor_options(torch::kInt));
      source.publish_source.block_tables =
          torch::tensor({{100, 101, 102, 103}, {200, 201, 202, 203}},
                        cpu_tensor_options(torch::kInt));
      source.continuation.anchor_tokens =
          torch::tensor({90, 91, 92}, cpu_tensor_options(torch::kLong));
      source.continuation.anchor_context_hidden =
          torch::tensor({{900.0F, 901.0F}, {910.0F, 911.0F}, {920.0F, 921.0F}},
                        cpu_tensor_options(torch::kFloat));
      source.continuation.base_positions =
          torch::tensor({30, 31, 32}, cpu_tensor_options(torch::kInt));
      slot_sources_.emplace_back(std::move(source));
    }
  }

  const block_spec_async::BlockSpecDeviceStepState& published_state(
      int32_t slot_id) const {
    return device_step_state(slot_id);
  }

  int32_t predecessor_patch_count() const { return predecessor_patch_count_; }

  int32_t reuse_wait_count() const { return reuse_wait_count_; }

  BlockSpecPreparedPublishSource& publish_source(int32_t slot_id) {
    CHECK_GE(slot_id, 0);
    CHECK_LT(static_cast<size_t>(slot_id), slot_sources_.size());
    return slot_sources_[static_cast<size_t>(slot_id)].publish_source;
  }

  void bind_publish_source_for_test(int32_t slot_id) {
    bind_publish_source(slot_id, publish_source(slot_id));
  }

  const block_spec_async::BlockSpecContinuationState& continuation(
      int32_t slot_id) const {
    return slot_sources_[static_cast<size_t>(slot_id)].continuation;
  }

  std::vector<BlockSpecPreparedInvocationKind> launches() const {
    return launches_;
  }

 protected:
  void prepare_staged_input(const ForwardInput& input,
                            ExecutionSlot& slot,
                            const BlockSpecPreparedTaskPlan& plan) override {
    CHECK(stage_primary_input(slot.slot_id, input, slot.prepared_input));
    if (plan.task_kind == PreparedTaskKind::DECODE) {
      bind_publish_source(
          slot.slot_id,
          slot_sources_[static_cast<size_t>(slot.slot_id)].publish_source);
    }
  }

  void launch_predecessor_patch(
      const block_spec_async::BlockSpecDeviceStepState& predecessor_state,
      ExecutionSlot& slot) override {
    EXPECT_GT(predecessor_state.accepted_lengths[0].item<int64_t>(), 0);
    SlotSource& source = slot_sources_[static_cast<size_t>(slot.slot_id)];
    patch_continuation_state(
        slot.slot_id,
        predecessor_state,
        slot.prepared_input.input_params.embedding.predecessor_rows,
        source.continuation);
    ++predecessor_patch_count_;
  }

  void launch_model_invocation(const BlockSpecPreparedInvocation& invocation,
                               ExecutionSlot&) override {
    EXPECT_EQ(c10::impl::VirtualGuardImpl(c10::DeviceType::CPU)
                  .getStream(c10::Device(torch::kCPU)),
              task_stream());
    launches_.emplace_back(invocation.kind);
  }

  c10::Stream task_stream() const override {
    return c10::Stream(c10::Stream::DEFAULT, c10::Device(torch::kCPU));
  }

  StreamEventPtr record_task_stream_event() const override {
    return make_unrecorded_test_event();
  }

  bool wait_prepare_stream_event(const StreamEventPtr& event) const override {
    EXPECT_NE(event, nullptr);
    ++reuse_wait_count_;
    return true;
  }

 private:
  struct SlotSource {
    BlockSpecPreparedPublishSource publish_source;
    block_spec_async::BlockSpecContinuationState continuation;
  };

  std::vector<SlotSource> slot_sources_;
  int32_t predecessor_patch_count_ = 0;
  mutable int32_t reuse_wait_count_ = 0;
  std::vector<BlockSpecPreparedInvocationKind> launches_;
};

ForwardInput make_input(BatchForwardType forward_type, uint64_t batch_id) {
  ForwardInput input;
  input.input_params.meta.batch_forward_type = forward_type;
  input.input_params.meta.batch_id = batch_id;
  return input;
}

TEST(PreparedPipelineActivatorTest, InitializesImmediatePipelineOnce) {
  PreparedPipelineActivator activator;
  activator.configure(PreparedPipelineActivationStage::IMMEDIATE);
  int32_t factory_calls = 0;

  activator.initialize_at_construction([&factory_calls]() { ++factory_calls; });

  EXPECT_TRUE(activator.initialized());
  EXPECT_EQ(factory_calls, 1);
}

TEST(PreparedPipelineActivatorTest, DefersUntilFirstSuccessfulCacheAllocation) {
  PreparedPipelineActivator activator;
  activator.configure(PreparedPipelineActivationStage::AFTER_CACHE_ALLOCATION);
  int32_t factory_calls = 0;
  const auto factory = [&factory_calls]() { ++factory_calls; };

  activator.finish_cache_allocation(/*success=*/false, factory);
  EXPECT_FALSE(activator.initialized());
  EXPECT_EQ(factory_calls, 0);

  activator.finish_cache_allocation(/*success=*/true, factory);
  EXPECT_TRUE(activator.initialized());
  EXPECT_EQ(factory_calls, 1);

  activator.finish_cache_allocation(/*success=*/true, factory);
  EXPECT_EQ(factory_calls, 1);
}

TEST(PreparedSpeculativeBindingContractTest,
     BuildsPredecessorRowsForEveryStatefulAdapter) {
  EXPECT_TRUE(SpeculativeConfig::requires_prepared_predecessor_rows("MTP"));
  EXPECT_TRUE(SpeculativeConfig::requires_prepared_predecessor_rows("mtp"));
  EXPECT_TRUE(SpeculativeConfig::requires_prepared_predecessor_rows("Eagle3"));
  EXPECT_TRUE(SpeculativeConfig::requires_prepared_predecessor_rows("DFlash"));
  EXPECT_TRUE(SpeculativeConfig::requires_prepared_predecessor_rows("DSpark"));
  EXPECT_FALSE(SpeculativeConfig::requires_prepared_predecessor_rows("Suffix"));
}

TEST(PreparedTaskPipelineTest, RunsPrepareLaunchConsumeInOrder) {
  auto adapter = std::make_unique<FakePreparedTaskAdapter>();
  FakePreparedTaskAdapter* adapter_ptr = adapter.get();
  PreparedTaskPipeline pipeline(std::move(adapter), /*slot_count=*/1);

  ForwardInput input = make_input(BatchForwardType::DECODE, /*batch_id=*/17);
  auto future = pipeline.submit(input);
  std::optional<ForwardOutput> output = std::move(future).get();

  ASSERT_TRUE(output.has_value());
  EXPECT_EQ(output->prepared_token, 17);
  const std::vector<AdapterOperation> operations = adapter_ptr->operations();
  ASSERT_EQ(operations.size(), 3);
  EXPECT_EQ(operations[0].phase, "prepare");
  EXPECT_EQ(operations[1].phase, "launch");
  EXPECT_EQ(operations[2].phase, "consume");
  EXPECT_EQ(operations[0].task_seq_no, 0);
  EXPECT_EQ(operations[1].task_seq_no, 0);
  EXPECT_EQ(operations[2].task_seq_no, 0);
  EXPECT_NE(operations[0].thread_id, operations[1].thread_id);
  EXPECT_EQ(operations[0].thread_id, operations[2].thread_id);
}

TEST(PreparedTaskPipelineTest, PrepareAckReleasesCallerInputBeforeCompletion) {
  auto adapter = std::make_unique<BlockingLaunchPreparedTaskAdapter>();
  BlockingLaunchPreparedTaskAdapter* adapter_ptr = adapter.get();
  PreparedTaskPipeline pipeline(std::move(adapter), /*slot_count=*/1);

  ForwardInput input = make_input(BatchForwardType::DECODE, /*batch_id=*/23);
  auto future = pipeline.submit(input);
  adapter_ptr->wait_until_launch_started();
  input.input_params.meta.batch_id = 99;
  const bool completed_before_launch_release = future.isReady();
  adapter_ptr->release_launch();

  std::optional<ForwardOutput> output = std::move(future).get();
  EXPECT_FALSE(completed_before_launch_release);
  ASSERT_TRUE(output.has_value());
  EXPECT_EQ(output->prepared_token, 23);
}

TEST(PreparedTaskPipelineTest, ReusesSingleSlotWithMonotonicSequenceNumbers) {
  auto adapter = std::make_unique<FakePreparedTaskAdapter>();
  FakePreparedTaskAdapter* adapter_ptr = adapter.get();
  PreparedTaskPipeline pipeline(std::move(adapter), /*slot_count=*/1);

  for (uint64_t batch_id = 1; batch_id <= 2; ++batch_id) {
    ForwardInput input = make_input(BatchForwardType::PREFILL, batch_id);
    auto future = pipeline.submit(input);
    ASSERT_TRUE(std::move(future).get().has_value());
  }

  const std::vector<AdapterOperation> operations = adapter_ptr->operations();
  ASSERT_EQ(operations.size(), 6);
  for (size_t operation_index = 0; operation_index < operations.size();
       ++operation_index) {
    EXPECT_EQ(operations[operation_index].task_seq_no,
              static_cast<uint64_t>(operation_index / 3));
  }
}

TEST(PreparedTaskPipelineTest, UnpacksTransportPayloadBeforePrepare) {
  ForwardInput input = make_input(BatchForwardType::DECODE, /*batch_id=*/8);
  input.token_ids = torch::tensor({17, 19}, torch::kInt);
  input.positions = torch::tensor({3, 4}, torch::kInt);

  proto::PackedForwardInput packed_input;
  ASSERT_TRUE(forward_input_to_packed_proto(input, &packed_input));
  ForwardInput transport_input;
  packed_proto_to_forward_input(packed_input,
                                transport_input,
                                torch::Device(torch::kCPU),
                                /*stream=*/nullptr);
  ASSERT_TRUE(transport_input.input_host_buffer_has_layout);
  EXPECT_TRUE(transport_input.input_params.meta.batch_forward_type.is_empty());

  auto adapter = std::make_unique<FakePreparedTaskAdapter>();
  PreparedTaskPipeline pipeline(std::move(adapter), /*slot_count=*/1);
  std::optional<ForwardOutput> output =
      std::move(pipeline.submit(transport_input)).get();

  ASSERT_TRUE(output.has_value());
  EXPECT_EQ(output->prepared_token, 8);
}

TEST(PreparedTaskPipelineTest, QuiesceAndResumeReuseTheSamePipeline) {
  auto adapter = std::make_unique<FakePreparedTaskAdapter>();
  PreparedTaskPipeline pipeline(std::move(adapter), /*slot_count=*/1);

  pipeline.quiesce();
  EXPECT_EQ(pipeline.lifecycle(), PreparedPipelineLifecycle::QUIESCENT);
  pipeline.resume();
  EXPECT_EQ(pipeline.lifecycle(), PreparedPipelineLifecycle::RUNNING);

  ForwardInput input = make_input(BatchForwardType::EMPTY, /*batch_id=*/9);
  auto future = pipeline.submit(input);
  std::optional<ForwardOutput> output = std::move(future).get();
  ASSERT_TRUE(output.has_value());
  EXPECT_EQ(output->prepared_token, 9);
}

TEST(PreparedTaskPipelineTest, TwoSlotPrepareOverlapsPreviousLaunch) {
  auto adapter = std::make_unique<BlockingLaunchPreparedTaskAdapter>();
  BlockingLaunchPreparedTaskAdapter* adapter_ptr = adapter.get();
  PreparedTaskPipeline pipeline(std::move(adapter), /*slot_count=*/2);

  ForwardInput first_input =
      make_input(BatchForwardType::DECODE, /*batch_id=*/31);
  auto first_submit = pipeline.submit(first_input);
  EXPECT_FALSE(std::move(first_submit).get().has_value());
  adapter_ptr->wait_until_launch_started();

  ForwardInput second_input =
      make_input(BatchForwardType::DECODE, /*batch_id=*/32);
  auto second_submit = pipeline.submit(second_input);
  EXPECT_FALSE(std::move(second_submit).get().has_value());
  second_input.input_params.meta.batch_id = 99;

  adapter_ptr->release_launch();
  std::optional<ForwardOutput> first_output =
      std::move(pipeline.get_last_step_result()).get();
  std::optional<ForwardOutput> second_output =
      std::move(pipeline.get_last_step_result()).get();
  ASSERT_TRUE(first_output.has_value());
  ASSERT_TRUE(second_output.has_value());
  EXPECT_EQ(first_output->prepared_token, 31);
  EXPECT_EQ(second_output->prepared_token, 32);
}

TEST(PreparedTaskPipelineTest,
     RecordsStageLatencyAndReadySuccessorObservability) {
  const int64_t prepare_count_before =
      HISTOGRAM_prepared_task_prepare_cpu_latency_microseconds.count();
  const int64_t launch_count_before =
      HISTOGRAM_prepared_task_launch_submission_latency_microseconds.count();
  const int64_t consume_count_before =
      HISTOGRAM_prepared_task_consume_latency_microseconds.count();
  const double launch_submissions_before =
      COUNTER_VALUE(prepared_task_launch_submissions_total);
  const double empty_at_launch_end_before =
      COUNTER_VALUE(prepared_task_ready_queue_empty_at_launch_end_total);

  auto adapter = std::make_unique<BlockingLaunchPreparedTaskAdapter>();
  BlockingLaunchPreparedTaskAdapter* adapter_ptr = adapter.get();
  PreparedTaskPipeline pipeline(std::move(adapter), /*slot_count=*/2);

  ForwardInput first_input =
      make_input(BatchForwardType::DECODE, /*batch_id=*/33);
  EXPECT_FALSE(std::move(pipeline.submit(first_input)).get().has_value());
  adapter_ptr->wait_until_launch_started();

  ForwardInput second_input =
      make_input(BatchForwardType::DECODE, /*batch_id=*/34);
  EXPECT_FALSE(std::move(pipeline.submit(second_input)).get().has_value());
  adapter_ptr->release_launch();

  for (int64_t expected_token = 33; expected_token <= 34; ++expected_token) {
    std::optional<ForwardOutput> output =
        std::move(pipeline.get_last_step_result()).get();
    ASSERT_TRUE(output.has_value());
    EXPECT_EQ(output->prepared_token, expected_token);
  }

  EXPECT_EQ(HISTOGRAM_prepared_task_prepare_cpu_latency_microseconds.count() -
                prepare_count_before,
            2);
  EXPECT_EQ(
      HISTOGRAM_prepared_task_launch_submission_latency_microseconds.count() -
          launch_count_before,
      2);
  EXPECT_EQ(HISTOGRAM_prepared_task_consume_latency_microseconds.count() -
                consume_count_before,
            2);
  EXPECT_EQ(COUNTER_VALUE(prepared_task_launch_submissions_total) -
                launch_submissions_before,
            2);
  // Task 1 ends Launch submission with Task 2 already READY. Task 2 has no
  // successor, so exactly one of the two submissions observes an empty queue.
  EXPECT_EQ(COUNTER_VALUE(prepared_task_ready_queue_empty_at_launch_end_total) -
                empty_at_launch_end_before,
            1);
}

TEST(PreparedTaskPipelineTest, TwoSlotConsumesOutputsInFifoAndReusesSlots) {
  auto adapter = std::make_unique<FakePreparedTaskAdapter>();
  FakePreparedTaskAdapter* adapter_ptr = adapter.get();
  PreparedTaskPipeline pipeline(std::move(adapter), /*slot_count=*/2);

  for (uint64_t batch_id = 41; batch_id <= 42; ++batch_id) {
    ForwardInput input = make_input(BatchForwardType::DECODE, batch_id);
    auto submit_result = pipeline.submit(input);
    EXPECT_FALSE(std::move(submit_result).get().has_value());
  }
  std::optional<ForwardOutput> first_output =
      std::move(pipeline.get_last_step_result()).get();
  ASSERT_TRUE(first_output.has_value());
  EXPECT_EQ(first_output->prepared_token, 41);

  ForwardInput third_input =
      make_input(BatchForwardType::DECODE, /*batch_id=*/43);
  auto third_submit = pipeline.submit(third_input);
  EXPECT_FALSE(std::move(third_submit).get().has_value());

  for (int64_t expected_token = 42; expected_token <= 43; ++expected_token) {
    std::optional<ForwardOutput> output =
        std::move(pipeline.get_last_step_result()).get();
    ASSERT_TRUE(output.has_value());
    EXPECT_EQ(output->prepared_token, expected_token);
  }

  const std::vector<AdapterOperation> operations = adapter_ptr->operations();
  ASSERT_EQ(operations.size(), 9);
  auto second_prepare = std::find_if(
      operations.begin(), operations.end(), [](const AdapterOperation& item) {
        return item.phase == "prepare" && item.task_seq_no == 1;
      });
  auto first_consume = std::find_if(
      operations.begin(), operations.end(), [](const AdapterOperation& item) {
        return item.phase == "consume" && item.task_seq_no == 0;
      });
  auto third_prepare = std::find_if(
      operations.begin(), operations.end(), [](const AdapterOperation& item) {
        return item.phase == "prepare" && item.task_seq_no == 2;
      });
  ASSERT_NE(second_prepare, operations.end());
  ASSERT_NE(first_consume, operations.end());
  ASSERT_NE(third_prepare, operations.end());
  EXPECT_EQ(second_prepare->slot_id, 1);
  EXPECT_EQ(first_consume->slot_id, 0);
  EXPECT_EQ(third_prepare->slot_id, 0);
  EXPECT_TRUE(second_prepare < first_consume);
}

TEST(PreparedTaskPipelineTest, WaitsForPredecessorReadFenceBeforeReusingSlot) {
  auto adapter = std::make_unique<ReuseFencePreparedTaskAdapter>();
  ReuseFencePreparedTaskAdapter* adapter_ptr = adapter.get();
  PreparedTaskPipeline pipeline(std::move(adapter), /*slot_count=*/2);

  ForwardInput first_input =
      make_input(BatchForwardType::DECODE, /*batch_id=*/51);
  EXPECT_FALSE(std::move(pipeline.submit(first_input)).get().has_value());

  ForwardInput second_input =
      make_input(BatchForwardType::DECODE, /*batch_id=*/52);
  second_input.input_params.embedding.predecessor_rows =
      torch::tensor({0}, torch::kLong);
  EXPECT_FALSE(std::move(pipeline.submit(second_input)).get().has_value());
  adapter_ptr->wait_until_predecessor_read_started();

  auto first_result = pipeline.get_last_step_result();
  adapter_ptr->wait_until_first_consume_started();
  EXPECT_FALSE(first_result.isReady());

  adapter_ptr->release_predecessor_read();
  std::optional<ForwardOutput> first_output = std::move(first_result).get();
  ASSERT_TRUE(first_output.has_value());
  EXPECT_EQ(first_output->prepared_token, 51);

  ForwardInput third_input =
      make_input(BatchForwardType::DECODE, /*batch_id=*/53);
  EXPECT_FALSE(std::move(pipeline.submit(third_input)).get().has_value());
  EXPECT_TRUE(adapter_ptr->reuse_wait_observed());

  for (int64_t expected_token = 52; expected_token <= 53; ++expected_token) {
    std::optional<ForwardOutput> output =
        std::move(pipeline.get_last_step_result()).get();
    ASSERT_TRUE(output.has_value());
    EXPECT_EQ(output->prepared_token, expected_token);
  }
}

TEST(PreparedTaskPipelineTest,
     ReadsPredecessorStateAfterPredecessorSlotIsFree) {
  auto adapter = std::make_unique<TrackingPredecessorPreparedTaskAdapter>();
  TrackingPredecessorPreparedTaskAdapter* adapter_ptr = adapter.get();
  PreparedTaskPipeline pipeline(std::move(adapter), /*slot_count=*/2);

  ForwardInput first_input =
      make_input(BatchForwardType::DECODE, /*batch_id=*/61);
  EXPECT_FALSE(std::move(pipeline.submit(first_input)).get().has_value());
  std::optional<ForwardOutput> first_output =
      std::move(pipeline.get_last_step_result()).get();
  ASSERT_TRUE(first_output.has_value());
  EXPECT_EQ(first_output->prepared_token, 61);

  ForwardInput second_input =
      make_input(BatchForwardType::DECODE, /*batch_id=*/62);
  second_input.input_params.embedding.predecessor_rows =
      torch::tensor({0}, torch::kLong);
  EXPECT_FALSE(std::move(pipeline.submit(second_input)).get().has_value());
  std::optional<ForwardOutput> second_output =
      std::move(pipeline.get_last_step_result()).get();
  ASSERT_TRUE(second_output.has_value());
  EXPECT_EQ(second_output->prepared_token, 62);

  const std::vector<TrackingPredecessorPreparedTaskAdapter::PredecessorRead>
      reads = adapter_ptr->predecessor_reads();
  ASSERT_EQ(reads.size(), 1);
  EXPECT_EQ(reads[0].task_seq_no, 1);
  EXPECT_EQ(reads[0].slot_id, 1);
  EXPECT_EQ(reads[0].predecessor_slot_id, 0);
  EXPECT_EQ(reads[0].predecessor_state, ExecutionSlotState::FREE);
}

TEST(PreparedTaskPipelineTest, SupportsSingleSlotSelfPredecessor) {
  auto adapter = std::make_unique<TrackingPredecessorPreparedTaskAdapter>();
  TrackingPredecessorPreparedTaskAdapter* adapter_ptr = adapter.get();
  PreparedTaskPipeline pipeline(std::move(adapter), /*slot_count=*/1);

  ForwardInput first_input =
      make_input(BatchForwardType::DECODE, /*batch_id=*/71);
  ASSERT_TRUE(std::move(pipeline.submit(first_input)).get().has_value());

  ForwardInput second_input =
      make_input(BatchForwardType::DECODE, /*batch_id=*/72);
  second_input.input_params.embedding.predecessor_rows =
      torch::tensor({0}, torch::kLong);
  std::optional<ForwardOutput> second_output =
      std::move(pipeline.submit(second_input)).get();
  ASSERT_TRUE(second_output.has_value());
  EXPECT_EQ(second_output->prepared_token, 72);

  const std::vector<TrackingPredecessorPreparedTaskAdapter::PredecessorRead>
      reads = adapter_ptr->predecessor_reads();
  ASSERT_EQ(reads.size(), 1);
  EXPECT_EQ(reads[0].task_seq_no, 1);
  EXPECT_EQ(reads[0].slot_id, 0);
  EXPECT_EQ(reads[0].predecessor_slot_id, 0);
  EXPECT_EQ(reads[0].predecessor_state, ExecutionSlotState::RUNNING);
}

TEST(PreparedTaskPipelineTest, PrefillAndEmptyTasksDoNotReadPredecessorState) {
  auto adapter = std::make_unique<TrackingPredecessorPreparedTaskAdapter>();
  TrackingPredecessorPreparedTaskAdapter* adapter_ptr = adapter.get();
  PreparedTaskPipeline pipeline(std::move(adapter), /*slot_count=*/1);

  const std::vector<BatchForwardType> non_consuming_types = {
      BatchForwardType::PREFILL, BatchForwardType::EMPTY};
  uint64_t batch_id = 81;
  for (const BatchForwardType forward_type : non_consuming_types) {
    ForwardInput input = make_input(forward_type, batch_id++);
    input.input_params.embedding.predecessor_rows =
        torch::tensor({0}, torch::kLong);
    ASSERT_TRUE(std::move(pipeline.submit(input)).get().has_value());
  }

  EXPECT_TRUE(adapter_ptr->predecessor_reads().empty());
}

TEST(PreparedInputLayoutTest, SignatureTracksStableOffsetsAndShapes) {
  torch::Tensor first_target;
  torch::Tensor second_target;
  detail::ForwardInputBufferPlan first_plan;
  ASSERT_TRUE(first_plan.add(torch::ones({2}, torch::kInt32), &first_target));
  ASSERT_TRUE(first_plan.add(torch::zeros({3}, torch::kInt64), &second_target));
  first_plan.prepare_layout();

  torch::Tensor matching_first_target;
  torch::Tensor matching_second_target;
  detail::ForwardInputBufferPlan matching_plan;
  ASSERT_TRUE(matching_plan.add(torch::zeros({2}, torch::kInt32),
                                &matching_first_target));
  ASSERT_TRUE(matching_plan.add(torch::ones({3}, torch::kInt64),
                                &matching_second_target));
  matching_plan.prepare_layout();
  EXPECT_EQ(first_plan.layout_signature(), matching_plan.layout_signature());

  torch::Tensor changed_target;
  detail::ForwardInputBufferPlan changed_plan;
  ASSERT_TRUE(
      changed_plan.add(torch::zeros({4}, torch::kInt32), &changed_target));
  changed_plan.prepare_layout();
  EXPECT_NE(first_plan.layout_signature(), changed_plan.layout_signature());
}

TEST(PreparedInputLayoutTest, RequiresPerFieldDeviceSourceDeclaration) {
  const torch::Tensor device_source = torch::empty(
      {2},
      torch::TensorOptions().dtype(torch::kInt).device(torch::Device("meta")));
  torch::Tensor target;
  detail::ForwardInputBufferPlan implicit_plan;
  EXPECT_FALSE(implicit_plan.add(device_source, &target));
  EXPECT_TRUE(implicit_plan.entries.empty());

  detail::ForwardInputBufferPlan explicit_plan;
  EXPECT_TRUE(explicit_plan.add_external_device_source(device_source, &target));
  EXPECT_EQ(explicit_plan.entries.size(), 1);
  EXPECT_EQ(explicit_plan.device_source_count(), 1);
  EXPECT_EQ(explicit_plan.device_source_bytes(),
            static_cast<uint64_t>(device_source.numel() *
                                  device_source.element_size()));
}

TEST(PreparedInputLayoutTest, HostCopiesSkipExternalDeviceSourceRanges) {
  torch::Tensor first_target;
  torch::Tensor device_target;
  torch::Tensor last_target;
  detail::ForwardInputBufferPlan plan;
  ASSERT_TRUE(plan.add(torch::tensor({1, 2}, torch::kInt), &first_target));
  const torch::Tensor device_source = torch::empty(
      {4},
      torch::TensorOptions().dtype(torch::kInt).device(torch::Device("meta")));
  ASSERT_TRUE(plan.add_external_device_source(device_source, &device_target));
  ASSERT_TRUE(plan.add(torch::tensor({3}, torch::kLong), &last_target));
  const uint64_t total_bytes = plan.prepare_layout();
  ASSERT_EQ(total_bytes, 48);

  const torch::Tensor host_buffer = plan.build_host_buffer(total_bytes);
  torch::Tensor destination = torch::full(
      {static_cast<int64_t>(total_bytes)},
      0x7f,
      torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU));
  const detail::ForwardInputHostCopyStats stats =
      plan.copy_host_sources(host_buffer, destination);
  EXPECT_EQ(stats.bytes, 32);
  EXPECT_EQ(stats.copies, 2);
  EXPECT_TRUE(torch::equal(destination.narrow(/*dim=*/0,
                                              /*start=*/0,
                                              /*length=*/16),
                           host_buffer.narrow(/*dim=*/0,
                                              /*start=*/0,
                                              /*length=*/16)));
  EXPECT_TRUE(torch::all(destination.narrow(/*dim=*/0,
                                            /*start=*/16,
                                            /*length=*/16) == 0x7f)
                  .item<bool>());
  EXPECT_TRUE(torch::equal(destination.narrow(/*dim=*/0,
                                              /*start=*/32,
                                              /*length=*/16),
                           host_buffer.narrow(/*dim=*/0,
                                              /*start=*/32,
                                              /*length=*/16)));
}

TEST(PreparedInputArenaTest, RejectsUndeclaredDeviceMetadataSources) {
  PreparedInputArena arena(torch::Device(torch::kCPU),
                           /*capacity_bytes=*/4096);
  ForwardInput input = make_input(BatchForwardType::DECODE, /*batch_id=*/84);
  input.token_ids_host = torch::tensor({1, 2}, cpu_tensor_options(torch::kInt));
  input.positions_host =
      torch::tensor({10, 11}, cpu_tensor_options(torch::kInt));
  input.sampling_params.selected_token_idxes = torch::empty(
      {2},
      torch::TensorOptions().dtype(torch::kInt).device(torch::Device("meta")));

  ForwardInput staged;
  EXPECT_FALSE(arena.stage(input, staged));
}

TEST(PreparedInputArenaTest, ReleasesReboundAttentionPackedBuffers) {
  PreparedInputArena arena(torch::Device(torch::kCPU),
                           /*capacity_bytes=*/4096);
  ForwardInput input = make_input(BatchForwardType::DECODE, /*batch_id=*/85);
  input.token_ids_host = torch::tensor({1, 2}, cpu_tensor_options(torch::kInt));
  input.positions_host =
      torch::tensor({10, 11}, cpu_tensor_options(torch::kInt));
  AttentionInput& attention = input.input_params.attention;
  attention.attention_host_buffer = torch::zeros({64}, torch::kUInt8);
  attention.attention_device_buffer = torch::zeros({64}, torch::kUInt8);
  attention.attention_buffer_bytes = 32;
  attention.attention_buffer_capacity = 64;

  ForwardInput staged;
  ASSERT_TRUE(arena.stage(input, staged));
  const AttentionInput& staged_attention = staged.input_params.attention;
  EXPECT_FALSE(staged_attention.attention_host_buffer.defined());
  EXPECT_FALSE(staged_attention.attention_device_buffer.defined());
  EXPECT_EQ(staged_attention.attention_buffer_bytes, 0);
  EXPECT_EQ(staged_attention.attention_buffer_capacity, 0);
}

TEST(PreparedInputArenaTest, RetainsPackedBufferOwningExternalHistoryState) {
  PreparedInputArena arena(torch::Device(torch::kCPU),
                           /*capacity_bytes=*/4096);
  ForwardInput input = make_input(BatchForwardType::DECODE, /*batch_id=*/86);
  input.token_ids_host = torch::tensor({1, 2}, cpu_tensor_options(torch::kInt));
  input.positions_host =
      torch::tensor({10, 11}, cpu_tensor_options(torch::kInt));
  AttentionInput& attention = input.input_params.attention;
  attention.attention_host_buffer = torch::zeros({64}, torch::kUInt8);
  attention.attention_device_buffer = torch::zeros({16}, torch::kFloat);
  attention.attention_buffer_bytes = 64;
  attention.attention_buffer_capacity = 64;
  attention.device.history_compressed_kv =
      attention.attention_device_buffer.narrow(/*dim=*/0,
                                               /*start=*/2,
                                               /*length=*/4);
  const void* history_address =
      attention.device.history_compressed_kv.data_ptr();

  ForwardInput staged;
  ASSERT_TRUE(arena.stage(input, staged));
  const AttentionInput& staged_attention = staged.input_params.attention;
  EXPECT_FALSE(staged_attention.attention_host_buffer.defined());
  ASSERT_TRUE(staged_attention.attention_device_buffer.defined());
  EXPECT_EQ(staged_attention.device.history_compressed_kv.data_ptr(),
            history_address);
}

TEST(PreparedInputArenaTest, PartitionsOneSlotAcrossFixedInvocations) {
  PreparedInputArena arena(torch::Device(torch::kCPU),
                           /*capacity_bytes=*/4096);
  ForwardInput first_input =
      make_input(BatchForwardType::DECODE, /*batch_id=*/87);
  first_input.token_ids_host =
      torch::tensor({1, 2}, cpu_tensor_options(torch::kInt));
  first_input.positions_host =
      torch::tensor({10, 11}, cpu_tensor_options(torch::kInt));
  ForwardInput second_input = first_input;
  second_input.token_ids_host =
      torch::tensor({3, 4, 5}, cpu_tensor_options(torch::kInt));
  second_input.positions_host =
      torch::tensor({20, 21, 22}, cpu_tensor_options(torch::kInt));

  ForwardInput first_staged;
  ForwardInput second_staged;
  arena.begin_partitioned_task(/*partition_count=*/2);
  ASSERT_TRUE(arena.stage_next(first_input, first_staged));
  ASSERT_TRUE(arena.stage_next(second_input, second_staged));
  ASSERT_TRUE(first_staged.token_ids.defined());
  ASSERT_TRUE(second_staged.token_ids.defined());
  EXPECT_NE(first_staged.token_ids.data_ptr(),
            second_staged.token_ids.data_ptr());
  EXPECT_GT(arena.used_bytes(), 0);

  const void* first_token_address = first_staged.token_ids.data_ptr();
  const void* second_token_address = second_staged.token_ids.data_ptr();
  first_input.token_ids_host =
      torch::tensor({6, 7, 8, 9}, cpu_tensor_options(torch::kInt));
  first_input.positions_host =
      torch::tensor({30, 31, 32, 33}, cpu_tensor_options(torch::kInt));
  ForwardInput first_staged_again;
  ForwardInput second_staged_again;
  arena.begin_partitioned_task(/*partition_count=*/2);
  ASSERT_TRUE(arena.stage_next(first_input, first_staged_again));
  ASSERT_TRUE(arena.stage_next(second_input, second_staged_again));
  EXPECT_EQ(first_staged_again.token_ids.data_ptr(), first_token_address);
  EXPECT_EQ(second_staged_again.token_ids.data_ptr(), second_token_address);
}

TEST(PreparedInputArenaTest,
     StagesGraphTilingAtStableSpeculativePartitionAddresses) {
  PreparedInputArena arena(torch::Device(torch::kCPU),
                           /*capacity_bytes=*/4096);
  ForwardInput first_input =
      make_input(BatchForwardType::DECODE, /*batch_id=*/85);
  first_input.token_ids_host =
      torch::tensor({1, 2}, cpu_tensor_options(torch::kInt));
  first_input.positions_host =
      torch::tensor({10, 11}, cpu_tensor_options(torch::kInt));
  first_input.input_params.graph.tiling_data =
      torch::tensor({21, 22, 23}, cpu_tensor_options(torch::kInt));
  ForwardInput second_input = first_input;
  second_input.input_params.graph.tiling_data =
      torch::tensor({31, 32, 33}, cpu_tensor_options(torch::kInt));

  ForwardInput first_staged;
  ForwardInput second_staged;
  arena.begin_partitioned_task(/*partition_count=*/2);
  ASSERT_TRUE(arena.stage_next(first_input, first_staged));
  ASSERT_TRUE(arena.stage_next(second_input, second_staged));
  const void* first_tiling_address =
      first_staged.input_params.graph.tiling_data.data_ptr();
  const void* second_tiling_address =
      second_staged.input_params.graph.tiling_data.data_ptr();
  EXPECT_NE(first_tiling_address,
            first_input.input_params.graph.tiling_data.data_ptr());
  EXPECT_NE(second_tiling_address,
            second_input.input_params.graph.tiling_data.data_ptr());
  EXPECT_NE(first_tiling_address, second_tiling_address);
  EXPECT_EQ(first_staged.prepared_arena_d2d_bytes, 0);
  EXPECT_EQ(second_staged.prepared_arena_d2d_bytes, 0);

  first_input.input_params.graph.tiling_data =
      torch::tensor({41, 42, 43}, cpu_tensor_options(torch::kInt));
  second_input.input_params.graph.tiling_data =
      torch::tensor({51, 52, 53}, cpu_tensor_options(torch::kInt));
  ForwardInput first_staged_again;
  ForwardInput second_staged_again;
  arena.begin_partitioned_task(/*partition_count=*/2);
  ASSERT_TRUE(arena.stage_next(first_input, first_staged_again));
  ASSERT_TRUE(arena.stage_next(second_input, second_staged_again));
  EXPECT_EQ(first_staged_again.input_params.graph.tiling_data.data_ptr(),
            first_tiling_address);
  EXPECT_EQ(second_staged_again.input_params.graph.tiling_data.data_ptr(),
            second_tiling_address);
  EXPECT_TRUE(torch::equal(
      first_staged_again.input_params.graph.tiling_data,
      torch::tensor({41, 42, 43}, cpu_tensor_options(torch::kInt))));
  EXPECT_TRUE(torch::equal(
      second_staged_again.input_params.graph.tiling_data,
      torch::tensor({51, 52, 53}, cpu_tensor_options(torch::kInt))));
}

TEST(PreparedInputArenaTest,
     PrefersHostAttentionMetadataOverTemporaryDeviceSources) {
  PreparedInputArena arena(torch::Device(torch::kCPU),
                           /*capacity_bytes=*/4096);
  ForwardInput input = make_input(BatchForwardType::DECODE, /*batch_id=*/86);
  input.token_ids_host = torch::tensor({1, 2}, cpu_tensor_options(torch::kInt));
  input.positions_host =
      torch::tensor({10, 11}, cpu_tensor_options(torch::kInt));
  AttentionInput& attention = input.input_params.attention;
  attention.host.q_seq_lens = {1, 1};
  attention.host.q_cu_seq_lens = {1, 2};
  attention.host.kv_seq_lens = {11, 12};
  attention.host.new_cache_slots = {21, 22};
  attention.host.kv_cache_tokens_nums = {31, 32};
  attention.host.ring_cur_seqlen = {41, 42};
  attention.host.ring_cache_seqlen = {51, 52};
  attention.host.block_tables =
      torch::tensor({{61, 62}, {63, 64}}, cpu_tensor_options(torch::kInt));

  attention.device.q_seq_lens =
      torch::tensor({101, 102}, cpu_tensor_options(torch::kInt));
  attention.device.q_cu_seq_lens =
      torch::tensor({103, 104}, cpu_tensor_options(torch::kInt));
  attention.device.kv_seq_lens =
      torch::tensor({105, 106}, cpu_tensor_options(torch::kInt));
  attention.device.new_cache_slots =
      torch::tensor({107, 108}, cpu_tensor_options(torch::kInt));
  attention.device.kv_cache_tokens_nums =
      torch::tensor({109, 110}, cpu_tensor_options(torch::kInt));
  attention.device.ring_cur_seqlen =
      torch::tensor({111, 112}, cpu_tensor_options(torch::kInt));
  attention.device.ring_cache_seqlen =
      torch::tensor({113, 114}, cpu_tensor_options(torch::kInt));
  attention.device.block_tables =
      torch::tensor({{115, 116}, {117, 118}}, cpu_tensor_options(torch::kInt));

  ForwardInput staged;
  ASSERT_TRUE(arena.stage(input, staged));
  EXPECT_TRUE(
      torch::equal(staged.input_params.attention.device.q_seq_lens,
                   torch::tensor({1, 1}, cpu_tensor_options(torch::kInt))));
  EXPECT_TRUE(
      torch::equal(staged.input_params.attention.device.q_cu_seq_lens,
                   torch::tensor({1, 2}, cpu_tensor_options(torch::kInt))));
  EXPECT_TRUE(
      torch::equal(staged.input_params.attention.device.kv_seq_lens,
                   torch::tensor({11, 12}, cpu_tensor_options(torch::kInt))));
  EXPECT_TRUE(
      torch::equal(staged.input_params.attention.device.new_cache_slots,
                   torch::tensor({21, 22}, cpu_tensor_options(torch::kInt))));
  EXPECT_TRUE(
      torch::equal(staged.input_params.attention.device.kv_cache_tokens_nums,
                   torch::tensor({31, 32}, cpu_tensor_options(torch::kInt))));
  EXPECT_TRUE(
      torch::equal(staged.input_params.attention.device.ring_cur_seqlen,
                   torch::tensor({41, 42}, cpu_tensor_options(torch::kInt))));
  EXPECT_TRUE(
      torch::equal(staged.input_params.attention.device.ring_cache_seqlen,
                   torch::tensor({51, 52}, cpu_tensor_options(torch::kInt))));
  EXPECT_TRUE(torch::equal(
      staged.input_params.attention.device.block_tables,
      torch::tensor({{61, 62}, {63, 64}}, cpu_tensor_options(torch::kInt))));
}

TEST(PreparedInputArenaTest,
     StagesModelManagedBlockTablesInStableHostArenaStorage) {
  PreparedInputArena arena(torch::Device(torch::kCPU),
                           /*capacity_bytes=*/4096);
  ForwardInput input = make_input(BatchForwardType::DECODE, /*batch_id=*/99);
  input.token_ids_host = torch::tensor({1, 2}, cpu_tensor_options(torch::kInt));
  input.positions_host =
      torch::tensor({10, 11}, cpu_tensor_options(torch::kInt));

  ForwardInput baseline_staged;
  ASSERT_TRUE(arena.stage(input, baseline_staged));
  const uint64_t baseline_h2d_bytes = baseline_staged.prepared_arena_h2d_bytes;
  const int32_t baseline_h2d_copies = baseline_staged.prepared_arena_h2d_copies;
  const int64_t baseline_buffer_bytes =
      baseline_staged.input_host_buffer.numel();

  input.input_params.multi_block_tables = {
      torch::tensor({{10, 11}, {12, 13}}, cpu_tensor_options(torch::kInt)),
      torch::tensor({{20, 21, 22}, {23, 24, 25}},
                    cpu_tensor_options(torch::kInt))};
  const void* first_source_address =
      input.input_params.multi_block_tables[0].data_ptr();
  ForwardInput first_staged;
  ASSERT_TRUE(arena.stage(input, first_staged));
  ASSERT_EQ(first_staged.input_params.multi_block_tables.size(), 2);
  EXPECT_TRUE(
      first_staged.input_params.multi_block_tables[0].device().is_cpu());
  EXPECT_NE(first_staged.input_params.multi_block_tables[0].data_ptr(),
            first_source_address);
  EXPECT_TRUE(torch::equal(
      first_staged.input_params.multi_block_tables[0],
      torch::tensor({{10, 11}, {12, 13}}, cpu_tensor_options(torch::kInt))));
  EXPECT_TRUE(torch::equal(first_staged.input_params.multi_block_tables[1],
                           torch::tensor({{20, 21, 22}, {23, 24, 25}},
                                         cpu_tensor_options(torch::kInt))));
  EXPECT_EQ(first_staged.prepared_arena_h2d_bytes, baseline_h2d_bytes);
  EXPECT_EQ(first_staged.prepared_arena_h2d_copies, baseline_h2d_copies);
  EXPECT_GT(first_staged.input_host_buffer.numel(), baseline_buffer_bytes);

  input.input_params.multi_block_tables[0].fill_(99);
  EXPECT_TRUE(torch::equal(
      first_staged.input_params.multi_block_tables[0],
      torch::tensor({{10, 11}, {12, 13}}, cpu_tensor_options(torch::kInt))));
  const void* first_staged_address =
      first_staged.input_params.multi_block_tables[0].data_ptr();
  input.input_params.multi_block_tables = {
      torch::tensor({{30, 31}, {32, 33}}, cpu_tensor_options(torch::kInt)),
      torch::tensor({{40, 41, 42}, {43, 44, 45}},
                    cpu_tensor_options(torch::kInt))};
  ForwardInput second_staged;
  ASSERT_TRUE(arena.stage(input, second_staged));
  EXPECT_EQ(second_staged.input_params.multi_block_tables[0].data_ptr(),
            first_staged_address);
  EXPECT_TRUE(torch::equal(
      second_staged.input_params.multi_block_tables[0],
      torch::tensor({{30, 31}, {32, 33}}, cpu_tensor_options(torch::kInt))));
}

TEST(PreparedInputArenaTest, ReusesCallerOwnedMetadataVectorCapacity) {
  PreparedInputArena arena(torch::Device(torch::kCPU),
                           /*capacity_bytes=*/4096);
  ForwardInput input = make_input(BatchForwardType::DECODE, /*batch_id=*/96);
  input.token_ids_host = torch::tensor({1, 2}, cpu_tensor_options(torch::kInt));
  input.positions_host =
      torch::tensor({10, 11}, cpu_tensor_options(torch::kInt));
  input.input_params.attention.host.q_seq_lens = {1, 1};
  input.input_params.attention.host.q_cu_seq_lens = {1, 2};
  input.input_params.attention.host.kv_seq_lens = {11, 12};
  input.input_params.attention.host.new_cache_slots = {21, 22};

  ForwardInput staged;
  ASSERT_TRUE(arena.stage(input, staged));
  const int32_t* q_seq_lens_address =
      staged.input_params.attention.host.q_seq_lens.data();
  const int32_t* kv_seq_lens_address =
      staged.input_params.attention.host.kv_seq_lens.data();
  const int32_t* cache_slots_address =
      staged.input_params.attention.host.new_cache_slots.data();

  input.input_params.attention.host.q_seq_lens = {2, 2};
  input.input_params.attention.host.q_cu_seq_lens = {2, 4};
  input.input_params.attention.host.kv_seq_lens = {13, 14};
  input.input_params.attention.host.new_cache_slots = {23, 24};
  ASSERT_TRUE(arena.stage(input, staged));

  EXPECT_EQ(staged.input_params.attention.host.q_seq_lens.data(),
            q_seq_lens_address);
  EXPECT_EQ(staged.input_params.attention.host.kv_seq_lens.data(),
            kv_seq_lens_address);
  EXPECT_EQ(staged.input_params.attention.host.new_cache_slots.data(),
            cache_slots_address);
  EXPECT_EQ(staged.input_params.attention.host.q_seq_lens,
            (std::vector<int32_t>{2, 2}));
  EXPECT_EQ(staged.input_params.attention.host.kv_seq_lens,
            (std::vector<int32_t>{13, 14}));
  EXPECT_EQ(staged.input_params.attention.host.new_cache_slots,
            (std::vector<int32_t>{23, 24}));
}

TEST(PreparedInputArenaTest, ClearsReusedTensorViewsForEmptyInput) {
  PreparedInputArena arena(torch::Device(torch::kCPU),
                           /*capacity_bytes=*/4096);
  ForwardInput decode_input =
      make_input(BatchForwardType::DECODE, /*batch_id=*/97);
  decode_input.token_ids_host =
      torch::tensor({1, 2}, cpu_tensor_options(torch::kInt));
  decode_input.positions_host =
      torch::tensor({10, 11}, cpu_tensor_options(torch::kInt));

  ForwardInput staged;
  ASSERT_TRUE(arena.stage(decode_input, staged));
  ASSERT_TRUE(staged.token_ids.defined());
  ASSERT_TRUE(staged.device_input_buffer.defined());

  const ForwardInput empty_input =
      make_input(BatchForwardType::EMPTY, /*batch_id=*/98);
  ASSERT_TRUE(arena.stage(empty_input, staged));
  EXPECT_FALSE(staged.token_ids.defined());
  EXPECT_FALSE(staged.positions.defined());
  EXPECT_FALSE(staged.input_host_buffer.defined());
  EXPECT_FALSE(staged.device_input_buffer.defined());
}

TEST(PreparedInputArenaTest, StagesSupplementalModelMetadataAtStableAddresses) {
  PreparedInputArena arena(torch::Device(torch::kCPU),
                           /*capacity_bytes=*/8192);
  ForwardInput input = make_input(BatchForwardType::DECODE, /*batch_id=*/87);
  input.token_ids_host = torch::tensor({1, 2}, cpu_tensor_options(torch::kInt));
  input.positions_host =
      torch::tensor({10, 11}, cpu_tensor_options(torch::kInt));
  input.input_params.embedding.mtp_shifted_token_ids =
      torch::tensor({3, 4}, cpu_tensor_options(torch::kInt));
  input.input_params.parallel.dp_ep_padding_data.attn_padding_idx() =
      torch::tensor({5, 6}, cpu_tensor_options(torch::kInt));
  input.input_params.parallel.dp_ep_padding_data.expert_array() =
      torch::tensor({7, 8}, cpu_tensor_options(torch::kInt));
  input.input_params.expert.eplb_decode_token_mask =
      torch::tensor({true, false}, cpu_tensor_options(torch::kBool));
  input.input_params.graph.expanded_kv_seq_lens_vec = {9, 10};
  input.input_params.graph.expanded_kv_seq_lens =
      torch::tensor({109, 110}, cpu_tensor_options(torch::kInt));
  input.input_params.graph.expanded_block_tables =
      torch::tensor({{11, 12}, {13, 14}}, cpu_tensor_options(torch::kInt));
  input.input_params.num_accepted_tokens_host = {15, 16};
  input.input_params.num_accepted_tokens =
      torch::tensor({115, 116}, cpu_tensor_options(torch::kLong));

  const void* source_padding_address =
      input.input_params.parallel.dp_ep_padding_data.attn_padding_idx()
          .data_ptr();
  ForwardInput first_staged;
  ASSERT_TRUE(arena.stage(input, first_staged));
  const DpEpPaddingData& first_padding =
      first_staged.input_params.parallel.dp_ep_padding_data;
  EXPECT_NE(first_padding.attn_padding_idx().data_ptr(),
            source_padding_address);
  EXPECT_TRUE(
      torch::equal(first_padding.attn_padding_idx(),
                   torch::tensor({5, 6}, cpu_tensor_options(torch::kInt))));
  EXPECT_TRUE(
      torch::equal(first_staged.input_params.graph.expanded_kv_seq_lens,
                   torch::tensor({9, 10}, cpu_tensor_options(torch::kInt))));
  EXPECT_TRUE(
      torch::equal(first_staged.input_params.num_accepted_tokens,
                   torch::tensor({15, 16}, cpu_tensor_options(torch::kLong))));
  EXPECT_TRUE(torch::equal(
      first_staged.input_params.expert.eplb_decode_token_mask,
      torch::tensor({true, false}, cpu_tensor_options(torch::kBool))));

  const void* first_padding_address =
      first_padding.attn_padding_idx().data_ptr();
  const void* first_expanded_address =
      first_staged.input_params.graph.expanded_block_tables.data_ptr();
  input.input_params.parallel.dp_ep_padding_data.attn_padding_idx() =
      torch::tensor({25, 26}, cpu_tensor_options(torch::kInt));
  input.input_params.graph.expanded_block_tables =
      torch::tensor({{31, 32}, {33, 34}}, cpu_tensor_options(torch::kInt));
  ForwardInput second_staged;
  ASSERT_TRUE(arena.stage(input, second_staged));
  EXPECT_EQ(
      second_staged.input_params.parallel.dp_ep_padding_data.attn_padding_idx()
          .data_ptr(),
      first_padding_address);
  EXPECT_EQ(second_staged.input_params.graph.expanded_block_tables.data_ptr(),
            first_expanded_address);
  EXPECT_TRUE(torch::equal(
      second_staged.input_params.parallel.dp_ep_padding_data.attn_padding_idx(),
      torch::tensor({25, 26}, cpu_tensor_options(torch::kInt))));
}

TEST(PreparedInputArenaTest, AppendsWorkerGeneratedMetadataToStableArenaTail) {
  PreparedInputArena arena(torch::Device(torch::kCPU),
                           /*capacity_bytes=*/4096);
  ForwardInput input = make_input(BatchForwardType::DECODE, /*batch_id=*/88);
  input.token_ids_host = torch::tensor({1, 2}, cpu_tensor_options(torch::kInt));
  input.positions_host =
      torch::tensor({10, 11}, cpu_tensor_options(torch::kInt));

  ForwardInput first_staged;
  ASSERT_TRUE(arena.stage(input, first_staged));
  const int64_t primary_buffer_bytes = first_staged.device_input_buffer.numel();
  EXPECT_EQ(first_staged.prepared_arena_h2d_bytes,
            static_cast<uint64_t>(primary_buffer_bytes));
  EXPECT_EQ(first_staged.prepared_arena_h2d_copies, 1);
  EXPECT_EQ(first_staged.prepared_arena_d2d_bytes, 0);
  EXPECT_EQ(first_staged.prepared_arena_d2d_copies, 0);
  torch::Tensor first_generated_source =
      torch::tensor({21, 22}, cpu_tensor_options(torch::kInt));
  first_staged.input_params.parallel.dp_ep_padding_data.ffn_padding_idx() =
      first_generated_source;
  torch::Tensor first_tiling_source =
      torch::tensor({41, 42, 43}, cpu_tensor_options(torch::kInt));
  first_staged.input_params.graph.tiling_data = first_tiling_source;
  ASSERT_TRUE(arena.stage_generated_metadata(first_staged));
  const void* first_generated_address =
      first_staged.input_params.parallel.dp_ep_padding_data.ffn_padding_idx()
          .data_ptr();
  EXPECT_NE(first_generated_address, first_generated_source.data_ptr());
  const void* first_tiling_address =
      first_staged.input_params.graph.tiling_data.data_ptr();
  EXPECT_NE(first_tiling_address, first_tiling_source.data_ptr());
  EXPECT_GT(first_staged.device_input_buffer.numel(), primary_buffer_bytes);
  EXPECT_GT(first_staged.prepared_arena_h2d_bytes,
            static_cast<uint64_t>(primary_buffer_bytes));
  EXPECT_EQ(first_staged.prepared_arena_h2d_copies, 2);
  EXPECT_EQ(first_staged.prepared_arena_d2d_bytes, 0);
  EXPECT_EQ(first_staged.prepared_arena_d2d_copies, 0);
  const uint64_t first_signature = first_staged.prepared_input_layout_signature;

  ForwardInput second_staged;
  ASSERT_TRUE(arena.stage(input, second_staged));
  second_staged.input_params.parallel.dp_ep_padding_data.ffn_padding_idx() =
      torch::tensor({31, 32}, cpu_tensor_options(torch::kInt));
  second_staged.input_params.graph.tiling_data =
      torch::tensor({51, 52, 53}, cpu_tensor_options(torch::kInt));
  ASSERT_TRUE(arena.stage_generated_metadata(second_staged));
  EXPECT_EQ(
      second_staged.input_params.parallel.dp_ep_padding_data.ffn_padding_idx()
          .data_ptr(),
      first_generated_address);
  EXPECT_EQ(second_staged.prepared_input_layout_signature, first_signature);
  EXPECT_EQ(second_staged.input_params.graph.tiling_data.data_ptr(),
            first_tiling_address);
  EXPECT_TRUE(torch::equal(
      second_staged.input_params.graph.tiling_data,
      torch::tensor({51, 52, 53}, cpu_tensor_options(torch::kInt))));
  EXPECT_TRUE(torch::equal(
      second_staged.input_params.parallel.dp_ep_padding_data.ffn_padding_idx(),
      torch::tensor({31, 32}, cpu_tensor_options(torch::kInt))));
}

TEST(PreparedInputArenaTest,
     KeepsExistingArenaViewsWithoutAppendingGeneratedTail) {
  PreparedInputArena arena(torch::Device(torch::kCPU),
                           /*capacity_bytes=*/4096);
  ForwardInput input = make_input(BatchForwardType::DECODE, /*batch_id=*/91);
  input.token_ids_host = torch::tensor({1, 2}, cpu_tensor_options(torch::kInt));
  input.positions_host =
      torch::tensor({10, 11}, cpu_tensor_options(torch::kInt));
  input.input_params.attention.device.paged_kv_indptr =
      torch::tensor({0, 1, 2}, cpu_tensor_options(torch::kInt));
  input.input_params.attention.device.paged_kv_indices =
      torch::tensor({3, 4}, cpu_tensor_options(torch::kInt));

  ForwardInput staged;
  ASSERT_TRUE(arena.stage(input, staged));
  const int64_t primary_buffer_bytes = staged.device_input_buffer.numel();
  const void* paged_indices_address =
      staged.input_params.attention.device.paged_kv_indices.data_ptr();
  const uint64_t primary_h2d_bytes = staged.prepared_arena_h2d_bytes;
  const int32_t primary_h2d_copies = staged.prepared_arena_h2d_copies;

  ASSERT_TRUE(arena.stage_generated_metadata(staged));

  EXPECT_EQ(staged.device_input_buffer.numel(), primary_buffer_bytes);
  EXPECT_EQ(staged.input_params.attention.device.paged_kv_indices.data_ptr(),
            paged_indices_address);
  EXPECT_EQ(staged.prepared_arena_h2d_bytes, primary_h2d_bytes);
  EXPECT_EQ(staged.prepared_arena_h2d_copies, primary_h2d_copies);
}

TEST(PreparedInputArenaTest, RebindsGraphTokenOverrideToArenaTokens) {
  PreparedInputArena arena(torch::Device(torch::kCPU),
                           /*capacity_bytes=*/4096);
  ForwardInput input = make_input(BatchForwardType::DECODE, /*batch_id=*/89);
  input.token_ids = torch::tensor({1, 2}, cpu_tensor_options(torch::kInt));
  input.token_ids_host = input.token_ids;
  input.positions_host =
      torch::tensor({10, 11}, cpu_tensor_options(torch::kInt));
  input.input_params.graph.input_tokens_override = input.token_ids;

  ForwardInput staged;
  ASSERT_TRUE(arena.stage(input, staged));
  EXPECT_TRUE(staged.input_params.graph.input_tokens_override.is_same(
      staged.token_ids));
  EXPECT_NE(staged.token_ids.data_ptr(), input.token_ids.data_ptr());
}

TEST(PreparedInputArenaTest, RejectsUnstableExternalGraphSources) {
  PreparedInputArena arena(torch::Device(torch::kCPU),
                           /*capacity_bytes=*/4096);
  ForwardInput input = make_input(BatchForwardType::DECODE, /*batch_id=*/90);
  input.token_ids = torch::tensor({1, 2}, cpu_tensor_options(torch::kInt));
  input.token_ids_host = input.token_ids;
  input.positions_host =
      torch::tensor({10, 11}, cpu_tensor_options(torch::kInt));
  input.input_params.graph.input_tokens_override =
      torch::tensor({3, 4}, cpu_tensor_options(torch::kInt));
  ForwardInput staged;
  EXPECT_FALSE(arena.stage(input, staged));

  input.input_params.graph.input_tokens_override = input.token_ids;
  input.input_params.graph.spec_verify_draft_token_sources.emplace_back(
      torch::tensor({5, 6}, cpu_tensor_options(torch::kInt)));
  EXPECT_FALSE(arena.stage(input, staged));
  input.input_params.graph.spec_verify_source_addresses_stable = true;
  EXPECT_TRUE(arena.stage(input, staged));
}

TEST(PreparedInputArenaTest,
     StagesSpecVerifyDirectBindContractAtStableAddresses) {
  PreparedInputArena arena(torch::Device(torch::kCPU),
                           /*capacity_bytes=*/16384);
  ForwardInput input =
      make_input(BatchForwardType::CHUNKED_PREFILL, /*batch_id=*/92);
  input.input_params.is_spec_verify = true;
  input.input_params.meta.num_sequences = 2;
  input.input_params.meta.q_max_seq_len = 3;
  input.input_params.meta.kv_max_seq_len = 11;
  input.token_ids_host =
      torch::tensor({10, -1, -2, 20, -1, -2}, cpu_tensor_options(torch::kInt));
  input.token_ids = input.token_ids_host;
  input.positions_host =
      torch::tensor({4, 5, 6, 8, 9, 10}, cpu_tensor_options(torch::kInt));
  input.positions = input.positions_host;
  AttentionHostInput& attention = input.input_params.attention.host;
  attention.q_seq_lens = {3, 3};
  attention.q_cu_seq_lens = {0, 3, 6};
  attention.kv_seq_lens = {7, 11};
  attention.new_cache_slots = {16, 17, 18, 32, 33, 34};
  attention.block_tables =
      torch::tensor({{1, 2, 3}, {4, 5, 6}}, cpu_tensor_options(torch::kInt));
  input.input_params.embedding.linear_state_ids = {7, 9};
  input.input_params.num_accepted_tokens_host = {1, 2};

  ForwardInput first_staged;
  ASSERT_TRUE(arena.stage(input, first_staged));
  ASSERT_TRUE(prepared_spec_verify_graph_contract_is_complete(first_staged));
  EXPECT_TRUE(first_staged.input_params.graph.input_tokens_override.is_same(
      first_staged.token_ids));
  EXPECT_TRUE(
      first_staged.input_params.graph.spec_verify_draft_token_sources.empty());
  EXPECT_FALSE(
      first_staged.input_params.graph.spec_verify_source_addresses_stable);
  EXPECT_FALSE(
      first_staged.input_params.graph.spec_verify_static_graph_tasks_prepared);
  EXPECT_TRUE(first_staged.input_params.graph.prepared_spec_verify_direct_bind);
  const void* token_address = first_staged.token_ids.data_ptr();
  const void* accepted_address =
      first_staged.input_params.num_accepted_tokens.data_ptr();
  const void* linear_state_address =
      first_staged.input_params.embedding.linear_state_indices.data_ptr();
  const uint64_t layout_signature =
      first_staged.prepared_input_layout_signature;

  input.token_ids_host.copy_(
      torch::tensor({30, -1, -2, 40, -1, -2}, cpu_tensor_options(torch::kInt)));
  input.input_params.num_accepted_tokens_host = {3, 1};
  ForwardInput second_staged;
  ASSERT_TRUE(arena.stage(input, second_staged));
  EXPECT_TRUE(prepared_spec_verify_graph_contract_is_complete(second_staged));
  EXPECT_EQ(second_staged.token_ids.data_ptr(), token_address);
  EXPECT_EQ(second_staged.input_params.num_accepted_tokens.data_ptr(),
            accepted_address);
  EXPECT_EQ(
      second_staged.input_params.embedding.linear_state_indices.data_ptr(),
      linear_state_address);
  EXPECT_EQ(second_staged.prepared_input_layout_signature, layout_signature);
  EXPECT_TRUE(
      torch::equal(second_staged.input_params.num_accepted_tokens,
                   torch::tensor({3, 1}, cpu_tensor_options(torch::kLong))));

  second_staged.input_params.graph
      .use_expanded_decode_for_spec_verify_attention = true;
  EXPECT_FALSE(prepared_spec_verify_graph_contract_is_complete(second_staged));
}

TEST(PreparedInputArenaTest,
     RejectsLegacySpecVerifyReplaySourcesFromDirectBindArena) {
  PreparedInputArena arena(torch::Device(torch::kCPU),
                           /*capacity_bytes=*/4096);
  ForwardInput input =
      make_input(BatchForwardType::CHUNKED_PREFILL, /*batch_id=*/93);
  input.input_params.is_spec_verify = true;
  input.input_params.meta.num_sequences = 1;
  input.input_params.meta.q_max_seq_len = 2;
  input.token_ids = torch::tensor({1, -1}, cpu_tensor_options(torch::kInt));
  input.token_ids_host = input.token_ids;
  input.positions_host = torch::tensor({4, 5}, cpu_tensor_options(torch::kInt));
  input.input_params.graph.input_tokens_override = input.token_ids;
  input.input_params.graph.spec_verify_draft_token_sources.emplace_back(
      torch::tensor({2}, cpu_tensor_options(torch::kLong)));
  input.input_params.graph.spec_verify_source_addresses_stable = true;
  ForwardInput staged;
  EXPECT_FALSE(arena.stage(input, staged));

  input.input_params.graph.spec_verify_draft_token_sources.clear();
  input.input_params.graph.spec_verify_static_graph_tasks_prepared = true;
  EXPECT_FALSE(arena.stage(input, staged));
}

TEST(PreparedInputArenaTest,
     StagesExpandedSpecVerifyMetadataAtStableArenaAddresses) {
  PreparedInputArena arena(torch::Device(torch::kCPU),
                           /*capacity_bytes=*/32768);
  ForwardInput input =
      make_input(BatchForwardType::CHUNKED_PREFILL, /*batch_id=*/94);
  input.input_params.is_spec_verify = true;
  input.input_params.meta.num_sequences = 2;
  input.input_params.meta.q_max_seq_len = 3;
  input.input_params.meta.kv_max_seq_len = 11;
  input.token_ids_host =
      torch::tensor({10, -1, -2, 20, -1, -2}, cpu_tensor_options(torch::kInt));
  input.token_ids = input.token_ids_host;
  input.positions_host =
      torch::tensor({4, 5, 6, 8, 9, 10}, cpu_tensor_options(torch::kInt));
  input.positions = input.positions_host;
  AttentionHostInput& attention = input.input_params.attention.host;
  attention.q_seq_lens = {3, 3};
  attention.q_cu_seq_lens = {0, 3, 6};
  attention.kv_seq_lens = {7, 11};
  attention.new_cache_slots = {16, 17, 18, 32, 33, 34};
  attention.block_tables =
      torch::tensor({{1, 2, 3}, {4, 5, 6}}, cpu_tensor_options(torch::kInt));
  input.input_params.parallel.query_start_loc = {0, 3, 6};
  input.input_params.embedding.linear_state_ids = {7, 9};
  input.input_params.num_accepted_tokens_host = {1, 2};

  GraphInput& graph = input.input_params.graph;
  graph.use_expanded_decode_for_spec_verify_attention = true;
  graph.spec_verify_kv_seq_len_headroom = 2;
  graph.expanded_kv_seq_lens_vec = {5, 6, 7, 9, 10, 11};
  graph.expanded_kv_seq_lens = torch::tensor(graph.expanded_kv_seq_lens_vec,
                                             cpu_tensor_options(torch::kInt));
  graph.expanded_block_tables = torch::tensor(
      {{1, 2, 3}, {1, 2, 3}, {1, 2, 3}, {4, 5, 6}, {4, 5, 6}, {4, 5, 6}},
      cpu_tensor_options(torch::kInt));
  graph.expanded_paged_kv_indptr =
      torch::tensor({0, 2, 4, 6, 9, 12, 15}, cpu_tensor_options(torch::kInt));
  graph.expanded_paged_kv_indices =
      torch::zeros({36}, cpu_tensor_options(torch::kInt));
  graph.expanded_paged_kv_last_page_len =
      torch::tensor({1, 2, 3, 1, 2, 3}, cpu_tensor_options(torch::kInt));

  ForwardInput first_staged;
  ASSERT_TRUE(arena.stage(input, first_staged));
  ASSERT_TRUE(prepared_spec_verify_graph_contract_is_complete(first_staged));
  const layer::ExpandedDecodeMetadata first_metadata =
      layer::ExpandedDecodeMetadataBuilder::build(first_staged.input_params);
  EXPECT_TRUE(first_metadata.kv_seq_lens.is_same(
      first_staged.input_params.graph.expanded_kv_seq_lens));
  EXPECT_FALSE(first_metadata.kv_seq_lens_host.defined());
  EXPECT_TRUE(first_metadata.kv_seq_lens_host_vec.empty());
  const void* expanded_kv_address =
      first_staged.input_params.graph.expanded_kv_seq_lens.data_ptr();
  const void* expanded_block_address =
      first_staged.input_params.graph.expanded_block_tables.data_ptr();
  const void* expanded_indptr_address =
      first_staged.input_params.graph.expanded_paged_kv_indptr.data_ptr();
  const void* expanded_indices_address =
      first_staged.input_params.graph.expanded_paged_kv_indices.data_ptr();
  const void* expanded_last_page_address =
      first_staged.input_params.graph.expanded_paged_kv_last_page_len
          .data_ptr();
  const uint64_t layout_signature =
      first_staged.prepared_input_layout_signature;

  graph.expanded_kv_seq_lens_vec = {6, 7, 8, 10, 11, 12};
  graph.expanded_kv_seq_lens.copy_(torch::tensor(
      graph.expanded_kv_seq_lens_vec, cpu_tensor_options(torch::kInt)));
  attention.kv_seq_lens = {8, 12};
  input.input_params.meta.kv_max_seq_len = 12;
  ForwardInput second_staged;
  ASSERT_TRUE(arena.stage(input, second_staged));
  ASSERT_TRUE(prepared_spec_verify_graph_contract_is_complete(second_staged));
  const layer::ExpandedDecodeMetadata second_metadata =
      layer::ExpandedDecodeMetadataBuilder::build(second_staged.input_params);
  EXPECT_TRUE(second_metadata.kv_seq_lens.is_same(
      second_staged.input_params.graph.expanded_kv_seq_lens));
  EXPECT_FALSE(second_metadata.kv_seq_lens_host.defined());
  EXPECT_TRUE(second_metadata.kv_seq_lens_host_vec.empty());
  EXPECT_TRUE(torch::equal(
      second_metadata.kv_seq_lens,
      torch::tensor({6, 7, 8, 10, 11, 12}, cpu_tensor_options(torch::kInt))));
  EXPECT_EQ(second_staged.input_params.graph.expanded_kv_seq_lens.data_ptr(),
            expanded_kv_address);
  EXPECT_EQ(second_staged.input_params.graph.expanded_block_tables.data_ptr(),
            expanded_block_address);
  EXPECT_EQ(
      second_staged.input_params.graph.expanded_paged_kv_indptr.data_ptr(),
      expanded_indptr_address);
  EXPECT_EQ(
      second_staged.input_params.graph.expanded_paged_kv_indices.data_ptr(),
      expanded_indices_address);
  EXPECT_EQ(second_staged.input_params.graph.expanded_paged_kv_last_page_len
                .data_ptr(),
            expanded_last_page_address);
  EXPECT_EQ(second_staged.prepared_input_layout_signature, layout_signature);

  ForwardInput invalid_contract = second_staged;
  invalid_contract.input_params.parallel.query_start_loc[1] = 2;
  EXPECT_FALSE(
      prepared_spec_verify_graph_contract_is_complete(invalid_contract));
  invalid_contract.input_params.parallel.query_start_loc[1] = 3;
  invalid_contract.input_params.graph.expanded_kv_seq_lens_vec[0] = 5;
  EXPECT_FALSE(
      prepared_spec_verify_graph_contract_is_complete(invalid_contract));
  invalid_contract.input_params.graph.expanded_kv_seq_lens_vec[0] = 6;
  invalid_contract.input_params.graph.spec_verify_kv_seq_len_headroom = 1;
  EXPECT_FALSE(
      prepared_spec_verify_graph_contract_is_complete(invalid_contract));
  invalid_contract.input_params.graph.spec_verify_kv_seq_len_headroom = 2;
  invalid_contract.input_params.graph.expanded_paged_kv_last_page_len =
      invalid_contract.input_params.graph.expanded_paged_kv_last_page_len
          .narrow(/*dim=*/0, /*start=*/0, /*length=*/5);
  EXPECT_FALSE(
      prepared_spec_verify_graph_contract_is_complete(invalid_contract));
}

TEST(BlockSpecDeviceStepStateTest,
     PublishesFixedWidthContextKvMetadataWithoutReplacingStorage) {
  const torch::TensorOptions token_options = cpu_tensor_options(torch::kLong);
  const torch::TensorOptions hidden_options = cpu_tensor_options(torch::kFloat);
  const torch::TensorOptions position_options = cpu_tensor_options(torch::kInt);
  block_spec_async::BlockSpecDeviceStepState state =
      block_spec_async::allocate_block_spec_device_step_state(
          /*max_rows=*/3,
          /*accepted_token_capacity=*/4,
          /*context_hidden_size=*/2,
          token_options,
          hidden_options,
          position_options,
          position_options);
  block_spec_async::BlockSpecContextKvPatchWorkspace workspace =
      block_spec_async::allocate_block_spec_context_kv_patch_workspace(
          /*max_rows=*/3,
          /*accepted_token_capacity=*/4,
          position_options,
          position_options);
  const void* token_address = state.accepted_tokens.data_ptr();
  const void* hidden_address = state.context_hidden.data_ptr();
  const void* position_address = state.context_positions.data_ptr();
  const void* cache_slot_address = state.context_cache_slots.data_ptr();

  torch::Tensor accepted_tokens =
      torch::tensor({{10, 11, -1, -1}, {20, 21, 22, -1}}, token_options);
  torch::Tensor accepted_hidden = torch::arange(
      /*start=*/0,
      /*end=*/16,
      hidden_options);
  accepted_hidden = accepted_hidden.view({2, 4, 2}).contiguous();
  torch::Tensor base_positions = torch::tensor({5, 7}, position_options);
  torch::Tensor block_tables = torch::tensor(
      {{100, 101, 102, 103}, {200, 201, 202, 203}}, position_options);
  block_spec_async::publish_block_spec_context_kv_state(
      accepted_tokens,
      accepted_hidden,
      base_positions,
      block_tables,
      /*block_size=*/4,
      block_spec_async::CacheSlotMappingMode::LINEAR,
      state,
      workspace);

  EXPECT_EQ(state.accepted_tokens.data_ptr(), token_address);
  EXPECT_EQ(state.context_hidden.data_ptr(), hidden_address);
  EXPECT_EQ(state.context_positions.data_ptr(), position_address);
  EXPECT_EQ(state.context_cache_slots.data_ptr(), cache_slot_address);
  EXPECT_TRUE(torch::equal(state.accepted_lengths,
                           torch::tensor({2, 3, 0}, token_options)));
  EXPECT_TRUE(torch::equal(state.valid_tokens,
                           torch::tensor({{true, true, false, false},
                                          {true, true, true, false},
                                          {false, false, false, false}},
                                         cpu_tensor_options(torch::kBool))));
  EXPECT_TRUE(
      torch::equal(state.context_positions,
                   torch::tensor({{5, 6, 0, 0}, {7, 8, 9, 0}, {0, 0, 0, 0}},
                                 position_options)));
  EXPECT_TRUE(torch::equal(
      state.context_cache_slots,
      torch::tensor({{405, 406, 0, 0}, {807, 808, 809, 0}, {0, 0, 0, 0}},
                    position_options)));
  EXPECT_TRUE(torch::equal(
      state.context_hidden.narrow(/*dim=*/0, /*start=*/0, /*length=*/2),
      accepted_hidden));
  EXPECT_TRUE(torch::equal(state.tail_tokens,
                           torch::tensor({11, 22, -1}, token_options)));
  EXPECT_TRUE(torch::equal(state.next_positions,
                           torch::tensor({7, 10, 0}, position_options)));
  EXPECT_TRUE(torch::equal(
      state.valid_rows,
      torch::tensor({true, true, false}, cpu_tensor_options(torch::kBool))));
  EXPECT_TRUE(torch::equal(
      state.tail_context_hidden.narrow(
          /*dim=*/0, /*start=*/0, /*length=*/2),
      torch::stack({accepted_hidden[0][1], accepted_hidden[1][2]})));

  block_spec_async::publish_block_spec_context_kv_state(
      torch::tensor({{30, -1, -1, -1}}, token_options),
      torch::zeros({1, 4, 2}, hidden_options),
      torch::tensor({3}, position_options),
      torch::tensor({{300, 301, 302, 303}}, position_options),
      /*block_size=*/4,
      block_spec_async::CacheSlotMappingMode::LINEAR,
      state,
      workspace);
  EXPECT_EQ(state.accepted_tokens.data_ptr(), token_address);
  EXPECT_EQ(state.context_hidden.data_ptr(), hidden_address);
  EXPECT_EQ(state.context_positions.data_ptr(), position_address);
  EXPECT_EQ(state.context_cache_slots.data_ptr(), cache_slot_address);
  EXPECT_TRUE(torch::equal(state.accepted_lengths,
                           torch::tensor({1, 0, 0}, token_options)));
  EXPECT_EQ(state.context_cache_slots[0][0].item<int32_t>(), 1203);
  EXPECT_EQ(state.tail_tokens[0].item<int64_t>(), 30);
  EXPECT_EQ(state.next_positions[0].item<int32_t>(), 4);
}

TEST(BlockSpecDeviceStepStateTest, RejectsPublishWidthBelowVerifyCapacity) {
  const torch::TensorOptions token_options = cpu_tensor_options(torch::kLong);
  const torch::TensorOptions hidden_options = cpu_tensor_options(torch::kFloat);
  const torch::TensorOptions position_options = cpu_tensor_options(torch::kInt);
  block_spec_async::BlockSpecDeviceStepState state =
      block_spec_async::allocate_block_spec_device_step_state(
          /*max_rows=*/1,
          /*accepted_token_capacity=*/4,
          /*context_hidden_size=*/2,
          token_options,
          hidden_options,
          position_options,
          position_options);
  block_spec_async::BlockSpecContextKvPatchWorkspace workspace =
      block_spec_async::allocate_block_spec_context_kv_patch_workspace(
          /*max_rows=*/1,
          /*accepted_token_capacity=*/4,
          position_options,
          position_options);

  EXPECT_DEATH(block_spec_async::publish_block_spec_context_kv_state(
                   torch::tensor({{10, 11, -1}}, token_options),
                   torch::zeros({1, 3, 2}, hidden_options),
                   torch::tensor({5}, position_options),
                   torch::tensor({{100, 101}}, position_options),
                   /*block_size=*/4,
                   block_spec_async::CacheSlotMappingMode::LINEAR,
                   state,
                   workspace),
               "must match the configured Target Verify capacity");
}

TEST(BlockSpecDeviceStepStateTest,
     RejectsPublishDeviceAndDtypeMismatchesBeforeWrites) {
  const torch::TensorOptions token_options = cpu_tensor_options(torch::kLong);
  const torch::TensorOptions hidden_options = cpu_tensor_options(torch::kFloat);
  const torch::TensorOptions position_options = cpu_tensor_options(torch::kInt);
  block_spec_async::BlockSpecDeviceStepState state =
      block_spec_async::allocate_block_spec_device_step_state(
          /*max_rows=*/1,
          /*accepted_token_capacity=*/4,
          /*context_hidden_size=*/2,
          token_options,
          hidden_options,
          position_options,
          position_options);
  block_spec_async::BlockSpecContextKvPatchWorkspace workspace =
      block_spec_async::allocate_block_spec_context_kv_patch_workspace(
          /*max_rows=*/1,
          /*accepted_token_capacity=*/4,
          position_options,
          position_options);
  const torch::Tensor accepted_tokens =
      torch::tensor({{10, 11, -1, -1}}, token_options);
  const torch::Tensor accepted_hidden = torch::zeros({1, 4, 2}, hidden_options);
  const torch::Tensor base_positions = torch::tensor({5}, position_options);
  const torch::Tensor block_tables =
      torch::tensor({{100, 101}}, position_options);

  state.context_hidden =
      torch::empty({1, 4, 2}, hidden_options.device(torch::Device("meta")));
  EXPECT_DEATH(block_spec_async::publish_block_spec_context_kv_state(
                   accepted_tokens,
                   accepted_hidden,
                   base_positions,
                   block_tables,
                   /*block_size=*/4,
                   block_spec_async::CacheSlotMappingMode::LINEAR,
                   state,
                   workspace),
               "destination.context_hidden.*contract device");

  state.context_hidden = torch::empty({1, 4, 2}, hidden_options);
  workspace.invalid_tokens =
      torch::empty({1, 4}, cpu_tensor_options(torch::kLong));
  EXPECT_DEATH(block_spec_async::publish_block_spec_context_kv_state(
                   accepted_tokens,
                   accepted_hidden,
                   base_positions,
                   block_tables,
                   /*block_size=*/4,
                   block_spec_async::CacheSlotMappingMode::LINEAR,
                   state,
                   workspace),
               "workspace.invalid_tokens.*incompatible dtype");
}

TEST(BlockSpecDeviceStepStateTest,
     RejectsContinuationDeviceAndDtypeMismatchesBeforeWrites) {
  const torch::TensorOptions token_options = cpu_tensor_options(torch::kLong);
  const torch::TensorOptions hidden_options = cpu_tensor_options(torch::kFloat);
  const torch::TensorOptions position_options = cpu_tensor_options(torch::kInt);
  block_spec_async::BlockSpecDeviceStepState predecessor =
      block_spec_async::allocate_block_spec_device_step_state(
          /*max_rows=*/2,
          /*accepted_token_capacity=*/4,
          /*context_hidden_size=*/2,
          token_options,
          hidden_options,
          position_options,
          position_options);
  block_spec_async::BlockSpecContinuationState continuation;
  continuation.anchor_tokens = torch::zeros({2}, token_options);
  continuation.anchor_context_hidden = torch::zeros({2, 2}, hidden_options);
  continuation.base_positions = torch::zeros({2}, position_options);
  block_spec_async::BlockSpecPredecessorPatchWorkspace workspace =
      block_spec_async::allocate_block_spec_predecessor_patch_workspace(
          /*max_rows=*/2,
          /*context_hidden_size=*/2,
          token_options,
          hidden_options,
          position_options);
  const torch::Tensor predecessor_rows = torch::tensor({0, 1}, token_options);

  predecessor.tail_context_hidden =
      torch::empty({2, 2}, hidden_options.device(torch::Device("meta")));
  EXPECT_DEATH(block_spec_async::patch_block_spec_continuation_rows(
                   predecessor, predecessor_rows, continuation, workspace),
               "predecessor.tail_context_hidden.*contract device");

  predecessor.tail_context_hidden = torch::empty({2, 2}, hidden_options);
  workspace.gathered_positions =
      torch::empty({2}, cpu_tensor_options(torch::kLong));
  EXPECT_DEATH(block_spec_async::patch_block_spec_continuation_rows(
                   predecessor, predecessor_rows, continuation, workspace),
               "workspace.gathered_positions.*incompatible dtype");
}

TEST(BlockSpecDeviceStepStateTest,
     PublishesCircularSwaContextCacheSlotsInPlace) {
  const torch::TensorOptions token_options = cpu_tensor_options(torch::kLong);
  const torch::TensorOptions hidden_options = cpu_tensor_options(torch::kFloat);
  const torch::TensorOptions position_options = cpu_tensor_options(torch::kInt);
  block_spec_async::BlockSpecDeviceStepState state =
      block_spec_async::allocate_block_spec_device_step_state(
          /*max_rows=*/1,
          /*accepted_token_capacity=*/4,
          /*context_hidden_size=*/2,
          token_options,
          hidden_options,
          position_options,
          position_options);
  block_spec_async::BlockSpecContextKvPatchWorkspace workspace =
      block_spec_async::allocate_block_spec_context_kv_patch_workspace(
          /*max_rows=*/1,
          /*accepted_token_capacity=*/4,
          position_options,
          position_options);
  const void* cache_slot_address = state.context_cache_slots.data_ptr();

  block_spec_async::publish_block_spec_context_kv_state(
      torch::tensor({{10, 11, 12, -1}}, token_options),
      torch::zeros({1, 4, 2}, hidden_options),
      torch::tensor({7}, position_options),
      torch::tensor({{10, 11}}, position_options),
      /*block_size=*/4,
      block_spec_async::CacheSlotMappingMode::CIRCULAR,
      state,
      workspace);

  EXPECT_EQ(state.context_cache_slots.data_ptr(), cache_slot_address);
  EXPECT_TRUE(torch::equal(state.context_cache_slots,
                           torch::tensor({{47, 40, 41, 0}}, position_options)));
}

TEST(BlockSpecDecodeInputPatchTest,
     PatchesTokenwiseQueryAndTargetGeometryInPlace) {
  const torch::TensorOptions token_options = cpu_tensor_options(torch::kLong);
  const torch::TensorOptions position_options = cpu_tensor_options(torch::kInt);
  block_spec_async::BlockSpecContinuationState continuation;
  continuation.anchor_tokens = torch::tensor({10, 20}, token_options);
  continuation.base_positions = torch::tensor({3, 6}, position_options);

  block_spec_async::BlockSpecDecodeInputPatchTarget target;
  target.query_token_ids = torch::full({2, 4}, -1, token_options);
  target.query_positions = torch::empty({2, 4}, position_options);
  target.query_kv_seq_lens = torch::empty({2}, position_options);
  target.query_new_cache_slots = torch::empty({2, 4}, position_options);
  target.target_token_ids = torch::full({2, 4}, -1, token_options);
  target.target_positions = torch::empty({2, 4}, position_options);
  target.target_kv_seq_lens = torch::empty({8}, position_options);
  target.target_new_cache_slots = torch::empty({2, 4}, position_options);
  target.source_block_tables =
      torch::tensor({{100, 101, 102}, {200, 201, 202}}, position_options);
  block_spec_async::BlockSpecDecodeInputPatchWorkspace workspace =
      block_spec_async::allocate_block_spec_decode_input_patch_workspace(
          /*max_rows=*/2,
          /*query_width=*/4,
          /*target_width=*/4,
          position_options,
          position_options);
  const void* query_token_address = target.query_token_ids.data_ptr();
  const void* query_position_address = target.query_positions.data_ptr();
  const void* query_cache_slot_address =
      target.query_new_cache_slots.data_ptr();
  const void* target_token_address = target.target_token_ids.data_ptr();
  const void* target_position_address = target.target_positions.data_ptr();
  const void* target_kv_length_address = target.target_kv_seq_lens.data_ptr();
  const void* target_cache_slot_address =
      target.target_new_cache_slots.data_ptr();

  block_spec_async::patch_block_spec_decode_input_geometry(
      continuation, /*block_size=*/4, target, workspace);
  block_spec_async::patch_block_spec_target_token_ids(
      torch::tensor({{11, 12, 13}, {21, 22, 23}}, token_options),
      target.target_token_ids);

  EXPECT_EQ(target.query_token_ids.data_ptr(), query_token_address);
  EXPECT_EQ(target.query_positions.data_ptr(), query_position_address);
  EXPECT_EQ(target.query_new_cache_slots.data_ptr(), query_cache_slot_address);
  EXPECT_EQ(target.target_token_ids.data_ptr(), target_token_address);
  EXPECT_EQ(target.target_positions.data_ptr(), target_position_address);
  EXPECT_EQ(target.target_kv_seq_lens.data_ptr(), target_kv_length_address);
  EXPECT_EQ(target.target_new_cache_slots.data_ptr(),
            target_cache_slot_address);
  EXPECT_TRUE(torch::equal(
      target.query_token_ids,
      torch::tensor({{10, -1, -1, -1}, {20, -1, -1, -1}}, token_options)));
  EXPECT_TRUE(torch::equal(
      target.target_token_ids,
      torch::tensor({{10, 11, 12, 13}, {20, 21, 22, 23}}, token_options)));
  EXPECT_TRUE(torch::equal(
      target.query_positions,
      torch::tensor({{3, 4, 5, 6}, {6, 7, 8, 9}}, position_options)));
  EXPECT_TRUE(torch::equal(target.target_positions, target.query_positions));
  EXPECT_TRUE(torch::equal(target.query_kv_seq_lens,
                           torch::tensor({7, 10}, position_options)));
  EXPECT_TRUE(
      torch::equal(target.target_kv_seq_lens,
                   torch::tensor({4, 5, 6, 7, 7, 8, 9, 10}, position_options)));
  EXPECT_TRUE(
      torch::equal(target.query_new_cache_slots,
                   torch::tensor({{403, 404, 405, 406}, {806, 807, 808, 809}},
                                 position_options)));
  EXPECT_TRUE(torch::equal(target.target_new_cache_slots,
                           target.query_new_cache_slots));
}

TEST(BlockSpecDecodeInputPatchTest,
     PatchesCircularSwaQueryAndTargetCacheSlotsInPlace) {
  const torch::TensorOptions token_options = cpu_tensor_options(torch::kLong);
  const torch::TensorOptions position_options = cpu_tensor_options(torch::kInt);
  block_spec_async::BlockSpecContinuationState continuation;
  continuation.anchor_tokens = torch::tensor({10}, token_options);
  continuation.base_positions = torch::tensor({7}, position_options);

  block_spec_async::BlockSpecDecodeInputPatchTarget target;
  target.query_token_ids = torch::empty({1, 3}, token_options);
  target.query_positions = torch::empty({1, 3}, position_options);
  target.query_kv_seq_lens = torch::empty({1}, position_options);
  target.query_new_cache_slots = torch::empty({1, 3}, position_options);
  target.target_token_ids = torch::empty({1, 4}, token_options);
  target.target_positions = torch::empty({1, 4}, position_options);
  target.target_kv_seq_lens = torch::empty({1}, position_options);
  target.target_new_cache_slots = torch::empty({1, 4}, position_options);
  target.source_block_tables = torch::tensor({{10, 11}}, position_options);
  target.cache_slot_mapping_mode =
      block_spec_async::CacheSlotMappingMode::CIRCULAR;
  block_spec_async::BlockSpecDecodeInputPatchWorkspace workspace =
      block_spec_async::allocate_block_spec_decode_input_patch_workspace(
          /*max_rows=*/1,
          /*query_width=*/3,
          /*target_width=*/4,
          position_options,
          position_options);
  const void* query_cache_slot_address =
      target.query_new_cache_slots.data_ptr();
  const void* target_cache_slot_address =
      target.target_new_cache_slots.data_ptr();

  block_spec_async::patch_block_spec_decode_input_geometry(
      continuation, /*block_size=*/4, target, workspace);

  EXPECT_EQ(target.query_new_cache_slots.data_ptr(), query_cache_slot_address);
  EXPECT_EQ(target.target_new_cache_slots.data_ptr(),
            target_cache_slot_address);
  EXPECT_TRUE(torch::equal(target.query_new_cache_slots,
                           torch::tensor({{47, 40, 41}}, position_options)));
  EXPECT_TRUE(
      torch::equal(target.target_new_cache_slots,
                   torch::tensor({{47, 40, 41, 42}}, position_options)));
}

TEST(BlockSpecDecodeInputPatchTest, SupportsChunkedTargetKvLengths) {
  const torch::TensorOptions token_options = cpu_tensor_options(torch::kLong);
  const torch::TensorOptions position_options = cpu_tensor_options(torch::kInt);
  block_spec_async::BlockSpecContinuationState continuation;
  continuation.anchor_tokens = torch::tensor({10, 20}, token_options);
  continuation.base_positions = torch::tensor({3, 6}, position_options);

  block_spec_async::BlockSpecDecodeInputPatchTarget target;
  target.query_token_ids = torch::empty({2, 4}, token_options);
  target.query_positions = torch::empty({2, 4}, position_options);
  target.query_kv_seq_lens = torch::empty({2}, position_options);
  target.query_new_cache_slots = torch::empty({2, 4}, position_options);
  target.target_token_ids = torch::empty({2, 4}, token_options);
  target.target_positions = torch::empty({2, 4}, position_options);
  target.target_kv_seq_lens = torch::empty({2}, position_options);
  target.target_new_cache_slots = torch::empty({2, 4}, position_options);
  target.source_block_tables =
      torch::tensor({{100, 101, 102}, {200, 201, 202}}, position_options);
  block_spec_async::BlockSpecDecodeInputPatchWorkspace workspace =
      block_spec_async::allocate_block_spec_decode_input_patch_workspace(
          /*max_rows=*/2,
          /*query_width=*/4,
          /*target_width=*/4,
          position_options,
          position_options);
  const void* target_kv_length_address = target.target_kv_seq_lens.data_ptr();

  block_spec_async::patch_block_spec_decode_input_geometry(
      continuation, /*block_size=*/4, target, workspace);

  EXPECT_EQ(target.target_kv_seq_lens.data_ptr(), target_kv_length_address);
  EXPECT_TRUE(torch::equal(target.target_kv_seq_lens,
                           torch::tensor({7, 10}, position_options)));
}

TEST(BlockSpecDecodeInputPatchTest,
     SupportsDSparkQueryAndTargetWidthsWithoutReplacingStorage) {
  const torch::TensorOptions token_options = cpu_tensor_options(torch::kLong);
  const torch::TensorOptions position_options = cpu_tensor_options(torch::kInt);
  block_spec_async::BlockSpecContinuationState continuation;
  continuation.anchor_tokens = torch::tensor({10, 20}, token_options);
  continuation.base_positions = torch::tensor({3, 6}, position_options);

  block_spec_async::BlockSpecDecodeInputPatchTarget target;
  target.query_token_ids = torch::full({2, 3}, -1, token_options);
  target.query_positions = torch::empty({2, 3}, position_options);
  target.query_kv_seq_lens = torch::empty({2}, position_options);
  target.query_new_cache_slots = torch::empty({2, 3}, position_options);
  target.target_token_ids = torch::full({2, 4}, -1, token_options);
  target.target_positions = torch::empty({2, 4}, position_options);
  target.target_kv_seq_lens = torch::empty({2}, position_options);
  target.target_new_cache_slots = torch::empty({2, 4}, position_options);
  target.source_block_tables =
      torch::tensor({{100, 101, 102}, {200, 201, 202}}, position_options);
  block_spec_async::BlockSpecDecodeInputPatchWorkspace workspace =
      block_spec_async::allocate_block_spec_decode_input_patch_workspace(
          /*max_rows=*/2,
          /*query_width=*/3,
          /*target_width=*/4,
          position_options,
          position_options);
  const void* query_token_address = target.query_token_ids.data_ptr();
  const void* query_position_address = target.query_positions.data_ptr();
  const void* query_cache_slot_address =
      target.query_new_cache_slots.data_ptr();
  const void* target_token_address = target.target_token_ids.data_ptr();
  const void* target_position_address = target.target_positions.data_ptr();
  const void* target_cache_slot_address =
      target.target_new_cache_slots.data_ptr();

  block_spec_async::patch_block_spec_decode_input_geometry(
      continuation, /*block_size=*/4, target, workspace);
  block_spec_async::patch_block_spec_target_token_ids(
      torch::tensor({{11, 12, 13}, {21, 22, 23}}, token_options),
      target.target_token_ids);

  EXPECT_EQ(target.query_token_ids.data_ptr(), query_token_address);
  EXPECT_EQ(target.query_positions.data_ptr(), query_position_address);
  EXPECT_EQ(target.query_new_cache_slots.data_ptr(), query_cache_slot_address);
  EXPECT_EQ(target.target_token_ids.data_ptr(), target_token_address);
  EXPECT_EQ(target.target_positions.data_ptr(), target_position_address);
  EXPECT_EQ(target.target_new_cache_slots.data_ptr(),
            target_cache_slot_address);
  EXPECT_TRUE(
      torch::equal(target.query_token_ids,
                   torch::tensor({{10, -1, -1}, {20, -1, -1}}, token_options)));
  EXPECT_TRUE(torch::equal(
      target.target_token_ids,
      torch::tensor({{10, 11, 12, 13}, {20, 21, 22, 23}}, token_options)));
  EXPECT_TRUE(
      torch::equal(target.query_positions,
                   torch::tensor({{3, 4, 5}, {6, 7, 8}}, position_options)));
  EXPECT_TRUE(torch::equal(
      target.target_positions,
      torch::tensor({{3, 4, 5, 6}, {6, 7, 8, 9}}, position_options)));
  EXPECT_TRUE(torch::equal(target.query_kv_seq_lens,
                           torch::tensor({6, 9}, position_options)));
  EXPECT_TRUE(torch::equal(target.target_kv_seq_lens,
                           torch::tensor({7, 10}, position_options)));
  EXPECT_TRUE(torch::equal(
      target.query_new_cache_slots,
      torch::tensor({{403, 404, 405}, {806, 807, 808}}, position_options)));
  EXPECT_TRUE(
      torch::equal(target.target_new_cache_slots,
                   torch::tensor({{403, 404, 405, 406}, {806, 807, 808, 809}},
                                 position_options)));
}

TEST(BlockSpecDecodeInputPatchTest,
     RejectsGeometryDeviceAndDtypeMismatchesBeforeWrites) {
  const torch::TensorOptions token_options = cpu_tensor_options(torch::kLong);
  const torch::TensorOptions position_options = cpu_tensor_options(torch::kInt);
  block_spec_async::BlockSpecContinuationState continuation;
  continuation.anchor_tokens = torch::tensor({10}, token_options);
  continuation.base_positions = torch::tensor({3}, position_options);

  block_spec_async::BlockSpecDecodeInputPatchTarget target;
  target.query_token_ids = torch::empty({1, 3}, token_options);
  target.query_positions = torch::empty({1, 3}, position_options);
  target.query_kv_seq_lens = torch::empty({1}, position_options);
  target.query_new_cache_slots = torch::empty({1, 3}, position_options);
  target.target_token_ids = torch::empty({1, 4}, token_options);
  target.target_positions = torch::empty({1, 4}, position_options);
  target.target_kv_seq_lens = torch::empty({1}, position_options);
  target.target_new_cache_slots = torch::empty({1, 4}, position_options);
  target.source_block_tables = torch::tensor({{100, 101}}, position_options);
  block_spec_async::BlockSpecDecodeInputPatchWorkspace workspace =
      block_spec_async::allocate_block_spec_decode_input_patch_workspace(
          /*max_rows=*/1,
          /*query_width=*/3,
          /*target_width=*/4,
          position_options,
          position_options);

  target.query_positions =
      torch::empty({1, 3}, position_options.device(torch::Device("meta")));
  EXPECT_DEATH(block_spec_async::patch_block_spec_decode_input_geometry(
                   continuation, /*block_size=*/4, target, workspace),
               "target.query_positions.*contract device");

  target.query_positions = torch::empty({1, 3}, position_options);
  workspace.target_cache_offsets =
      torch::empty({1, 4}, cpu_tensor_options(torch::kLong));
  EXPECT_DEATH(block_spec_async::patch_block_spec_decode_input_geometry(
                   continuation, /*block_size=*/4, target, workspace),
               "workspace.target_cache_offsets.*incompatible dtype");
}

TEST(BlockSpecDecodeInputPatchTest,
     RejectsTargetTokenDeviceAndDtypeMismatchesBeforeWrites) {
  const torch::Tensor draft_token_ids =
      torch::tensor({{11, 12, 13}}, cpu_tensor_options(torch::kLong));
  torch::Tensor target_token_ids = torch::empty(
      {1, 4}, cpu_tensor_options(torch::kLong).device(torch::Device("meta")));

  EXPECT_DEATH(block_spec_async::patch_block_spec_target_token_ids(
                   draft_token_ids, target_token_ids),
               "target_token_ids.*contract device");

  target_token_ids = torch::empty({1, 4}, cpu_tensor_options(torch::kInt));
  EXPECT_DEATH(block_spec_async::patch_block_spec_target_token_ids(
                   draft_token_ids, target_token_ids),
               "target_token_ids.*incompatible dtype");
}

TEST(BlockSpecPreparedBindingContractTest,
     BindsDSparkQueryNAndTargetNPlusOneCacheSlots) {
  constexpr int32_t kBatchSize = 2;
  constexpr int32_t kSpeculativeWidth = 3;
  const torch::TensorOptions position_options = cpu_tensor_options(torch::kInt);
  const torch::Tensor query_cache_slots =
      torch::arange(kBatchSize * kSpeculativeWidth, position_options);
  const torch::Tensor target_cache_slots =
      torch::arange(kBatchSize * (kSpeculativeWidth + 1), position_options);
  const void* query_address = query_cache_slots.data_ptr();
  const void* target_address = target_cache_slots.data_ptr();

  const dflash_detail::PreparedDecodeCacheSlotViews views =
      dflash_detail::view_prepared_decode_cache_slots(
          query_cache_slots,
          target_cache_slots,
          kBatchSize,
          kSpeculativeWidth,
          /*sample_from_anchor=*/true);

  EXPECT_EQ(views.query.size(0), kBatchSize);
  EXPECT_EQ(views.query.size(1), kSpeculativeWidth);
  EXPECT_EQ(views.target.size(0), kBatchSize);
  EXPECT_EQ(views.target.size(1), kSpeculativeWidth + 1);
  EXPECT_EQ(views.query.data_ptr(), query_address);
  EXPECT_EQ(views.target.data_ptr(), target_address);
}

TEST(BlockSpecPreparedOutputBindingContractTest, AcceptsFixedAddressView) {
  const torch::Tensor fixed_storage =
      torch::empty({6}, cpu_tensor_options(torch::kLong));
  const torch::Tensor actual = fixed_storage.view({2, 3});

  EXPECT_NO_FATAL_FAILURE(dflash_detail::check_fixed_prepared_output_binding(
      actual,
      fixed_storage,
      torch::Device(torch::kCPU),
      torch::kLong,
      /*expected_numel=*/6,
      "Draft token output"));
}

TEST(BlockSpecPreparedOutputBindingContractTest,
     RejectsDeviceDtypeAndAddressMismatches) {
  const torch::TensorOptions token_options = cpu_tensor_options(torch::kLong);
  const torch::Tensor fixed_storage = torch::empty({6}, token_options);

  EXPECT_DEATH(
      dflash_detail::check_fixed_prepared_output_binding(
          torch::empty({6}, token_options.device(torch::Device("meta"))),
          fixed_storage,
          torch::Device(torch::kCPU),
          torch::kLong,
          /*expected_numel=*/6,
          "Target token output"),
      "Target token output.*Worker device");
  EXPECT_DEATH(dflash_detail::check_fixed_prepared_output_binding(
                   torch::empty({6}, cpu_tensor_options(torch::kInt)),
                   fixed_storage,
                   torch::Device(torch::kCPU),
                   torch::kLong,
                   /*expected_numel=*/6,
                   "Target token output"),
               "Target token output.*incompatible dtype");
  EXPECT_DEATH(dflash_detail::check_fixed_prepared_output_binding(
                   torch::empty({6}, token_options),
                   fixed_storage,
                   torch::Device(torch::kCPU),
                   torch::kLong,
                   /*expected_numel=*/6,
                   "Target token output"),
               "Target token output.*replaced the fixed output storage");
}

TEST(DFlashContextKvWriteContractTest, AcceptsFixedWriteTensors) {
  const torch::TensorOptions hidden_options = cpu_tensor_options(torch::kFloat);
  const torch::TensorOptions position_options = cpu_tensor_options(torch::kInt);
  const torch::Tensor context_hidden = torch::zeros({2, 4}, hidden_options);
  const torch::Tensor positions = torch::tensor({3, 7}, position_options);
  const torch::Tensor cache_slots = torch::tensor({103, 207}, position_options);

  EXPECT_NO_FATAL_FAILURE(dflash_detail::check_context_kv_write_tensor_contract(
      context_hidden,
      positions,
      cache_slots,
      torch::Device(torch::kCPU),
      /*expected_hidden_size=*/4,
      torch::kFloat));
}

TEST(DFlashContextKvWriteContractTest,
     RejectsDeviceDtypeAndRowMismatchesBeforeScatter) {
  const torch::TensorOptions hidden_options = cpu_tensor_options(torch::kFloat);
  const torch::TensorOptions position_options = cpu_tensor_options(torch::kInt);
  const torch::Tensor context_hidden = torch::zeros({2, 4}, hidden_options);
  const torch::Tensor positions = torch::tensor({3, 7}, position_options);
  const torch::Tensor cache_slots = torch::tensor({103, 207}, position_options);

  EXPECT_DEATH(
      dflash_detail::check_context_kv_write_tensor_contract(
          torch::empty({2, 4}, hidden_options.device(torch::Device("meta"))),
          positions,
          cache_slots,
          torch::Device(torch::kCPU),
          /*expected_hidden_size=*/4,
          torch::kFloat),
      "context hidden.*contract device");
  EXPECT_DEATH(dflash_detail::check_context_kv_write_tensor_contract(
                   context_hidden,
                   torch::tensor({3, 7}, cpu_tensor_options(torch::kLong)),
                   cache_slots,
                   torch::Device(torch::kCPU),
                   /*expected_hidden_size=*/4,
                   torch::kFloat),
               "context positions.*int32 dtype");
  EXPECT_DEATH(dflash_detail::check_context_kv_write_tensor_contract(
                   context_hidden,
                   positions,
                   torch::tensor({103}, position_options),
                   torch::Device(torch::kCPU),
                   /*expected_hidden_size=*/4,
                   torch::kFloat),
               "context cache slots.*context hidden rows");
}

TEST(DSparkPreparedSamplingWorkspaceTest,
     KeepsStepViewsFixedAndBuildsPreviousTokenChain) {
  const torch::TensorOptions token_options = cpu_tensor_options(torch::kLong);
  const torch::TensorOptions logit_options = cpu_tensor_options(torch::kFloat);
  dspark_detail::PreparedSamplingWorkspace workspace =
      dspark_detail::allocate_prepared_sampling_workspace(
          /*max_rows=*/4,
          /*speculative_width=*/3,
          /*markov_rank=*/2,
          /*draft_vocab_size=*/5,
          token_options,
          logit_options);
  ASSERT_EQ(workspace.steps.size(), 3);
  const void* token_matrix_address = workspace.token_ids.data_ptr();
  const void* previous_token_address = workspace.previous_token_ids.data_ptr();
  const void* greedy_output_address = workspace.greedy_token_outputs.data_ptr();
  const void* proposal_prob_address = workspace.proposal_probs.data_ptr();
  const void* markov_embedding_address = workspace.markov_embeddings.data_ptr();
  const void* markov_bias_address = workspace.markov_bias.data_ptr();
  const void* step_logits_address = workspace.step_logits.data_ptr();

  const torch::Tensor anchor_tokens = torch::tensor({4, 1}, token_options);
  const torch::Tensor base_logits =
      torch::tensor({{{0.0F, 1.0F, 2.0F, 3.0F, 4.0F},
                      {5.0F, 4.0F, 3.0F, 2.0F, 1.0F},
                      {0.0F, 7.0F, 1.0F, 2.0F, 3.0F}},
                     {{9.0F, 1.0F, 2.0F, 3.0F, 4.0F},
                      {0.0F, 1.0F, 8.0F, 2.0F, 3.0F},
                      {0.0F, 1.0F, 2.0F, 9.0F, 3.0F}}},
                    logit_options);
  for (int32_t block_step = 0; block_step < 3; ++block_step) {
    dspark_detail::prepare_sampling_step(
        anchor_tokens, /*row_count=*/2, block_step, workspace);
    dspark_detail::PreparedSamplingStepWorkspace& step =
        workspace.steps[static_cast<size_t>(block_step)];
    torch::Tensor step_logits = step.step_logits.narrow(
        /*dim=*/0, /*start=*/0, /*length=*/2);
    step_logits.copy_(base_logits.select(/*dim=*/1, block_step));
    torch::Tensor greedy_output = step.greedy_token_output.narrow(
        /*dim=*/0, /*start=*/0, /*length=*/2);
    const void* step_output_address = greedy_output.data_ptr();
    torch::argmax_out(greedy_output,
                      step_logits,
                      /*dim=*/-1,
                      /*keepdim=*/false);
    EXPECT_EQ(greedy_output.data_ptr(), step_output_address);
    dspark_detail::commit_sampling_step(
        /*row_count=*/2, block_step, workspace);
    EXPECT_EQ(step.markov_embeddings.data_ptr(), markov_embedding_address);
    EXPECT_EQ(step.markov_bias.data_ptr(), markov_bias_address);
    EXPECT_EQ(step.step_logits.data_ptr(), step_logits_address);
  }

  EXPECT_EQ(workspace.token_ids.data_ptr(), token_matrix_address);
  EXPECT_EQ(workspace.previous_token_ids.data_ptr(), previous_token_address);
  EXPECT_EQ(workspace.greedy_token_outputs.data_ptr(), greedy_output_address);
  EXPECT_EQ(workspace.proposal_probs.data_ptr(), proposal_prob_address);
  EXPECT_TRUE(torch::equal(
      workspace.token_ids.narrow(/*dim=*/0, /*start=*/0, /*length=*/2),
      torch::tensor({{4, 0, 1}, {0, 2, 3}}, token_options)));
  EXPECT_TRUE(torch::equal(
      workspace.previous_token_ids.narrow(/*dim=*/0, /*start=*/0, /*length=*/3)
          .narrow(/*dim=*/1, /*start=*/0, /*length=*/2),
      torch::tensor({{4, 1}, {4, 0}, {0, 2}}, token_options)));
  EXPECT_TRUE(torch::equal(workspace.proposal_probs.narrow(
                               /*dim=*/0, /*start=*/0, /*length=*/2),
                           torch::ones({2, 3}, logit_options)));
}

TEST(DSparkPreparedSamplingWorkspaceTest,
     RejectsDeviceAndDtypeMismatchesBeforeWrites) {
  const torch::TensorOptions token_options = cpu_tensor_options(torch::kLong);
  const torch::TensorOptions logit_options = cpu_tensor_options(torch::kFloat);
  dspark_detail::PreparedSamplingWorkspace workspace =
      dspark_detail::allocate_prepared_sampling_workspace(
          /*max_rows=*/4,
          /*speculative_width=*/3,
          /*markov_rank=*/2,
          /*draft_vocab_size=*/5,
          token_options,
          logit_options);
  const torch::Tensor anchor_tokens = torch::tensor({4, 1}, token_options);

  EXPECT_DEATH(
      dspark_detail::prepare_sampling_step(
          torch::empty({2}, token_options.device(torch::Device("meta"))),
          /*row_count=*/2,
          /*block_step=*/0,
          workspace),
      "anchor_token_ids.*contract device");

  workspace.steps[0].step_logits =
      torch::empty({4, 5}, logit_options.dtype(torch::kDouble));
  EXPECT_DEATH(dspark_detail::prepare_sampling_step(anchor_tokens,
                                                    /*row_count=*/2,
                                                    /*block_step=*/0,
                                                    workspace),
               "step.step_logits.*incompatible dtype");

  workspace.steps[0].step_logits = workspace.step_logits;
  workspace.proposal_probs =
      torch::empty({4, 3}, logit_options.dtype(torch::kDouble));
  EXPECT_DEATH(dspark_detail::reset_prepared_sampling_workspace(workspace),
               "workspace.proposal_probs.*incompatible dtype");

  workspace.proposal_probs =
      torch::empty({4, 3}, logit_options.dtype(torch::kFloat32));
  workspace.steps[0].greedy_token_output =
      torch::empty({4}, token_options.device(torch::Device("meta")));
  EXPECT_DEATH(dspark_detail::commit_sampling_step(
                   /*row_count=*/2, /*block_step=*/0, workspace),
               "step.greedy_token_output.*contract device");
}

TEST(BlockSpecPreparedTaskBackendBaseTest,
     OwnsFixedStateAndRecordsPredecessorReuseFence) {
  FixedBufferBlockSpecPreparedTaskBackend backend;
  const BlockSpecPreparedTaskPlan decode_plan =
      build_block_spec_prepared_task_plan(BlockSpecAlgorithm::DFLASH,
                                          PreparedTaskKind::DECODE,
                                          /*speculative_width=*/3);

  ExecutionSlot predecessor_slot;
  predecessor_slot.slot_id = 0;
  ForwardInput predecessor_input =
      make_input(BatchForwardType::DECODE, /*batch_id=*/89);
  predecessor_input.token_ids_host =
      torch::tensor({1, 2}, cpu_tensor_options(torch::kInt));
  predecessor_input.positions_host =
      torch::tensor({5, 7}, cpu_tensor_options(torch::kInt));
  const void* state_address =
      backend.published_state(0).accepted_tokens.data_ptr();
  backend.prepare(predecessor_input, predecessor_slot, decode_plan);
  backend.launch(decode_plan.invocations[5],
                 predecessor_slot,
                 /*predecessor_slot=*/nullptr);
  EXPECT_EQ(backend.published_state(0).accepted_tokens.data_ptr(),
            state_address);
  EXPECT_TRUE(
      torch::equal(backend.published_state(0).accepted_lengths,
                   torch::tensor({2, 3, 0}, cpu_tensor_options(torch::kLong))));

  ExecutionSlot current_slot;
  current_slot.slot_id = 1;
  ForwardInput current_input =
      make_input(BatchForwardType::DECODE, /*batch_id=*/90);
  current_input.positions_host =
      torch::tensor({6, 8, 9}, cpu_tensor_options(torch::kInt));
  current_input.token_ids_host =
      torch::tensor({3, 4, 5}, cpu_tensor_options(torch::kInt));
  current_input.input_params.embedding.predecessor_rows =
      torch::tensor({1, -1, 0}, cpu_tensor_options(torch::kLong));
  backend.prepare(current_input, current_slot, decode_plan);
  backend.launch(
      decode_plan.invocations.front(), current_slot, &predecessor_slot);

  EXPECT_EQ(backend.predecessor_patch_count(), 1);
  EXPECT_TRUE(torch::equal(
      backend.continuation(1).anchor_tokens,
      torch::tensor({22, 91, 11}, cpu_tensor_options(torch::kLong))));
  EXPECT_TRUE(torch::equal(
      backend.continuation(1).base_positions,
      torch::tensor({10, 31, 7}, cpu_tensor_options(torch::kInt))));
  EXPECT_NE(predecessor_slot.reuse_fence, nullptr);
  backend.prepare_slot_for_reuse(predecessor_slot);
  EXPECT_EQ(backend.reuse_wait_count(), 1);
  const std::vector<BlockSpecPreparedInvocationKind> launches =
      backend.launches();
  ASSERT_EQ(launches.size(), 2);
  EXPECT_EQ(launches[0],
            BlockSpecPreparedInvocationKind::PUBLISH_CONTEXT_KV_STATE);
  EXPECT_EQ(launches[1],
            BlockSpecPreparedInvocationKind::TASK_CONTINUATION_PATCH);
}

TEST(BlockSpecPreparedTaskBackendBaseTest,
     RejectsPublishSourceOnDifferentDeviceAtBinding) {
  FixedBufferBlockSpecPreparedTaskBackend backend;
  backend.publish_source(0).accepted_context_hidden = torch::empty(
      {2, 4, 2},
      cpu_tensor_options(torch::kFloat).device(torch::Device("meta")));

  EXPECT_DEATH(backend.bind_publish_source_for_test(/*slot_id=*/0),
               "accepted context hidden.*share a device");
}

TEST(BlockSpecPreparedTaskBackendBaseTest,
     RejectsPublishSourceWithIncompatibleDtypeAtBinding) {
  FixedBufferBlockSpecPreparedTaskBackend backend;
  backend.publish_source(0).block_tables =
      torch::empty({2, 4}, cpu_tensor_options(torch::kLong));

  EXPECT_DEATH(backend.bind_publish_source_for_test(/*slot_id=*/0),
               "block tables.*share a dtype");
}

TEST(BlockSpecPreparedTaskPlanTest, BuildsDFlashFixedTopology) {
  const BlockSpecPreparedTaskPlan plan =
      build_block_spec_prepared_task_plan(BlockSpecAlgorithm::DFLASH,
                                          PreparedTaskKind::DECODE,
                                          /*speculative_width=*/3);

  EXPECT_EQ(plan.input_partition_count, 3);
  const std::vector<BlockSpecPreparedInvocationKind> expected_kinds = {
      BlockSpecPreparedInvocationKind::TASK_CONTINUATION_PATCH,
      BlockSpecPreparedInvocationKind::BLOCK_DRAFT,
      BlockSpecPreparedInvocationKind::TARGET_VALIDATE_PATCH,
      BlockSpecPreparedInvocationKind::TARGET_VALIDATE,
      BlockSpecPreparedInvocationKind::REJECTION_SAMPLE,
      BlockSpecPreparedInvocationKind::PUBLISH_CONTEXT_KV_STATE,
      BlockSpecPreparedInvocationKind::WRITE_CONTEXT_KV};
  ASSERT_EQ(plan.invocations.size(), expected_kinds.size());
  for (size_t invocation_index = 0; invocation_index < expected_kinds.size();
       ++invocation_index) {
    EXPECT_EQ(plan.invocations[invocation_index].kind,
              expected_kinds[invocation_index]);
    EXPECT_EQ(plan.invocations[invocation_index].invocation_index,
              static_cast<int32_t>(invocation_index));
  }
  EXPECT_EQ(plan.invocations[0].input_partition, 1);
  EXPECT_EQ(plan.invocations[1].input_partition, 1);
  EXPECT_EQ(plan.invocations[2].input_partition, 2);
  EXPECT_EQ(plan.invocations[3].input_partition, 2);
}

TEST(BlockSpecPreparedTaskPlanTest, BuildsDSparkMarkovPairsAtFixedWidth) {
  const BlockSpecPreparedTaskPlan plan =
      build_block_spec_prepared_task_plan(BlockSpecAlgorithm::DSPARK,
                                          PreparedTaskKind::DECODE,
                                          /*speculative_width=*/3);

  ASSERT_EQ(plan.invocations.size(), 13);
  for (int32_t block_step = 0; block_step < 3; ++block_step) {
    const size_t sample_index = static_cast<size_t>(2 + block_step * 2);
    EXPECT_EQ(plan.invocations[sample_index].kind,
              BlockSpecPreparedInvocationKind::DSPARK_MARKOV_SAMPLE);
    EXPECT_EQ(plan.invocations[sample_index].block_step, block_step);
    EXPECT_EQ(plan.invocations[sample_index + 1].kind,
              BlockSpecPreparedInvocationKind::DSPARK_TOKEN_BROADCAST);
    EXPECT_EQ(plan.invocations[sample_index + 1].block_step, block_step);
  }
  EXPECT_EQ(plan.invocations[8].kind,
            BlockSpecPreparedInvocationKind::TARGET_VALIDATE_PATCH);
  EXPECT_EQ(plan.invocations.back().kind,
            BlockSpecPreparedInvocationKind::WRITE_CONTEXT_KV);
}

TEST(BlockSpecPreparedTaskPlanTest, BuildsPrefillAndEmptyTopologies) {
  const BlockSpecPreparedTaskPlan prefill_plan =
      build_block_spec_prepared_task_plan(BlockSpecAlgorithm::DFLASH,
                                          PreparedTaskKind::PREFILL_LIKE,
                                          /*speculative_width=*/3);
  ASSERT_EQ(prefill_plan.invocations.size(), 2);
  EXPECT_EQ(prefill_plan.invocations[0].kind,
            BlockSpecPreparedInvocationKind::TARGET_PREFILL);
  EXPECT_EQ(prefill_plan.invocations[1].kind,
            BlockSpecPreparedInvocationKind::WRITE_CONTEXT_KV);

  const BlockSpecPreparedTaskPlan empty_plan =
      build_block_spec_prepared_task_plan(BlockSpecAlgorithm::DSPARK,
                                          PreparedTaskKind::EMPTY,
                                          /*speculative_width=*/3);
  ASSERT_EQ(empty_plan.invocations.size(), 1);
  EXPECT_EQ(empty_plan.invocations[0].kind,
            BlockSpecPreparedInvocationKind::EMPTY_COLLECTIVE);
}

TEST(BlockSpecPreparedTaskAdapterTest,
     LaunchesOneContinuationAndFixedDSparkLoop) {
  auto backend = std::make_unique<RecordingBlockSpecPreparedTaskBackend>();
  RecordingBlockSpecPreparedTaskBackend* backend_ptr = backend.get();
  auto adapter =
      std::make_unique<BlockSpecPreparedTaskAdapter>(std::move(backend),
                                                     BlockSpecAlgorithm::DSPARK,
                                                     /*speculative_width=*/3,
                                                     /*slot_count=*/1);
  PreparedTaskPipeline pipeline(std::move(adapter), /*slot_count=*/1);

  ForwardInput prefill_input =
      make_input(BatchForwardType::PREFILL, /*batch_id=*/87);
  ASSERT_TRUE(std::move(pipeline.submit(prefill_input)).get().has_value());

  ForwardInput decode_input =
      make_input(BatchForwardType::DECODE, /*batch_id=*/88);
  decode_input.input_params.embedding.predecessor_rows =
      torch::tensor({0}, cpu_tensor_options(torch::kLong));
  std::optional<ForwardOutput> decode_output =
      std::move(pipeline.submit(decode_input)).get();
  ASSERT_TRUE(decode_output.has_value());
  EXPECT_EQ(decode_output->prepared_token, 88);

  const std::vector<size_t> plan_sizes = backend_ptr->prepared_plan_sizes();
  ASSERT_EQ(plan_sizes.size(), 2);
  EXPECT_EQ(plan_sizes[0], 2);
  EXPECT_EQ(plan_sizes[1], 13);
  const std::vector<BlockSpecBackendLaunch> launches = backend_ptr->launches();
  ASSERT_EQ(launches.size(), 15);
  EXPECT_EQ(launches[2].kind,
            BlockSpecPreparedInvocationKind::TASK_CONTINUATION_PATCH);
  EXPECT_EQ(launches[2].predecessor_slot_id, 0);
  EXPECT_EQ(
      std::count_if(
          launches.begin(),
          launches.end(),
          [](const BlockSpecBackendLaunch& launch) {
            return launch.kind ==
                   BlockSpecPreparedInvocationKind::TASK_CONTINUATION_PATCH;
          }),
      1);
  EXPECT_EQ(std::count_if(
                launches.begin(),
                launches.end(),
                [](const BlockSpecBackendLaunch& launch) {
                  return launch.kind ==
                         BlockSpecPreparedInvocationKind::DSPARK_MARKOV_SAMPLE;
                }),
            3);
}

TEST(BlockSpecPreparedTaskAdapterTest,
     AlternatesTwoSlotsAndReadsTheImmediatelyPreviousTask) {
  auto backend = std::make_unique<RecordingBlockSpecPreparedTaskBackend>();
  RecordingBlockSpecPreparedTaskBackend* backend_ptr = backend.get();
  auto adapter =
      std::make_unique<BlockSpecPreparedTaskAdapter>(std::move(backend),
                                                     BlockSpecAlgorithm::DSPARK,
                                                     /*speculative_width=*/3,
                                                     /*slot_count=*/2);
  PreparedTaskPipeline pipeline(std::move(adapter), /*slot_count=*/2);

  ForwardInput first_input =
      make_input(BatchForwardType::DECODE, /*batch_id=*/201);
  EXPECT_FALSE(std::move(pipeline.submit(first_input)).get().has_value());

  ForwardInput second_input =
      make_input(BatchForwardType::DECODE, /*batch_id=*/202);
  second_input.input_params.embedding.predecessor_rows =
      torch::tensor({0}, cpu_tensor_options(torch::kLong));
  EXPECT_FALSE(std::move(pipeline.submit(second_input)).get().has_value());

  std::optional<ForwardOutput> first_output =
      std::move(pipeline.get_last_step_result()).get();
  ASSERT_TRUE(first_output.has_value());
  EXPECT_EQ(first_output->prepared_token, 201);

  ForwardInput third_input =
      make_input(BatchForwardType::DECODE, /*batch_id=*/203);
  third_input.input_params.embedding.predecessor_rows =
      torch::tensor({0}, cpu_tensor_options(torch::kLong));
  EXPECT_FALSE(std::move(pipeline.submit(third_input)).get().has_value());

  for (int64_t expected_token = 202; expected_token <= 203; ++expected_token) {
    std::optional<ForwardOutput> output =
        std::move(pipeline.get_last_step_result()).get();
    ASSERT_TRUE(output.has_value());
    EXPECT_EQ(output->prepared_token, expected_token);
  }

  const std::vector<BlockSpecBackendLaunch> launches = backend_ptr->launches();
  constexpr size_t kDecodeInvocationCount = 13;
  ASSERT_EQ(launches.size(), 3 * kDecodeInvocationCount);
  const BlockSpecBackendLaunch& second_continuation =
      launches[kDecodeInvocationCount];
  EXPECT_EQ(second_continuation.task_seq_no, 1);
  EXPECT_EQ(second_continuation.slot_id, 1);
  EXPECT_EQ(second_continuation.predecessor_slot_id, 0);
  EXPECT_EQ(second_continuation.kind,
            BlockSpecPreparedInvocationKind::TASK_CONTINUATION_PATCH);
  const BlockSpecBackendLaunch& third_continuation =
      launches[2 * kDecodeInvocationCount];
  EXPECT_EQ(third_continuation.task_seq_no, 2);
  EXPECT_EQ(third_continuation.slot_id, 0);
  EXPECT_EQ(third_continuation.predecessor_slot_id, 1);
  EXPECT_EQ(third_continuation.kind,
            BlockSpecPreparedInvocationKind::TASK_CONTINUATION_PATCH);
}

TEST(MtpPreparedTaskPlanTest, BuildsFixedPrefillTopology) {
  const MtpPreparedTaskPlan plan = build_mtp_prepared_task_plan(
      PreparedTaskKind::PREFILL_LIKE, /*num_speculative_tokens=*/3);

  EXPECT_EQ(plan.input_partition_count, 2);
  ASSERT_EQ(plan.invocations.size(), 4);
  EXPECT_EQ(plan.invocations[0].kind,
            MtpPreparedInvocationKind::TARGET_PREFILL);
  EXPECT_EQ(plan.invocations[1].kind,
            MtpPreparedInvocationKind::PREFILL_TO_DRAFT_PATCH);
  EXPECT_EQ(plan.invocations[2].kind, MtpPreparedInvocationKind::DRAFT_PREFILL);
  EXPECT_EQ(plan.invocations[2].draft_step, 0);
  EXPECT_EQ(plan.invocations[0].input_partition, 0);
  EXPECT_EQ(plan.invocations[1].input_partition, 1);
  EXPECT_EQ(plan.invocations[2].input_partition, 1);
  EXPECT_EQ(plan.invocations[3].kind,
            MtpPreparedInvocationKind::PUBLISH_STEP_STATE);
}

TEST(MtpPreparedTaskPlanTest, BuildsFixedDecodeTopology) {
  const MtpPreparedTaskPlan plan = build_mtp_prepared_task_plan(
      PreparedTaskKind::DECODE, /*num_speculative_tokens=*/3);

  EXPECT_EQ(plan.input_partition_count, 5);
  const std::vector<MtpPreparedInvocationKind> expected_kinds = {
      MtpPreparedInvocationKind::TASK_CONTINUATION_PATCH,
      MtpPreparedInvocationKind::DRAFT_DECODE,
      MtpPreparedInvocationKind::NEXT_DRAFT_PATCH,
      MtpPreparedInvocationKind::DRAFT_DECODE,
      MtpPreparedInvocationKind::NEXT_DRAFT_PATCH,
      MtpPreparedInvocationKind::DRAFT_DECODE,
      MtpPreparedInvocationKind::TARGET_VERIFY_PATCH,
      MtpPreparedInvocationKind::TARGET_VERIFY,
      MtpPreparedInvocationKind::REJECTION_SAMPLE,
      MtpPreparedInvocationKind::PUBLISH_STEP_STATE};
  ASSERT_EQ(plan.invocations.size(), expected_kinds.size());
  for (size_t invocation_index = 0; invocation_index < expected_kinds.size();
       ++invocation_index) {
    EXPECT_EQ(plan.invocations[invocation_index].kind,
              expected_kinds[invocation_index]);
    EXPECT_EQ(plan.invocations[invocation_index].invocation_index,
              static_cast<int32_t>(invocation_index));
  }
  EXPECT_EQ(plan.invocations[1].draft_step, 0);
  EXPECT_EQ(plan.invocations[1].input_partition, 1);
  EXPECT_EQ(plan.invocations[2].draft_step, 1);
  EXPECT_EQ(plan.invocations[2].input_partition, 2);
  EXPECT_EQ(plan.invocations[3].draft_step, 1);
  EXPECT_EQ(plan.invocations[3].input_partition, 2);
  EXPECT_EQ(plan.invocations[4].draft_step, 2);
  EXPECT_EQ(plan.invocations[4].input_partition, 3);
  EXPECT_EQ(plan.invocations[5].draft_step, 2);
  EXPECT_EQ(plan.invocations[5].input_partition, 3);
  EXPECT_EQ(plan.invocations[6].input_partition, 4);
  EXPECT_EQ(plan.invocations[7].input_partition, 4);
}

TEST(MtpPreparedTaskPlanTest, BuildsFixedEmptyTopology) {
  const MtpPreparedTaskPlan plan = build_mtp_prepared_task_plan(
      PreparedTaskKind::EMPTY, /*num_speculative_tokens=*/3);

  EXPECT_EQ(plan.input_partition_count, 1);
  ASSERT_EQ(plan.invocations.size(), 1);
  EXPECT_EQ(plan.invocations[0].kind,
            MtpPreparedInvocationKind::EMPTY_COLLECTIVE);
  EXPECT_EQ(plan.invocations[0].invocation_index, 0);
  EXPECT_EQ(plan.invocations[0].input_partition, 0);
}

TEST(MtpPreparedTaskAdapterTest,
     LaunchesPrefillAndDecodeWithOneContinuationPatch) {
  auto backend = std::make_unique<RecordingMtpPreparedTaskBackend>();
  RecordingMtpPreparedTaskBackend* backend_ptr = backend.get();
  auto adapter =
      std::make_unique<MtpPreparedTaskAdapter>(std::move(backend),
                                               /*num_speculative_tokens=*/3,
                                               /*slot_count=*/1);
  PreparedTaskPipeline pipeline(std::move(adapter), /*slot_count=*/1);

  ForwardInput prefill_input =
      make_input(BatchForwardType::PREFILL, /*batch_id=*/91);
  std::optional<ForwardOutput> prefill_output =
      std::move(pipeline.submit(prefill_input)).get();
  ASSERT_TRUE(prefill_output.has_value());
  EXPECT_EQ(prefill_output->prepared_token, 91);

  ForwardInput decode_input =
      make_input(BatchForwardType::DECODE, /*batch_id=*/92);
  decode_input.input_params.embedding.predecessor_rows =
      torch::tensor({0}, torch::kLong);
  std::optional<ForwardOutput> decode_output =
      std::move(pipeline.submit(decode_input)).get();
  ASSERT_TRUE(decode_output.has_value());
  EXPECT_EQ(decode_output->prepared_token, 92);

  const std::vector<size_t> prepared_plan_sizes =
      backend_ptr->prepared_plan_sizes();
  ASSERT_EQ(prepared_plan_sizes.size(), 2);
  EXPECT_EQ(prepared_plan_sizes[0], 4);
  EXPECT_EQ(prepared_plan_sizes[1], 10);

  const std::vector<MtpBackendLaunch> launches = backend_ptr->launches();
  ASSERT_EQ(launches.size(), 14);
  EXPECT_EQ(launches[4].kind,
            MtpPreparedInvocationKind::TASK_CONTINUATION_PATCH);
  EXPECT_EQ(launches[4].task_seq_no, 1);
  EXPECT_EQ(launches[4].slot_id, 0);
  EXPECT_EQ(launches[4].predecessor_slot_id, 0);
  EXPECT_EQ(
      std::count_if(launches.begin(),
                    launches.end(),
                    [](const MtpBackendLaunch& launch) {
                      return launch.kind ==
                             MtpPreparedInvocationKind::TASK_CONTINUATION_PATCH;
                    }),
      1);
  EXPECT_EQ(launches.back().kind,
            MtpPreparedInvocationKind::PUBLISH_STEP_STATE);
}

TEST(MtpPreparedTaskAdapterTest,
     AlternatesTwoSlotsAndReadsTheImmediatelyPreviousTask) {
  auto backend = std::make_unique<RecordingMtpPreparedTaskBackend>();
  RecordingMtpPreparedTaskBackend* backend_ptr = backend.get();
  auto adapter =
      std::make_unique<MtpPreparedTaskAdapter>(std::move(backend),
                                               /*num_speculative_tokens=*/3,
                                               /*slot_count=*/2);
  PreparedTaskPipeline pipeline(std::move(adapter), /*slot_count=*/2);

  ForwardInput first_input =
      make_input(BatchForwardType::DECODE, /*batch_id=*/301);
  EXPECT_FALSE(std::move(pipeline.submit(first_input)).get().has_value());

  ForwardInput second_input =
      make_input(BatchForwardType::DECODE, /*batch_id=*/302);
  second_input.input_params.embedding.predecessor_rows =
      torch::tensor({0}, cpu_tensor_options(torch::kLong));
  EXPECT_FALSE(std::move(pipeline.submit(second_input)).get().has_value());

  std::optional<ForwardOutput> first_output =
      std::move(pipeline.get_last_step_result()).get();
  ASSERT_TRUE(first_output.has_value());
  EXPECT_EQ(first_output->prepared_token, 301);

  ForwardInput third_input =
      make_input(BatchForwardType::DECODE, /*batch_id=*/303);
  third_input.input_params.embedding.predecessor_rows =
      torch::tensor({0}, cpu_tensor_options(torch::kLong));
  EXPECT_FALSE(std::move(pipeline.submit(third_input)).get().has_value());

  for (int64_t expected_token = 302; expected_token <= 303; ++expected_token) {
    std::optional<ForwardOutput> output =
        std::move(pipeline.get_last_step_result()).get();
    ASSERT_TRUE(output.has_value());
    EXPECT_EQ(output->prepared_token, expected_token);
  }

  const std::vector<MtpBackendLaunch> launches = backend_ptr->launches();
  constexpr size_t kDecodeInvocationCount = 10;
  ASSERT_EQ(launches.size(), 3 * kDecodeInvocationCount);
  const MtpBackendLaunch& second_continuation =
      launches[kDecodeInvocationCount];
  EXPECT_EQ(second_continuation.task_seq_no, 1);
  EXPECT_EQ(second_continuation.slot_id, 1);
  EXPECT_EQ(second_continuation.predecessor_slot_id, 0);
  EXPECT_EQ(second_continuation.kind,
            MtpPreparedInvocationKind::TASK_CONTINUATION_PATCH);
  const MtpBackendLaunch& third_continuation =
      launches[2 * kDecodeInvocationCount];
  EXPECT_EQ(third_continuation.task_seq_no, 2);
  EXPECT_EQ(third_continuation.slot_id, 0);
  EXPECT_EQ(third_continuation.predecessor_slot_id, 1);
  EXPECT_EQ(third_continuation.kind,
            MtpPreparedInvocationKind::TASK_CONTINUATION_PATCH);
}

TEST(MtpPreparedTaskBackendBaseTest,
     StagesFixedArenaAndDispatchesModelInvocations) {
  FixedBufferMtpPreparedTaskBackend backend;
  const MtpPreparedTaskPlan decode_plan = build_mtp_prepared_task_plan(
      PreparedTaskKind::DECODE, /*num_speculative_tokens=*/3);

  ExecutionSlot slot;
  slot.slot_id = 0;
  ForwardInput input = make_input(BatchForwardType::DECODE, /*batch_id=*/101);
  input.token_ids_host =
      torch::tensor({1, 2, 3}, cpu_tensor_options(torch::kInt));
  input.positions_host =
      torch::tensor({10, 20, 30}, cpu_tensor_options(torch::kInt));
  input.input_params.embedding.predecessor_rows =
      torch::tensor({-1, -1, -1}, cpu_tensor_options(torch::kLong));
  const void* source_predecessor_rows_address =
      input.input_params.embedding.predecessor_rows.data_ptr();
  const void* step_state_address =
      backend.published_state(0).accepted_tokens.data_ptr();
  backend.prepare(input, slot, decode_plan);

  EXPECT_TRUE(slot.prepared_input.device_tensors_ready);
  ASSERT_TRUE(
      slot.prepared_input.input_params.embedding.predecessor_rows.defined());
  EXPECT_TRUE(
      slot.prepared_input.input_params.embedding.predecessor_rows.device()
          .is_cpu());
  EXPECT_NE(
      slot.prepared_input.input_params.embedding.predecessor_rows.data_ptr(),
      source_predecessor_rows_address);
  EXPECT_GT(slot.prepared_input.prepared_input_layout_signature, 0);
  EXPECT_EQ(backend.published_state(0).accepted_tokens.data_ptr(),
            step_state_address);

  backend.launch(decode_plan.invocations.front(),
                 slot,
                 /*predecessor_slot=*/nullptr);
  backend.launch(decode_plan.invocations[1],
                 slot,
                 /*predecessor_slot=*/nullptr);
  slot.reuse_fence = make_unrecorded_test_event();
  backend.prepare_slot_for_reuse(slot);
  EXPECT_EQ(backend.reuse_wait_count(), 1);
  const std::vector<MtpPreparedInvocationKind> launches = backend.launches();
  ASSERT_EQ(launches.size(), 2);
  EXPECT_EQ(launches[0], MtpPreparedInvocationKind::TASK_CONTINUATION_PATCH);
  EXPECT_EQ(launches[1], MtpPreparedInvocationKind::DRAFT_DECODE);
}

TEST(MtpPreparedTaskBackendBaseTest,
     PublishesAndConsumesReorderedPredecessorStateAcrossSlots) {
  FixedBufferMtpPreparedTaskBackend backend;
  const MtpPreparedTaskPlan decode_plan = build_mtp_prepared_task_plan(
      PreparedTaskKind::DECODE, /*num_speculative_tokens=*/3);

  MtpPreparedPublishSource& predecessor_source = backend.publish_source(0);
  predecessor_source.accepted_tokens =
      torch::tensor({{10, 11, -1, -1}, {20, 21, 22, -1}, {30, -1, -1, -1}},
                    cpu_tensor_options(torch::kLong));
  predecessor_source.accepted_embeddings =
      torch::arange(/*start=*/0,
                    /*end=*/24,
                    cpu_tensor_options(torch::kFloat))
          .view({3, 4, 2})
          .contiguous();
  predecessor_source.embedding_placeholder =
      torch::tensor({-100.0F, -101.0F}, cpu_tensor_options(torch::kFloat));
  predecessor_source.base_positions =
      torch::tensor({5, 7, 9}, cpu_tensor_options(torch::kInt));
  predecessor_source.base_kv_seq_lens =
      torch::tensor({6, 8, 10}, cpu_tensor_options(torch::kInt));

  ExecutionSlot predecessor_slot;
  predecessor_slot.slot_id = 0;
  ForwardInput predecessor_input =
      make_input(BatchForwardType::DECODE, /*batch_id=*/401);
  predecessor_input.token_ids_host =
      torch::tensor({1, 2, 3}, cpu_tensor_options(torch::kInt));
  predecessor_input.positions_host =
      torch::tensor({5, 7, 9}, cpu_tensor_options(torch::kInt));
  predecessor_input.input_params.embedding.predecessor_rows =
      torch::tensor({-1, -1, -1}, cpu_tensor_options(torch::kLong));
  const void* state_tokens_address =
      backend.published_state(0).accepted_tokens.data_ptr();
  backend.prepare(predecessor_input, predecessor_slot, decode_plan);
  backend.launch(decode_plan.invocations.back(),
                 predecessor_slot,
                 /*predecessor_slot=*/nullptr);

  const mtp_async::MtpDeviceStepState& published = backend.published_state(0);
  EXPECT_EQ(published.accepted_tokens.data_ptr(), state_tokens_address);
  EXPECT_TRUE(
      torch::equal(published.accepted_lengths,
                   torch::tensor({2, 3, 1}, cpu_tensor_options(torch::kLong))));
  EXPECT_TRUE(torch::equal(
      published.tail_tokens,
      torch::tensor({11, 22, 30}, cpu_tensor_options(torch::kLong))));
  EXPECT_TRUE(torch::equal(
      published.previous_tokens,
      torch::tensor({10, 21, 30}, cpu_tensor_options(torch::kLong))));
  EXPECT_TRUE(torch::equal(
      published.base_positions,
      torch::tensor({7, 10, 10}, cpu_tensor_options(torch::kInt))));
  EXPECT_TRUE(torch::equal(
      published.base_kv_seq_lens,
      torch::tensor({8, 11, 11}, cpu_tensor_options(torch::kInt))));

  mtp_async::MtpContinuationState& continuation = backend.continuation(1);
  continuation.tail_tokens.copy_(
      torch::tensor({90, 91, 92}, cpu_tensor_options(torch::kLong)));
  continuation.previous_tokens.copy_(
      torch::tensor({80, 81, 82}, cpu_tensor_options(torch::kLong)));
  continuation.tail_embeddings.copy_(
      torch::tensor({{900.0F, 901.0F}, {910.0F, 911.0F}, {920.0F, 921.0F}},
                    cpu_tensor_options(torch::kFloat)));
  continuation.previous_embeddings.copy_(
      torch::tensor({{800.0F, 801.0F}, {810.0F, 811.0F}, {820.0F, 821.0F}},
                    cpu_tensor_options(torch::kFloat)));
  continuation.base_positions.copy_(
      torch::tensor({30, 31, 32}, cpu_tensor_options(torch::kInt)));
  continuation.base_kv_seq_lens.copy_(
      torch::tensor({40, 41, 42}, cpu_tensor_options(torch::kInt)));
  const void* continuation_tokens_address = continuation.tail_tokens.data_ptr();

  ExecutionSlot current_slot;
  current_slot.slot_id = 1;
  ForwardInput current_input =
      make_input(BatchForwardType::DECODE, /*batch_id=*/402);
  current_input.token_ids_host =
      torch::tensor({4, 5, 6}, cpu_tensor_options(torch::kInt));
  current_input.positions_host =
      torch::tensor({30, 31, 32}, cpu_tensor_options(torch::kInt));
  current_input.input_params.embedding.predecessor_rows =
      torch::tensor({1, -1, 0}, cpu_tensor_options(torch::kLong));
  backend.prepare(current_input, current_slot, decode_plan);
  backend.launch(
      decode_plan.invocations.front(), current_slot, &predecessor_slot);

  EXPECT_EQ(continuation.tail_tokens.data_ptr(), continuation_tokens_address);
  EXPECT_TRUE(torch::equal(
      continuation.tail_tokens,
      torch::tensor({22, 91, 11}, cpu_tensor_options(torch::kLong))));
  EXPECT_TRUE(torch::equal(
      continuation.previous_tokens,
      torch::tensor({21, 81, 10}, cpu_tensor_options(torch::kLong))));
  EXPECT_TRUE(torch::equal(
      continuation.tail_embeddings,
      torch::tensor({{12.0F, 13.0F}, {910.0F, 911.0F}, {2.0F, 3.0F}},
                    cpu_tensor_options(torch::kFloat))));
  EXPECT_TRUE(torch::equal(
      continuation.previous_embeddings,
      torch::tensor({{10.0F, 11.0F}, {810.0F, 811.0F}, {0.0F, 1.0F}},
                    cpu_tensor_options(torch::kFloat))));
  EXPECT_TRUE(torch::equal(
      continuation.base_positions,
      torch::tensor({10, 31, 7}, cpu_tensor_options(torch::kInt))));
  EXPECT_TRUE(torch::equal(
      continuation.base_kv_seq_lens,
      torch::tensor({11, 41, 8}, cpu_tensor_options(torch::kInt))));
  EXPECT_NE(predecessor_slot.reuse_fence, nullptr);
  backend.prepare_slot_for_reuse(predecessor_slot);
  EXPECT_EQ(backend.reuse_wait_count(), 1);
}

TEST(MtpPreparedTaskBackendBaseTest,
     RejectsAcceptedCapacityDifferentFromVerifyWidth) {
  MtpPreparedTaskBufferConfig config = make_mtp_backend_buffer_config();
  config.max_verify_width = config.accepted_token_capacity + 1;

  EXPECT_DEATH(FixedBufferMtpPreparedTaskBackend backend(config),
               "must match the maximum Target Verify width");
}

TEST(MtpPreparedTaskBackendBaseTest,
     RejectsPublishPlaceholderOnDifferentDevice) {
  FixedBufferMtpPreparedTaskBackend backend;
  backend.publish_source(0).embedding_placeholder = torch::zeros(
      {2}, cpu_tensor_options(torch::kFloat).device(torch::Device("meta")));
  const MtpPreparedTaskPlan decode_plan = build_mtp_prepared_task_plan(
      PreparedTaskKind::DECODE, /*num_speculative_tokens=*/3);
  ExecutionSlot slot;
  slot.slot_id = 0;
  ForwardInput input = make_input(BatchForwardType::DECODE, /*batch_id=*/501);
  input.token_ids_host = torch::tensor({1}, cpu_tensor_options(torch::kInt));
  input.positions_host = torch::tensor({5}, cpu_tensor_options(torch::kInt));
  input.input_params.embedding.predecessor_rows =
      torch::tensor({-1}, cpu_tensor_options(torch::kLong));
  backend.prepare(input, slot, decode_plan);

  EXPECT_DEATH(backend.launch(decode_plan.invocations.back(),
                              slot,
                              /*predecessor_slot=*/nullptr),
               "embedding placeholder.*share a device");
}

TEST(MtpPreparedTaskBackendBaseTest,
     RejectsPublishBaseKvLengthsWithIncompatibleDtype) {
  FixedBufferMtpPreparedTaskBackend backend;
  backend.publish_source(0).base_kv_seq_lens =
      torch::zeros({3}, cpu_tensor_options(torch::kLong));
  const MtpPreparedTaskPlan decode_plan = build_mtp_prepared_task_plan(
      PreparedTaskKind::DECODE, /*num_speculative_tokens=*/3);
  ExecutionSlot slot;
  slot.slot_id = 0;
  ForwardInput input = make_input(BatchForwardType::DECODE, /*batch_id=*/502);
  input.token_ids_host = torch::tensor({1}, cpu_tensor_options(torch::kInt));
  input.positions_host = torch::tensor({5}, cpu_tensor_options(torch::kInt));
  input.input_params.embedding.predecessor_rows =
      torch::tensor({-1}, cpu_tensor_options(torch::kLong));
  backend.prepare(input, slot, decode_plan);

  EXPECT_DEATH(backend.launch(decode_plan.invocations.back(),
                              slot,
                              /*predecessor_slot=*/nullptr),
               "base KV lengths.*share a dtype");
}

TEST(MtpPreparedTaskBackendBaseTest,
     RejectsContinuationEmbeddingWithIncompatibleDtype) {
  FixedBufferMtpPreparedTaskBackend backend;
  backend.continuation(0).tail_embeddings =
      torch::zeros({3, 2}, cpu_tensor_options(torch::kDouble));

  EXPECT_DEATH(backend.bind_continuation_for_test(/*slot_id=*/0),
               "continuation embeddings.*share a dtype");
}

}  // namespace
}  // namespace xllm
