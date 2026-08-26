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

#include <gtest/gtest.h>
#include <torch/torch.h>

#include "common/mspti_helper.h"
#include "framework/model/model_input_params.h"
#include "framework/quant_args.h"
#include "layers/common/dsa_metadata_builder.h"
#include "layers/npu_torch/deepseek_sparse_attention.h"
#include "layers/npu_torch/deepseek_v4_indexer.h"
#include "platform/platform.h"
#include "torch_npu/csrc/core/npu/NPUCachingAllocator.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"

namespace xllm {
namespace layer {
namespace {

constexpr int32_t kDeepseekV4ManagerCount = 3;
constexpr char kPreparedDsaPatchAllManagers[] =
    "xllm.PreparedDSA.Patch.AllManagers";
constexpr char kPreparedDsaPatchGeometryOnly[] =
    "xllm.PreparedDSA.Patch.GeometryOnly";
constexpr char kPreparedDsaPatchC4Only[] = "xllm.PreparedDSA.Patch.C4Only";
constexpr char kPreparedDsaPatchC128Only[] = "xllm.PreparedDSA.Patch.C128Only";
constexpr char kPreparedDsaPatchSwaOnly[] = "xllm.PreparedDSA.Patch.SwaOnly";
constexpr char kPreparedDsaCacheScatter[] =
    "xllm.PreparedDSA.Scatter.CacheRows";

DSADeviceGeometryWorkspacePtr make_dsa_device_geometry_workspace(
    int64_t token_capacity,
    int64_t batch_capacity,
    int64_t swa_column_capacity,
    const torch::TensorOptions& position_options,
    const torch::TensorOptions& manager_options) {
  CHECK_GT(token_capacity, 0);
  CHECK_GT(batch_capacity, 0);
  CHECK_GT(swa_column_capacity, 0);
  const torch::TensorOptions index_options =
      manager_options.dtype(torch::kLong);
  const torch::TensorOptions mask_options = manager_options.dtype(torch::kBool);
  const int64_t compact_capacity = token_capacity * 2;
  const int64_t swa_matrix_capacity = batch_capacity * swa_column_capacity;

  auto workspace = std::make_shared<DSADeviceGeometryWorkspace>();
  workspace->actual_seq_lengths_query =
      torch::empty({batch_capacity + 1}, manager_options);
  workspace->kv_cu_seq_lens =
      torch::empty({batch_capacity + 1}, manager_options);
  workspace->max_seqlen_q = torch::empty({1}, manager_options);
  workspace->max_seqlen_kv = torch::empty({1}, manager_options);
  workspace->start_pos = torch::empty({batch_capacity}, manager_options);
  workspace->c4_compact_positions =
      torch::empty({compact_capacity}, position_options);
  workspace->c128_compact_positions =
      torch::empty({compact_capacity}, position_options);
  workspace->c4_compact_slots =
      torch::empty({compact_capacity}, manager_options);
  workspace->c128_compact_slots =
      torch::empty({compact_capacity}, manager_options);
  workspace->swa_block_table =
      torch::empty({swa_matrix_capacity}, manager_options);
  workspace->token_indices = torch::empty({token_capacity}, index_options);
  workspace->position_values = torch::empty({token_capacity}, position_options);
  workspace->position_remainders =
      torch::empty({token_capacity}, position_options);
  workspace->boundary_mask = torch::empty({token_capacity}, mask_options);
  workspace->boundary_ranks = torch::empty({token_capacity}, index_options);
  workspace->sentinel_indices = torch::empty({token_capacity}, index_options);
  workspace->destination_indices =
      torch::empty({token_capacity}, index_options);
  workspace->token_offsets = torch::empty({token_capacity}, manager_options);
  workspace->token_candidates = torch::empty({token_capacity}, manager_options);
  workspace->mapping_valid = torch::empty({token_capacity}, mask_options);
  workspace->mapping_valid_aux = torch::empty({token_capacity}, mask_options);
  workspace->swa_logical_column_indices =
      torch::empty({swa_column_capacity}, index_options);
  workspace->swa_physical_columns =
      torch::empty({swa_matrix_capacity}, index_options);
  workspace->swa_gathered =
      torch::empty({swa_matrix_capacity}, manager_options);
  workspace->swa_valid = torch::empty({swa_matrix_capacity}, mask_options);
  workspace->swa_valid_aux = torch::empty({swa_matrix_capacity}, mask_options);
  workspace->manager_row_indices =
      torch::empty({batch_capacity}, index_options);
  workspace->manager_expanded_block_tables.reserve(kDeepseekV4ManagerCount);
  for (int32_t manager_id = 0; manager_id < kDeepseekV4ManagerCount;
       ++manager_id) {
    workspace->manager_expanded_block_tables.emplace_back(
        torch::empty({swa_matrix_capacity}, manager_options));
  }
  return workspace;
}

}  // namespace

class DeepseekV4IndexerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    options_ = torch::TensorOptions()
                   .dtype(torch::kFloat32)
                   .device(torch::kCPU)
                   .requires_grad(false);
  }

  torch::TensorOptions options_;
};

TEST_F(DeepseekV4IndexerTest, ConstructorAndMetadataWorks) {
  const int64_t dim = 128;
  const int64_t index_n_heads = 8;
  const int64_t index_head_dim = 16;
  const int64_t rope_head_dim = 8;
  const int64_t index_topk = 32;
  const int64_t q_lora_rank = 32;
  const int64_t compress_ratio = 4;

  QuantArgs quant_args;
  auto indexer = DeepseekV4Indexer(DeepseekV4IndexerImpl(dim,
                                                         index_n_heads,
                                                         index_head_dim,
                                                         rope_head_dim,
                                                         index_topk,
                                                         q_lora_rank,
                                                         compress_ratio,
                                                         /*norm_eps=*/1e-6,
                                                         quant_args,
                                                         options_));

  EXPECT_EQ(indexer->dim(), dim);
  EXPECT_EQ(indexer->n_heads(), index_n_heads);
  EXPECT_EQ(indexer->head_dim(), index_head_dim);
  EXPECT_EQ(indexer->rope_head_dim(), rope_head_dim);
  EXPECT_EQ(indexer->index_topk(), index_topk);
  EXPECT_EQ(indexer->q_lora_rank(), q_lora_rank);
  EXPECT_EQ(indexer->compress_ratio(), compress_ratio);

  EXPECT_TRUE(indexer->wq_b());
  EXPECT_TRUE(indexer->weights_proj());
}

TEST_F(DeepseekV4IndexerTest, DsaTokenSlotsTrackCurrentDecodeStep) {
  ModelInputParams params;
  params.meta.batch_forward_type = BatchForwardType::DECODE;
  params.meta.num_sequences = 2;
  params.attention.host.kv_seq_lens = {5, 8};
  params.attention.host.q_seq_lens = {1, 1};
  params.attention.device.new_cache_slots =
      torch::tensor({10, 20}, torch::kInt32);
  params.multi_block_tables = {
      torch::tensor({{0}, {1}}, torch::kInt32),
      torch::tensor({{0}, {1}}, torch::kInt32),
  };

  const auto positions = torch::tensor({4, 7}, torch::kInt64);
  const std::vector<DSAGroupInfo> group_infos = {
      {DSACacheType::SLIDING_WINDOW, 1, 128},
      {DSACacheType::TOKEN, 4, 128},
  };
  const std::vector<std::vector<DSACacheInfo>> caches_info = {{
      {1, DSACacheType::TOKEN, 4, 128},
      {0, DSACacheType::SLIDING_WINDOW, 1, 128},
  }};

  auto metadata = DSAMetadataBuilder::build(
      params, positions, torch::Tensor(), caches_info, group_infos);

  ASSERT_TRUE(metadata.dsa_metadata != nullptr);
  const auto& dsa = *metadata.dsa_metadata;
  ASSERT_EQ(dsa.slot_mappings.size(), 1);
  ASSERT_EQ(dsa.slot_mappings[0].size(), 2);

  const auto token_slots = dsa.slot_mappings[0][0];
  const auto expected_slots = torch::tensor({129}, torch::kInt32);
  EXPECT_TRUE(torch::equal(token_slots, expected_slots))
      << "token slots should include only current-step committed compressed "
         "slots";
  EXPECT_TRUE(
      dsa.slot_mappings[0][1].is_same(params.attention.device.new_cache_slots))
      << "SWA should retain fixed Device cache-slot storage";
}

TEST_F(DeepseekV4IndexerTest, DsaSwaBlockTableUsesLogicalColumnsWithoutWrap) {
  ModelInputParams params;
  params.meta.batch_forward_type = BatchForwardType::DECODE;
  params.meta.num_sequences = 1;
  params.attention.host.kv_seq_lens = {1537};
  params.attention.host.q_seq_lens = {1};
  params.meta.q_max_seq_len = 1;
  params.meta.kv_max_seq_len = 1537;
  params.attention.device.new_cache_slots =
      torch::tensor({10 * 128}, torch::kInt32);
  params.multi_block_tables = {
      torch::tensor({{10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21}},
                    torch::kInt32),
  };

  const auto positions = torch::tensor({1536}, torch::kInt64);
  const std::vector<DSAGroupInfo> group_infos = {
      {DSACacheType::SLIDING_WINDOW, 1, 128},
  };
  const std::vector<std::vector<DSACacheInfo>> caches_info = {{
      {0, DSACacheType::SLIDING_WINDOW, 1, 128},
  }};

  auto metadata = DSAMetadataBuilder::build(
      params, positions, torch::Tensor(), caches_info, group_infos);

  ASSERT_TRUE(metadata.dsa_metadata != nullptr);
  const auto& dsa = *metadata.dsa_metadata;
  ASSERT_EQ(dsa.block_tables.size(), 1);
  ASSERT_EQ(dsa.block_tables[0].size(), 1);
  ASSERT_EQ(dsa.slot_mappings.size(), 1);
  ASSERT_EQ(dsa.slot_mappings[0].size(), 1);

  const auto expected_bt = torch::tensor(
      {{-1, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 10}}, torch::kInt32);
  EXPECT_TRUE(torch::equal(dsa.block_tables[0][0].cpu(), expected_bt));

  const auto expected_slot = torch::tensor({10 * 128}, torch::kInt32);
  EXPECT_TRUE(torch::equal(dsa.slot_mappings[0][0].cpu(), expected_slot));
}

TEST_F(DeepseekV4IndexerTest,
       PreparedDsaGeometryUsesDeviceLengthsAcrossCompressionBoundary) {
  ModelInputParams params;
  params.meta.batch_forward_type = BatchForwardType::CHUNKED_PREFILL;
  params.meta.num_sequences = 1;
  params.meta.q_max_seq_len = 5;
  params.meta.kv_max_seq_len = 8;
  params.attention.host.kv_seq_lens = {8};
  params.attention.host.q_seq_lens = {5};
  params.attention.device.kv_seq_lens = torch::tensor({13}, torch::kInt32);
  params.attention.device.q_seq_lens = torch::tensor({5}, torch::kInt32);
  params.attention.device.new_cache_slots =
      torch::tensor({40, 41, 42, 43, 44}, torch::kInt32);
  params.multi_block_tables = {
      torch::tensor({{10, 11}}, torch::kInt32),
      torch::tensor({{20}}, torch::kInt32),
  };
  params.device_multi_block_tables = params.multi_block_tables;
  params.enable_graph = true;
  params.dsa_device_geometry_authoritative = true;
  params.dsa_device_geometry_kv_headroom = 5;
  params.dsa_device_geometry_workspace = make_dsa_device_geometry_workspace(
      /*token_capacity=*/5,
      /*batch_capacity=*/1,
      /*swa_column_capacity=*/4,
      torch::TensorOptions().dtype(torch::kInt64),
      torch::TensorOptions().dtype(torch::kInt32));

  const torch::Tensor positions =
      torch::tensor({8, 9, 10, 11, 12}, torch::kInt64);
  const std::vector<DSAGroupInfo> group_infos = {
      {DSACacheType::SLIDING_WINDOW, 1, 4},
      {DSACacheType::TOKEN, 4, 128},
  };
  const std::vector<std::vector<DSACacheInfo>> caches_info = {{
      {1, DSACacheType::TOKEN, 4, 128},
      {0, DSACacheType::SLIDING_WINDOW, 1, 4},
  }};

  AttentionMetadata metadata = DSAMetadataBuilder::build(
      params, positions, torch::Tensor(), caches_info, group_infos);
  ASSERT_TRUE(metadata.dsa_metadata != nullptr);
  EXPECT_TRUE(metadata.dsa_metadata->is_acl_graph);
  EXPECT_FALSE(metadata.dsa_metadata->actual_seq_lengths_query.defined());
  EXPECT_FALSE(metadata.dsa_metadata->kv_cu_seq_lens.defined());
  EXPECT_FALSE(metadata.dsa_metadata->c4_pad_positions.defined());
  EXPECT_FALSE(metadata.dsa_metadata->c128_pad_positions.defined());
  EXPECT_TRUE(metadata.dsa_metadata->block_tables.empty());
  EXPECT_TRUE(metadata.dsa_metadata->slot_mappings.empty());
  DSAMetadataBuilder::patch_device_geometry(
      params, group_infos, *metadata.dsa_metadata);
  const DSAMetadata& dsa = *metadata.dsa_metadata;
  EXPECT_TRUE(dsa.device_geometry_authoritative);

  EXPECT_TRUE(torch::equal(dsa.actual_seq_lengths_kv,
                           torch::tensor({13}, torch::kInt32)));
  EXPECT_TRUE(torch::equal(dsa.actual_seq_lengths_query,
                           torch::tensor({0, 5}, torch::kInt32)));
  EXPECT_TRUE(
      torch::equal(dsa.kv_cu_seq_lens, torch::tensor({0, 13}, torch::kInt32)));
  EXPECT_TRUE(torch::equal(dsa.start_pos, torch::tensor({8}, torch::kInt32)));
  EXPECT_EQ(dsa.actual_seq_lengths_query.data_ptr(),
            params.dsa_device_geometry_workspace->actual_seq_lengths_query
                .data_ptr());
  EXPECT_EQ(dsa.kv_cu_seq_lens.data_ptr(),
            params.dsa_device_geometry_workspace->kv_cu_seq_lens.data_ptr());
  EXPECT_EQ(dsa.max_seqlen_q.data_ptr(),
            params.dsa_device_geometry_workspace->max_seqlen_q.data_ptr());
  EXPECT_EQ(dsa.max_seqlen_kv.data_ptr(),
            params.dsa_device_geometry_workspace->max_seqlen_kv.data_ptr());
  EXPECT_EQ(dsa.start_pos.data_ptr(),
            params.dsa_device_geometry_workspace->start_pos.data_ptr());
  EXPECT_EQ(
      dsa.c4_pad_positions.data_ptr(),
      params.dsa_device_geometry_workspace->c4_compact_positions.data_ptr());
  EXPECT_EQ(
      dsa.c128_pad_positions.data_ptr(),
      params.dsa_device_geometry_workspace->c128_compact_positions.data_ptr());
  EXPECT_EQ(dsa.max_query_len, 5);
  EXPECT_EQ(dsa.max_seq_len, 13);
  EXPECT_TRUE(
      torch::equal(dsa.c4_pad_positions, torch::tensor({8, 0}, torch::kInt64)));
  EXPECT_TRUE(
      torch::equal(dsa.c128_pad_positions, torch::tensor({0}, torch::kInt64)));
  EXPECT_TRUE(torch::equal(dsa.block_tables[0][0],
                           torch::tensor({{20}}, torch::kInt32)));
  EXPECT_TRUE(torch::equal(dsa.slot_mappings[0][0],
                           torch::tensor({2562, -1}, torch::kInt32)));
  EXPECT_EQ(dsa.slot_mappings[0][0].data_ptr(),
            params.dsa_device_geometry_workspace->c4_compact_slots.data_ptr());
  EXPECT_TRUE(torch::equal(dsa.block_tables[0][1],
                           torch::tensor({{-1, -1, 10, 11}}, torch::kInt32)));
  EXPECT_EQ(dsa.block_tables[0][1].data_ptr(),
            params.dsa_device_geometry_workspace->swa_block_table.data_ptr());
  EXPECT_TRUE(
      dsa.slot_mappings[0][1].is_same(params.attention.device.new_cache_slots));
}

TEST_F(DeepseekV4IndexerTest,
       PreparedDsaGeometryHandlesPaddedTokenRowsAndExpandsSwaRows) {
  const torch::Device device(Platform::type_torch(), 0);
  ModelInputParams params;
  params.meta.batch_forward_type = BatchForwardType::DECODE;
  params.meta.num_sequences = 8;
  params.meta.q_max_seq_len = 1;
  params.meta.kv_max_seq_len = 254;
  params.attention.host.kv_seq_lens = {123, 124, 125, 126, 251, 252, 253, 254};
  params.attention.host.q_seq_lens = {1, 1, 1, 1, 1, 1, 1, 1};
  params.attention.device.kv_seq_lens =
      torch::tensor({127, 128, 129, 130, 255, 256, 257, 258}, torch::kInt32)
          .to(device);
  params.attention.device.q_seq_lens =
      torch::ones({8}, torch::kInt32).to(device);
  params.attention.device.new_cache_slots =
      (torch::arange(8, torch::kInt32) + 40).to(device);

  params.multi_block_tables = {
      torch::tensor({{10, 11},
                     {10, 11},
                     {10, 11},
                     {10, 11},
                     {20, 21},
                     {20, 21},
                     {20, 21},
                     {20, 21}},
                    torch::kInt32),
      torch::tensor({{100}, {100}, {100}, {100}, {200}, {200}, {200}, {200}},
                    torch::kInt32),
      torch::tensor({{300}, {300}, {300}, {300}, {400}, {400}, {400}, {400}},
                    torch::kInt32),
  };
  const torch::Tensor c4_manager_storage = torch::tensor({{100, 1000, 1001},
                                                          {100, 1010, 1011},
                                                          {100, 1020, 1021},
                                                          {100, 1030, 1031},
                                                          {200, 1040, 1041},
                                                          {200, 1050, 1051},
                                                          {200, 1060, 1061},
                                                          {200, 1070, 1071}},
                                                         torch::kInt32)
                                               .to(device);
  params.device_multi_block_tables = {
      torch::tensor({{10, 11}, {20, 21}}, torch::kInt32).to(device),
      c4_manager_storage.narrow(/*dim=*/1, /*start=*/0, /*length=*/1),
      torch::tensor({{300}, {400}}, torch::kInt32).to(device),
  };
  ASSERT_EQ(params.device_multi_block_tables[1].stride(/*dim=*/0), 3);
  params.dsa_device_geometry_authoritative = true;
  params.dsa_device_geometry_kv_headroom = 4;
  params.dsa_device_geometry_workspace = make_dsa_device_geometry_workspace(
      /*token_capacity=*/8,
      /*batch_capacity=*/8,
      /*swa_column_capacity=*/3,
      torch::TensorOptions().dtype(torch::kInt64).device(device),
      torch::TensorOptions().dtype(torch::kInt32).device(device));

  const torch::Tensor positions =
      torch::tensor({126, 127, 128, 129, 254, 255, 256, 257}, torch::kInt64)
          .to(device);
  const std::vector<DSAGroupInfo> group_infos = {
      {DSACacheType::SLIDING_WINDOW, 1, 128},
      {DSACacheType::TOKEN, 4, 128},
      {DSACacheType::TOKEN, 128, 128},
  };
  const std::vector<std::vector<DSACacheInfo>> caches_info = {{
      {1, DSACacheType::TOKEN, 4, 128},
      {2, DSACacheType::TOKEN, 128, 128},
      {0, DSACacheType::SLIDING_WINDOW, 1, 128},
  }};

  AttentionMetadata metadata = DSAMetadataBuilder::build(
      params, positions, torch::Tensor(), caches_info, group_infos);
  ASSERT_TRUE(metadata.dsa_metadata != nullptr);
  DSAMetadataBuilder::patch_device_geometry(
      params, group_infos, *metadata.dsa_metadata);
  const DSAMetadata& dsa = *metadata.dsa_metadata;

  EXPECT_TRUE(
      torch::equal(dsa.c4_pad_positions.cpu(),
                   torch::tensor({124, 252, 0, 0, 0, 0, 0, 0}, torch::kInt64)));
  EXPECT_TRUE(
      torch::equal(dsa.c128_pad_positions.cpu(),
                   torch::tensor({0, 128, 0, 0, 0, 0, 0, 0}, torch::kInt64)));
  EXPECT_EQ(
      dsa.c4_pad_positions.data_ptr(),
      params.dsa_device_geometry_workspace->c4_compact_positions.data_ptr());
  EXPECT_EQ(
      dsa.c128_pad_positions.data_ptr(),
      params.dsa_device_geometry_workspace->c128_compact_positions.data_ptr());
  EXPECT_TRUE(torch::equal(
      dsa.slot_mappings[0][0].cpu(),
      torch::tensor({12831, 25663, -1, -1, -1, -1, -1, -1}, torch::kInt32)));
  EXPECT_TRUE(torch::equal(
      dsa.slot_mappings[0][1].cpu(),
      torch::tensor({38400, 51201, -1, -1, -1, -1, -1, -1}, torch::kInt32)));
  EXPECT_EQ(dsa.slot_mappings[0][0].data_ptr(),
            params.dsa_device_geometry_workspace->c4_compact_slots.data_ptr());
  EXPECT_EQ(
      dsa.slot_mappings[0][1].data_ptr(),
      params.dsa_device_geometry_workspace->c128_compact_slots.data_ptr());
  EXPECT_TRUE(torch::equal(
      dsa.block_tables[0][1].cpu(),
      torch::tensor({{300}, {300}, {300}, {300}, {400}, {400}, {400}, {400}},
                    torch::kInt32)));
  EXPECT_EQ(
      dsa.block_tables[0][1].data_ptr(),
      params.dsa_device_geometry_workspace->manager_expanded_block_tables[2]
          .data_ptr());
  EXPECT_TRUE(torch::equal(dsa.block_tables[0][2].cpu(),
                           torch::tensor({{10, -1, -1},
                                          {10, -1, -1},
                                          {10, 11, -1},
                                          {10, 11, -1},
                                          {20, 21, -1},
                                          {20, 21, -1},
                                          {-1, 21, 20},
                                          {-1, 21, 20}},
                                         torch::kInt32)));
  EXPECT_EQ(dsa.block_tables[0][2].data_ptr(),
            params.dsa_device_geometry_workspace->swa_block_table.data_ptr());
  EXPECT_TRUE(
      dsa.slot_mappings[0][2].is_same(params.attention.device.new_cache_slots));

  c10_npu::getCurrentNPUStream().synchronize();
  constexpr size_t kAggregateStat =
      static_cast<size_t>(c10_npu::NPUCachingAllocator::StatType::AGGREGATE);
  const c10_npu::NPUCachingAllocator::DeviceStats stats_before =
      c10_npu::NPUCachingAllocator::getDeviceStats(device.index());
  {
    MstxHostRange patch_range(kPreparedDsaPatchAllManagers);
    DSAMetadataBuilder::patch_device_geometry(
        params, group_infos, *metadata.dsa_metadata);
    c10_npu::getCurrentNPUStream().synchronize();
  }
  const c10_npu::NPUCachingAllocator::DeviceStats stats_after =
      c10_npu::NPUCachingAllocator::getDeviceStats(device.index());
  const int64_t allocation_request_delta =
      stats_after.allocation[kAggregateStat].allocated -
      stats_before.allocation[kAggregateStat].allocated;
  LOG(INFO) << "Warmed Prepared DSA patch caching-allocator request delta: "
            << allocation_request_delta;
  EXPECT_EQ(stats_after.allocation[kAggregateStat].current,
            stats_before.allocation[kAggregateStat].current)
      << "warmed Prepared DSA patch should not retain new allocation blocks";
  EXPECT_EQ(stats_after.allocated_bytes[kAggregateStat].current,
            stats_before.allocated_bytes[kAggregateStat].current)
      << "warmed Prepared DSA patch should retain stable allocated bytes";

  auto measure_patch_allocation_requests =
      [&](const char* marker_name,
          const ModelInputParams& audit_params,
          const std::vector<DSAGroupInfo>& audit_group_infos,
          DSAMetadata& audit_dsa) -> int64_t {
    c10_npu::getCurrentNPUStream().synchronize();
    const c10_npu::NPUCachingAllocator::DeviceStats audit_before =
        c10_npu::NPUCachingAllocator::getDeviceStats(device.index());
    {
      MstxHostRange patch_range(marker_name);
      DSAMetadataBuilder::patch_device_geometry(
          audit_params, audit_group_infos, audit_dsa);
      c10_npu::getCurrentNPUStream().synchronize();
    }
    const c10_npu::NPUCachingAllocator::DeviceStats audit_after =
        c10_npu::NPUCachingAllocator::getDeviceStats(device.index());
    EXPECT_EQ(audit_after.allocation[kAggregateStat].current,
              audit_before.allocation[kAggregateStat].current);
    EXPECT_EQ(audit_after.allocated_bytes[kAggregateStat].current,
              audit_before.allocated_bytes[kAggregateStat].current);
    return audit_after.allocation[kAggregateStat].allocated -
           audit_before.allocation[kAggregateStat].allocated;
  };

  ModelInputParams geometry_only_params = params;
  geometry_only_params.device_multi_block_tables.clear();
  const std::vector<DSAGroupInfo> geometry_only_group_infos;
  const std::vector<std::vector<DSACacheInfo>> geometry_only_caches_info;
  DSAMetadata geometry_only_dsa;
  geometry_only_dsa.input_positions = positions;
  geometry_only_dsa.caches_info = &geometry_only_caches_info;
  const int64_t geometry_only_allocation_requests =
      measure_patch_allocation_requests(kPreparedDsaPatchGeometryOnly,
                                        geometry_only_params,
                                        geometry_only_group_infos,
                                        geometry_only_dsa);

  ModelInputParams c4_only_params = params;
  c4_only_params.device_multi_block_tables = {
      params.device_multi_block_tables[1]};
  const std::vector<DSAGroupInfo> c4_only_group_infos = {
      {DSACacheType::TOKEN, 4, 128}};
  const std::vector<std::vector<DSACacheInfo>> c4_only_caches_info = {{
      {0, DSACacheType::TOKEN, 4, 128},
  }};
  DSAMetadata c4_only_dsa;
  c4_only_dsa.input_positions = positions;
  c4_only_dsa.caches_info = &c4_only_caches_info;
  const int64_t c4_only_allocation_requests =
      measure_patch_allocation_requests(kPreparedDsaPatchC4Only,
                                        c4_only_params,
                                        c4_only_group_infos,
                                        c4_only_dsa);

  ModelInputParams c128_only_params = params;
  c128_only_params.device_multi_block_tables = {
      params.device_multi_block_tables[2]};
  const std::vector<DSAGroupInfo> c128_only_group_infos = {
      {DSACacheType::TOKEN, 128, 128}};
  const std::vector<std::vector<DSACacheInfo>> c128_only_caches_info = {{
      {0, DSACacheType::TOKEN, 128, 128},
  }};
  DSAMetadata c128_only_dsa;
  c128_only_dsa.input_positions = positions;
  c128_only_dsa.caches_info = &c128_only_caches_info;
  const int64_t c128_only_allocation_requests =
      measure_patch_allocation_requests(kPreparedDsaPatchC128Only,
                                        c128_only_params,
                                        c128_only_group_infos,
                                        c128_only_dsa);

  ModelInputParams swa_only_params = params;
  swa_only_params.device_multi_block_tables = {
      params.device_multi_block_tables[0]};
  const std::vector<DSAGroupInfo> swa_only_group_infos = {
      {DSACacheType::SLIDING_WINDOW, 1, 128}};
  const std::vector<std::vector<DSACacheInfo>> swa_only_caches_info = {{
      {0, DSACacheType::SLIDING_WINDOW, 1, 128},
  }};
  DSAMetadata swa_only_dsa;
  swa_only_dsa.input_positions = positions;
  swa_only_dsa.caches_info = &swa_only_caches_info;
  const int64_t swa_only_allocation_requests =
      measure_patch_allocation_requests(kPreparedDsaPatchSwaOnly,
                                        swa_only_params,
                                        swa_only_group_infos,
                                        swa_only_dsa);

  LOG(INFO) << "Warmed Prepared DSA patch allocation audit: geometry_only="
            << geometry_only_allocation_requests
            << ", c4_only=" << c4_only_allocation_requests
            << ", c128_only=" << c128_only_allocation_requests
            << ", swa_only=" << swa_only_allocation_requests
            << ", all_managers=" << allocation_request_delta;
}

TEST_F(DeepseekV4IndexerTest,
       PreparedDsaCacheScatterIgnoresNegativePaddedRows) {
  const torch::Device device(Platform::type_torch(), 0);
  torch::Tensor cache = torch::full(
      {4, 2}, -9, torch::TensorOptions().dtype(torch::kInt8).device(device));
  torch::Tensor scale_cache =
      torch::full({4, 1},
                  -3.0,
                  torch::TensorOptions().dtype(torch::kFloat32).device(device));
  const torch::Tensor slots =
      torch::tensor({0, -1, 2, -1}, torch::kLong).to(device);
  const torch::Tensor values =
      torch::tensor({{1, 2}, {3, 4}, {5, 6}, {7, 8}}, torch::kInt8).to(device);
  const torch::Tensor scales =
      torch::tensor({{0.1}, {0.2}, {0.3}, {0.4}}, torch::kFloat32).to(device);

  deepseek_v4_indexer_detail::scatter_prepared_dsa_cache_rows(
      cache, &scale_cache, slots, values, scales);
  c10_npu::getCurrentNPUStream().synchronize();

  constexpr size_t kAggregateStat =
      static_cast<size_t>(c10_npu::NPUCachingAllocator::StatType::AGGREGATE);
  const c10_npu::NPUCachingAllocator::DeviceStats stats_before =
      c10_npu::NPUCachingAllocator::getDeviceStats(device.index());
  {
    MstxHostRange scatter_range(kPreparedDsaCacheScatter);
    deepseek_v4_indexer_detail::scatter_prepared_dsa_cache_rows(
        cache, &scale_cache, slots, values, scales);
    c10_npu::getCurrentNPUStream().synchronize();
  }
  const c10_npu::NPUCachingAllocator::DeviceStats stats_after =
      c10_npu::NPUCachingAllocator::getDeviceStats(device.index());
  const int64_t allocation_request_delta =
      stats_after.allocation[kAggregateStat].allocated -
      stats_before.allocation[kAggregateStat].allocated;
  LOG(INFO) << "Warmed Prepared DSA padding-safe scatter caching-allocator "
               "request delta: "
            << allocation_request_delta;
  EXPECT_EQ(stats_after.allocation[kAggregateStat].current,
            stats_before.allocation[kAggregateStat].current);
  EXPECT_EQ(stats_after.allocated_bytes[kAggregateStat].current,
            stats_before.allocated_bytes[kAggregateStat].current);

  const torch::Tensor expected_cache =
      torch::tensor({{1, 2}, {-9, -9}, {5, 6}, {-9, -9}}, torch::kInt8);
  const torch::Tensor expected_scale_cache =
      torch::tensor({{0.1}, {-3.0}, {0.3}, {-3.0}}, torch::kFloat32);
  EXPECT_TRUE(torch::equal(cache.cpu(), expected_cache));
  EXPECT_TRUE(torch::equal(scale_cache.cpu(), expected_scale_cache));
}

TEST_F(DeepseekV4IndexerTest, DsaSwaUsesExplicitBlockParallelSlots) {
  ModelInputParams params;
  params.meta.batch_forward_type = BatchForwardType::CHUNKED_PREFILL;
  params.meta.num_sequences = 3;
  params.meta.q_max_seq_len = 1;
  params.meta.kv_max_seq_len = 8;
  params.attention.host.kv_seq_lens = {8, 8, 8};
  params.attention.host.q_seq_lens = {1, 1, 1};
  params.attention.host.new_cache_slots = {5, 6, 7};
  params.multi_block_tables = {
      torch::tensor({{0, 1, 2}, {0, 1, 2}, {0, 1, 2}}, torch::kInt32)};

  const torch::Tensor positions = torch::tensor({5, 6, 7}, torch::kInt64);
  const std::vector<DSAGroupInfo> group_infos = {
      {DSACacheType::SLIDING_WINDOW, 1, 4}};
  const std::vector<std::vector<DSACacheInfo>> caches_info = {{
      {0, DSACacheType::SLIDING_WINDOW, 1, 4},
  }};

  const AttentionMetadata metadata = DSAMetadataBuilder::build(
      params, positions, torch::Tensor(), caches_info, group_infos);

  ASSERT_TRUE(metadata.dsa_metadata != nullptr);
  ASSERT_EQ(metadata.dsa_metadata->slot_mappings.size(), 1);
  ASSERT_EQ(metadata.dsa_metadata->slot_mappings[0].size(), 1);
  EXPECT_TRUE(torch::equal(metadata.dsa_metadata->slot_mappings[0][0],
                           torch::tensor({5, 6, 7}, torch::kInt32)));
}

TEST_F(DeepseekV4IndexerTest, DSparkSparseTilingUsesSupportedWindow) {
  ModelInputParams params;
  params.meta.batch_forward_type = BatchForwardType::CHUNKED_PREFILL;
  params.meta.q_max_seq_len = 1;

  EXPECT_TRUE(params.meta.batch_forward_type.no_decode());
  EXPECT_EQ(deepseek_v4_ori_window_left(/*window_size=*/128,
                                        /*dspark_block_size=*/5,
                                        /*use_native_dspark_sas=*/false),
            127);
  EXPECT_EQ(deepseek_v4_ori_window_left(/*window_size=*/128,
                                        /*dspark_block_size=*/0,
                                        /*use_native_dspark_sas=*/true),
            127);
  EXPECT_EQ(deepseek_v4_ori_window_left(/*window_size=*/128,
                                        /*dspark_block_size=*/5,
                                        /*use_native_dspark_sas=*/true),
            132);

  params.meta.batch_forward_type = BatchForwardType::DECODE;
  EXPECT_FALSE(params.meta.batch_forward_type.no_decode());
}

TEST_F(DeepseekV4IndexerTest, DSparkNativeSwaIndicesAreSharedByQueryRows) {
  const torch::Tensor block_table =
      torch::tensor({{10, 11, 12}}, torch::kInt32);
  const torch::Tensor query_cu_seq_lens = torch::tensor({0, 3}, torch::kInt32);
  const torch::Tensor seq_lens = torch::tensor({6}, torch::kInt32);

  const torch::Tensor indices =
      build_dspark_swa_indices(block_table,
                               query_cu_seq_lens,
                               seq_lens,
                               /*window_size=*/4,
                               /*dspark_block_size=*/3,
                               /*cache_block_size=*/4);

  ASSERT_EQ(indices.dim(), 3);
  ASSERT_EQ(indices.size(0), 3);
  ASSERT_EQ(indices.size(1), 1);
  ASSERT_EQ(indices.size(2), 128);
  const torch::Tensor expected_prefix =
      torch::tensor({40, 41, 42, 43, 44, 45, -1}, torch::kInt32);
  for (int64_t row = 0; row < indices.size(0); ++row) {
    EXPECT_TRUE(torch::equal(indices[row][0].slice(0, 0, 7), expected_prefix));
  }
}

TEST_F(DeepseekV4IndexerTest, DSparkNativeSwaIndicesWrapAroundRingBuffer) {
  // Two-block ring buffer with kv_len=10 forces the SWA window to span both
  // ring entries and to wrap. Visible positions 5..9 map through
  // block_column = pos/2 % ring_size(2) -> {0,1,1,0,0}, wrapping back to
  // block_column 0 once the ring rotates.
  const torch::Tensor block_table = torch::tensor({{20, 21}}, torch::kInt32);
  const torch::Tensor query_cu_seq_lens = torch::tensor({0, 1}, torch::kInt32);
  const torch::Tensor seq_lens = torch::tensor({10}, torch::kInt32);

  const torch::Tensor indices =
      build_dspark_swa_indices(block_table,
                               query_cu_seq_lens,
                               seq_lens,
                               /*window_size=*/4,
                               /*dspark_block_size=*/3,
                               /*cache_block_size=*/2);

  ASSERT_EQ(indices.dim(), 3);
  ASSERT_EQ(indices.size(0), 1);
  ASSERT_EQ(indices.size(1), 1);
  // start_pos = max((kv - q_len) - window, 0) = (10-1)-4 = 5, so the visible
  // window is positions 5,6,7,8,9. block_column = pos/2 % 2 -> {0,1,1,0,0};
  // slot = block_id*2 + pos%2 -> {20*2+1, 21*2+0, 21*2+1, 20*2+0, 20*2+1} =
  // {41, 42, 43, 40, 41}.
  const torch::Tensor expected_prefix =
      torch::tensor({41, 42, 43, 40, 41}, torch::kInt32);
  EXPECT_TRUE(torch::equal(indices[0][0].slice(0, 0, 5), expected_prefix));
}

TEST_F(DeepseekV4IndexerTest, DsaDummyAttentionUsesPositionDevice) {
  ModelInputParams params;
  params.meta.batch_forward_type = BatchForwardType::DECODE;
  params.meta.num_sequences = 1;
  params.meta.q_max_seq_len = 0;
  params.meta.kv_max_seq_len = 0;
  params.attention.host.kv_seq_lens = {0};
  params.attention.host.q_seq_lens = {0};

  const auto positions = torch::empty({0}, torch::kInt64);
  const std::vector<DSAGroupInfo> group_infos;
  const std::vector<std::vector<DSACacheInfo>> caches_info;

  auto metadata = DSAMetadataBuilder::build(
      params, positions, torch::Tensor(), caches_info, group_infos);

  EXPECT_TRUE(metadata.is_dummy);
  EXPECT_TRUE(metadata.slot_mapping.defined());
  EXPECT_EQ(metadata.slot_mapping.device(), positions.device());
  EXPECT_TRUE(torch::equal(metadata.q_seq_lens.cpu(),
                           torch::tensor({1}, torch::kInt32)));
}

}  // namespace layer
}  // namespace xllm
