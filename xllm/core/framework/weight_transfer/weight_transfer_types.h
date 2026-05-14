/* Copyright 2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#pragma once

#include <acl/acl.h>
#include <brpc/channel.h>
#include <hccl/hccl.h>
#include <torch/torch.h>

#include <atomic>
#include <cstddef>
#include <future>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "common/types.h"
#include "framework/model/model_args.h"
#include "framework/parallel_state/parallel_args.h"
#include "hccl_weight_transfer.pb.h"

namespace xllm {
namespace layer {
class BaseLoader;
}  // namespace layer

using LayerExpertIdsMap = std::unordered_map<int32_t, std::vector<int32_t>>;

struct LayerStorageInfo {
  bool available = false;
  void* base_ptr = nullptr;
  uint64_t storage_size = 0;
};

struct CommSessionContext {
  std::string session_id;
  xllm::proto::CommMode comm_mode = xllm::proto::COMM_MODE_P2P;
  uint32_t n_ranks = 0;
  uint32_t rank = 0;
  std::atomic<bool> is_comm_initialized = false;
  HcclComm hccl_comm = nullptr;
  aclrtStream stream = nullptr;
  std::unique_ptr<brpc::Channel> channel;
  std::unique_ptr<xllm::proto::WeightTransferService_Stub> stub;
  std::mutex operation_mutex;
};

struct CachedRpcEndpoint {
  std::shared_ptr<brpc::Channel> channel;
  std::shared_ptr<xllm::proto::WeightTransferService_Stub> stub;
};

struct MetaAllocateStats {
  double total_ms = 0.0;
  double rpc_ms = 0.0;
  double tensor_alloc_ms = 0.0;
  double storage_alloc_ms = 0.0;
  double storage_view_init_ms = 0.0;
  bool all_layer_storage_available = false;
};

struct ReceiverLayerStorage {
  bool available = false;
  void* base_ptr = nullptr;
  uint64_t storage_size = 0;
  uint64_t payload_nbytes = 0;
  bool views_initialized = false;
  layer::BaseLoader* loader = nullptr;
};

struct TriggerRpcResult {
  bool ok = false;
  double queue_wait_ms = 0.0;
  double rpc_ms = 0.0;
};

struct ReceiverTransferResult {
  bool ok = false;
  double queue_wait_ms = 0.0;
  double build_items_ms = 0.0;
  double storage_view_init_ms = 0.0;
  double hccl_exec_ms = 0.0;
  double thread_total_ms = 0.0;
  size_t total_nbytes = 0;
  size_t item_count = 0;
  size_t layer_storage_items = 0;
  size_t tensor_items = 0;
  uint64_t storage_padding_nbytes = 0;
};

struct ModelPullStageStatus {
  bool ok = true;
  std::string failed_stage;
};

struct ModelPullAsyncStageStatus {
  bool ok = true;
  std::string failed_stage;
  double elapsed_ms = 0.0;
};

struct ModelPullPrepareStageStatus {
  bool ok = true;
  std::string failed_stage;
  double base_endpoint_ms = 0.0;
  double base_comm_ms = 0.0;
  double expert_comm_ms = 0.0;
  double meta_alloc_ms = 0.0;
  double meta_rpc_ms = 0.0;
  double tensor_alloc_ms = 0.0;
  double storage_alloc_ms = 0.0;
  double storage_view_init_ms = 0.0;
  double prepare_wall_ms = 0.0;
};

struct ModelPullAsyncTransferStatus {
  bool ok = false;
  double transfer_ms = 0.0;
  std::string failed_stage;
};

struct SourceExpertTransferTask {
  std::string source_addr;
  LayerExpertIdsMap layer_expert_ids;
};

struct LocalExpertRun {
  int32_t local_expert_start = 0;
  int32_t local_expert_count = 0;
};

struct SourceRoundRun {
  bool active = false;
  LocalExpertRun run;
};

struct AlltoallRoundPlan {
  int32_t layer_id = -1;
  int32_t tensor_idx = -1;
  int32_t run_slot = -1;
  std::vector<SourceRoundRun> source_runs;
};

struct WeightPullTimingStats {
  double total_ms = 0.0;
  double comm_create_ms = 0.0;
  double weight_transfer_ms = 0.0;
  double other_ms = 0.0;
  bool parallel_mode = false;
  size_t expert_sources = 0;
  bool success = false;
  std::string failed_stage;
};

}  // namespace xllm
