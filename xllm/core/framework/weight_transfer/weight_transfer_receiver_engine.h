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

#include <functional>
#include <future>
#include <string>
#include <unordered_map>
#include <vector>

#include "framework/model/causal_lm.h"
#include "framework/model_context.h"
#include "framework/weight_transfer/weight_transfer_alltoall_planner.h"
#include "framework/weight_transfer/weight_transfer_session_manager.h"
#include "framework/weight_transfer/weight_transfer_types.h"

namespace xllm {

class WeightTransferReceiverEngine {
 public:
  WeightTransferReceiverEngine(const ModelContext& context,
                               CausalLM* model,
                               int32_t device_id,
                               const std::string& local_addr,
                               WeightTransferSessionManager* session_manager,
                               const WeightTransferAlltoallPlanner* planner);
  ~WeightTransferReceiverEngine();

  bool pull_model_from_instance(
      const std::string& remote_addr,
      const RankExpertTransferPlanData& rank_expert_transfer_plan);

 private:
  bool init_collective_comm_as_receiver(
      const std::vector<std::string>& source_addrs,
      uint32_t receiver_rank,
      const std::string& session_id);
  bool pull_expert_weights_alltoallv(
      const std::vector<std::string>& source_addrs,
      const std::vector<xllm::proto::AlltoAllRoundDesc>& receiver_rounds,
      const std::vector<xllm::proto::TriggerWeightsSendRequest>&
          source_requests,
      const std::string& session_id,
      bool comm_initialized = false);
  bool prepare_session_rpc_endpoint(const std::string& remote_addr,
                                    const std::string& session_id);
  bool init_session_p2p_comm(const std::string& remote_addr,
                             const std::string& session_id);
  bool prepare_local_tensors_from_meta(
      const std::string& session_id,
      const std::vector<int32_t>& layer_ids,
      const std::vector<std::vector<at::Tensor>*>& local_tensors_ptrs,
      bool enable_layer_storage_allocation,
      std::unordered_map<int32_t, ReceiverLayerStorage>* layer_storages,
      MetaAllocateStats* prepare_stats = nullptr);
  bool fetch_weights_meta_and_allocate_tensors(
      xllm::proto::WeightTransferService_Stub* target_stub,
      const std::vector<int32_t>& layer_ids,
      const std::vector<std::vector<at::Tensor>*>& local_tensors_ptrs,
      bool enable_layer_storage_allocation,
      std::unordered_map<int32_t, ReceiverLayerStorage>* layer_storages,
      const std::string& stage_name,
      MetaAllocateStats* stats);
  std::vector<int32_t> build_pull_layer_ids() const;
  void build_pull_tensor_ptrs(
      std::vector<at::Tensor>* global_tensors,
      std::vector<std::vector<at::Tensor>*>* local_tensors_ptrs);
  bool assign_global_tensors_after_pull(
      const std::vector<at::Tensor>& global_tensors);
  void release_layer_storage_views();
  LayerExpertIdsMap build_non_expert_only_plan(
      const std::vector<int32_t>& layer_ids) const;

  ModelPullPrepareStageStatus run_model_pull_prepare_stage(
      const std::string& remote_addr,
      const std::string& base_session_id,
      const std::vector<int32_t>& layer_ids,
      const std::vector<std::vector<at::Tensor>*>& local_tensors_ptrs,
      bool enable_layer_storage_allocation,
      std::unordered_map<int32_t, ReceiverLayerStorage>* layer_storages,
      bool has_expert_prepare,
      const std::function<ModelPullAsyncStageStatus()>& expert_prepare_fn);
  void log_model_pull_prepare_timing(
      const std::string& remote_addr,
      size_t expert_sources,
      const ModelPullPrepareStageStatus& prepare_status) const;
  ModelPullStageStatus run_prepared_expert_transfer_serial_stage(
      bool has_two_stage_expert,
      bool use_alltoallv_expert_transfer,
      const std::vector<std::string>& alltoall_source_addrs,
      const std::vector<xllm::proto::AlltoAllRoundDesc>&
          alltoall_receiver_rounds,
      const std::vector<xllm::proto::TriggerWeightsSendRequest>&
          alltoall_source_requests,
      const std::string& expert_session_id,
      const std::vector<int32_t>& single_source_layer_ids,
      const std::vector<std::vector<at::Tensor>*>& single_source_tensors_ptrs,
      const LayerExpertIdsMap& single_source_layer_expert_ids,
      double* transfer_ms);

  std::future<TriggerRpcResult> launch_trigger_rpc_stage(
      const std::shared_ptr<CommSessionContext>& session_ctx,
      const std::vector<int32_t>& layer_ids,
      const LayerExpertIdsMap& normalized_layer_expert_ids,
      bool include_non_expert,
      bool transfer_all_experts,
      bool use_layer_storage_transfer,
      const std::string& session_id);
  std::future<ReceiverTransferResult> launch_receiver_exec_stage(
      const std::shared_ptr<CommSessionContext>& session_ctx,
      const std::vector<int32_t>& layer_ids,
      const std::vector<std::vector<at::Tensor>*>& local_tensors_ptrs,
      const std::unordered_map<int32_t, ReceiverLayerStorage>& layer_storages,
      const LayerExpertIdsMap& normalized_layer_expert_ids,
      bool include_non_expert,
      bool transfer_all_experts,
      bool use_layer_storage_transfer);
  bool pull_weight_internal(
      const std::string& session_id,
      const std::vector<int32_t>& layer_ids,
      const std::vector<std::vector<at::Tensor>*>& local_tensors_ptrs,
      const std::unordered_map<int32_t, ReceiverLayerStorage>& layer_storages,
      const LayerExpertIdsMap& layer_expert_ids,
      bool include_non_expert,
      bool transfer_all_experts,
      bool allocate_tensors);

  const ModelContext& context_;
  CausalLM* model_;
  int32_t device_id_;
  const std::string& local_addr_;
  WeightTransferSessionManager* session_manager_;
  const WeightTransferAlltoallPlanner* planner_;
  bool use_layer_storage_transfer_ = false;
  std::unordered_map<int32_t, ReceiverLayerStorage> receiver_layer_storages_;
};

}  // namespace xllm
