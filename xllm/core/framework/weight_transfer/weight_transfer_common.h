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

#include <absl/time/time.h>
#include <torch/torch.h>

#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

#include "framework/model/causal_lm.h"
#include "framework/parallel_state/parallel_args.h"
#include "framework/weight_transfer/weight_transfer_types.h"

namespace xllm {

LayerStorageInfo get_contiguous_layer_storage_info(
    const std::vector<at::Tensor>& tensors,
    layer::BaseLoader* loader);

class SenderSessionLogGuard {
 public:
  SenderSessionLogGuard(const std::string& session_id, const std::string& mode);
  ~SenderSessionLogGuard();

  void mark_success();

 private:
  std::string session_id_;
  std::string mode_;
  bool enabled_ = false;
  bool success_ = false;
};

double elapsed_ms_since(const absl::Time& start_time);

void log_weight_pull_timing(const std::string& remote_addr,
                            const WeightPullTimingStats& timing_stats);

std::string generate_session_id();

bool get_cached_rpc_endpoint(const std::string& remote_addr,
                             std::shared_ptr<CachedRpcEndpoint>* endpoint_out);

std::shared_ptr<const std::unordered_set<int32_t>>
get_cached_expert_indices_set(CausalLM* model);

bool init_weight_transfer_stub(
    const std::string& remote_addr,
    std::unique_ptr<brpc::Channel>* channel,
    std::unique_ptr<xllm::proto::WeightTransferService_Stub>* stub);

bool call_init_comm_with_retry(xllm::proto::WeightTransferService_Stub* stub,
                               const xllm::proto::InitCommRequest& req,
                               const std::string& remote_addr);

std::vector<int32_t> normalize_expert_ids(
    const std::vector<int32_t>& expert_ids);

LayerExpertIdsMap normalize_layer_expert_ids_map(
    const LayerExpertIdsMap& input);

size_t count_expert_ids(const LayerExpertIdsMap& layer_expert_ids);

bool validate_layer_expert_ids_map(const std::vector<int32_t>& layer_ids,
                                   const LayerExpertIdsMap& layer_expert_ids,
                                   const std::string& stage_name);

int32_t get_ep_size(const ParallelArgs& parallel_args);
int32_t get_ep_rank(const ParallelArgs& parallel_args);

bool append_layer_transfer_items(
    const std::vector<at::Tensor>& tensors,
    int32_t layer_id,
    const std::vector<int32_t>& expert_ids,
    const std::unordered_set<int32_t>& expert_indices_set,
    bool include_non_expert,
    bool transfer_all_experts,
    int32_t ep_rank,
    int32_t ep_size,
    decltype(HCCL_SEND) operation,
    uint32_t peer_rank,
    std::vector<HcclSendRecvItem>* items,
    size_t* total_nbytes);

void append_contiguous_storage_transfer_items(
    void* base_ptr,
    uint64_t storage_size,
    decltype(HCCL_SEND) operation,
    uint32_t peer_rank,
    uint64_t chunk_bytes,
    std::vector<HcclSendRecvItem>* items,
    size_t* total_nbytes);

uint64_t sum_tensor_nbytes(const std::vector<at::Tensor>& tensors);

std::string summarize_hccl_transfer_items(
    const std::vector<HcclSendRecvItem>& items);

std::string summarize_hccl_transfer_item_addresses(
    const std::vector<HcclSendRecvItem>& items);

void fill_trigger_weights_send_request(
    const std::vector<int32_t>& layer_ids,
    const LayerExpertIdsMap& layer_expert_ids,
    bool include_non_expert,
    xllm::proto::TriggerWeightsSendRequest* req);

LayerExpertIdsMap parse_layer_expert_ids(
    const xllm::proto::TriggerWeightsSendRequest& request);

uint64_t get_expert_unit_bytes(const at::Tensor& tensor,
                               int32_t layer_id,
                               int32_t tensor_idx);

}  // namespace xllm
