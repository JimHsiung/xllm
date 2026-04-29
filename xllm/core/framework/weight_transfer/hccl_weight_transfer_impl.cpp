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

#include "framework/weight_transfer/hccl_weight_transfer_impl.h"

#include <glog/logging.h>

#include "util/net.h"

namespace xllm {

HcclWeightTransferImpl::HcclWeightTransferImpl(const ModelContext& context,
                                               CausalLM* model,
                                               int32_t device_id,
                                               int32_t listen_port)
    : context_(context),
      model_(model),
      device_id_(device_id),
      listen_port_(listen_port),
      session_manager_(device_id),
      sender_engine_(context_,
                     model_,
                     device_id_,
                     local_addr_,
                     &session_manager_,
                     &layer_registry_),
      receiver_engine_(context_,
                       model_,
                       device_id_,
                       local_addr_,
                       &session_manager_,
                       &alltoall_planner_) {
  aclrtSetDevice(device_id_);
  std::string ip = net::get_local_ip_addr();
  local_addr_ = ip + ":" + std::to_string(listen_port_);
}

HcclWeightTransferImpl::~HcclWeightTransferImpl() {
  if (server_.IsRunning()) {
    server_.Stop(0);
  }
  server_.Join();
  session_manager_.destroy_all_session_contexts();
}

void HcclWeightTransferImpl::register_layer(
    int32_t layer_id,
    const std::vector<at::Tensor>& tensors) {
  sender_engine_.register_layer(layer_id, tensors);
}

void HcclWeightTransferImpl::start_serving() {
  service_ = std::make_unique<WeightTransferServiceImpl>(this);
  if (server_.AddService(service_.get(), brpc::SERVER_DOESNT_OWN_SERVICE) !=
      0) {
    LOG(ERROR) << "Failed to add service to server";
  }
  brpc::ServerOptions options;
  if (server_.Start(listen_port_, &options) != 0) {
    LOG(ERROR) << "Failed to start Brpc rpc server";
  }
  LOG(INFO) << "Weight Transfer Server started on " << local_addr_;
}

void HcclWeightTransferImpl::process_weights_send_request(
    const std::string& session_id,
    const std::vector<int32_t>& layer_ids) {
  sender_engine_.process_weights_send_request(session_id, layer_ids);
}

void HcclWeightTransferImpl::process_weights_send_request(
    const std::string& session_id,
    const std::vector<int32_t>& layer_ids,
    const std::unordered_map<int32_t, std::vector<int32_t>>& layer_expert_ids,
    bool include_non_expert) {
  sender_engine_.process_weights_send_request(
      session_id, layer_ids, layer_expert_ids, include_non_expert);
}

bool HcclWeightTransferImpl::process_weights_alltoallv_send_request(
    const std::string& session_id,
    uint32_t receiver_rank,
    const std::vector<xllm::proto::AlltoAllRoundDesc>& alltoall_rounds,
    uint64_t* transferred_bytes,
    std::string* error_msg) {
  return sender_engine_.process_weights_alltoallv_send_request(
      session_id, receiver_rank, alltoall_rounds, transferred_bytes, error_msg);
}

bool HcclWeightTransferImpl::pull_model_from_instance(
    const std::string& remote_addr,
    const RankExpertTransferPlanData& rank_expert_transfer_plan) {
  return receiver_engine_.pull_model_from_instance(remote_addr,
                                                   rank_expert_transfer_plan);
}

bool HcclWeightTransferImpl::handle_init_comm(const std::string& remote_addr,
                                              const void* root_info_ptr,
                                              uint32_t n_ranks,
                                              uint32_t rank,
                                              const std::string& session_id,
                                              xllm::proto::CommMode comm_mode) {
  return sender_engine_.handle_init_comm(
      remote_addr, root_info_ptr, n_ranks, rank, session_id, comm_mode);
}

const std::vector<at::Tensor>& HcclWeightTransferImpl::get_registered_tensors(
    int32_t layer_id) const {
  return sender_engine_.get_registered_tensors(layer_id);
}

std::string HcclWeightTransferImpl::get_weight_transfer_addr() const {
  return local_addr_;
}

WeightTransferSessionManager* HcclWeightTransferImpl::session_manager() {
  return &session_manager_;
}

WeightTransferSenderEngine* HcclWeightTransferImpl::sender_engine() {
  return &sender_engine_;
}

}  // namespace xllm
