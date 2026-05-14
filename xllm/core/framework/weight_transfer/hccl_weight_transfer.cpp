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

#include "framework/weight_transfer/hccl_weight_transfer.h"

#include "framework/weight_transfer/hccl_weight_transfer_impl.h"

namespace xllm {

HcclWeightTransfer::HcclWeightTransfer(const ModelContext& context,
                                       CausalLM* model,
                                       int32_t device_id,
                                       int32_t listen_port)
    : impl_(std::make_unique<HcclWeightTransferImpl>(context,
                                                     model,
                                                     device_id,
                                                     listen_port)) {}

HcclWeightTransfer::~HcclWeightTransfer() = default;

void HcclWeightTransfer::register_layer(
    int32_t layer_id,
    const std::vector<at::Tensor>& tensors) {
  impl_->register_layer(layer_id, tensors);
}

void HcclWeightTransfer::register_layer_storage(int32_t layer_id,
                                                layer::BaseLoader* loader) {
  impl_->register_layer_storage(layer_id, loader);
}

void HcclWeightTransfer::start_serving() { impl_->start_serving(); }

void HcclWeightTransfer::process_weights_send_request(
    const std::string& session_id,
    const std::vector<int32_t>& layer_ids) {
  impl_->process_weights_send_request(session_id, layer_ids);
}

void HcclWeightTransfer::process_weights_send_request(
    const std::string& session_id,
    const std::vector<int32_t>& layer_ids,
    const std::unordered_map<int32_t, std::vector<int32_t>>& layer_expert_ids,
    bool include_non_expert) {
  impl_->process_weights_send_request(
      session_id, layer_ids, layer_expert_ids, include_non_expert);
}

bool HcclWeightTransfer::process_weights_alltoallv_send_request(
    const std::string& session_id,
    uint32_t receiver_rank,
    const std::vector<xllm::proto::AlltoAllRoundDesc>& alltoall_rounds,
    uint64_t* transferred_bytes,
    std::string* error_msg) {
  return impl_->process_weights_alltoallv_send_request(
      session_id, receiver_rank, alltoall_rounds, transferred_bytes, error_msg);
}

bool HcclWeightTransfer::pull_model_from_instance(
    const std::string& remote_addr,
    const RankExpertTransferPlanData& rank_expert_transfer_plan) {
  return impl_->pull_model_from_instance(remote_addr,
                                         rank_expert_transfer_plan);
}

bool HcclWeightTransfer::handle_init_comm(const std::string& remote_addr,
                                          const void* root_info_ptr,
                                          uint32_t n_ranks,
                                          uint32_t rank,
                                          const std::string& session_id,
                                          xllm::proto::CommMode comm_mode) {
  return impl_->handle_init_comm(
      remote_addr, root_info_ptr, n_ranks, rank, session_id, comm_mode);
}

const std::vector<at::Tensor>& HcclWeightTransfer::get_registered_tensors(
    int32_t layer_id) const {
  return impl_->get_registered_tensors(layer_id);
}

std::string HcclWeightTransfer::get_weight_transfer_addr() const {
  return impl_->get_weight_transfer_addr();
}

}  // namespace xllm
