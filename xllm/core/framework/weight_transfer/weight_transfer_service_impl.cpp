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

#include "framework/weight_transfer/weight_transfer_service_impl.h"

#include <brpc/closure_guard.h>
#include <glog/logging.h>
#include <torch_npu/csrc/core/npu/NPUFormat.h>

#include <cstdint>
#include <utility>

#include "framework/weight_transfer/hccl_weight_transfer_impl.h"
#include "framework/weight_transfer/weight_transfer_common.h"

namespace xllm {

WeightTransferServiceImpl::WeightTransferServiceImpl(
    HcclWeightTransferImpl* impl)
    : impl_(impl) {}

void WeightTransferServiceImpl::InitComm(
    google::protobuf::RpcController* controller,
    const xllm::proto::InitCommRequest* request,
    xllm::proto::InitCommResponse* response,
    google::protobuf::Closure* done) {
  brpc::ClosureGuard done_guard(done);

  std::string remote_addr = request->addr();
  std::string root_info_str = request->root_info();
  uint32_t n_ranks = request->n_ranks() == 0 ? 2 : request->n_ranks();
  uint32_t rank = request->rank() == 0 ? 1 : request->rank();
  std::string session_id = request->session_id();
  if (session_id.empty()) {
    session_id = "p2p-default";
  }
  auto comm_mode = request->comm_mode();

  impl_->session_manager()->schedule_background_task([this,
                                                      remote_addr,
                                                      root_info_str,
                                                      n_ranks,
                                                      rank,
                                                      session_id,
                                                      comm_mode]() {
    LOG(INFO) << "Sender Async Thread: Start Waiting for Receiver to join "
                 "HCCL group...";

    impl_->handle_init_comm(remote_addr,
                            root_info_str.data(),
                            n_ranks,
                            rank,
                            session_id,
                            comm_mode);

    LOG(INFO) << "Sender Async Thread: HCCL Init DONE! Handshake complete.";
  });
  response->set_success(true);
  LOG(INFO)
      << "Sender: RootInfo generated and sent back. Async Init triggered.";
}

void WeightTransferServiceImpl::GetWeightsMeta(
    google::protobuf::RpcController* controller,
    const xllm::proto::GetWeightsMetaRequest* request,
    xllm::proto::GetWeightsMetaResponse* response,
    google::protobuf::Closure* done) {
  brpc::ClosureGuard done_guard(done);
  for (int32_t layer_id : request->layer_ids()) {
    auto* layer_meta = response->add_layer_metas();
    layer_meta->set_layer_id(layer_id);
    const auto& tensors = impl_->get_registered_tensors(layer_id);
    for (const auto& t : tensors) {
      auto* meta = layer_meta->add_metas();
      meta->set_dtype(static_cast<int32_t>(t.scalar_type()));
      for (int i = 0; i < t.dim(); ++i) {
        meta->add_shape(t.size(i));
      }
      meta->set_npu_format(at_npu::native::get_npu_format(t));
    }
    auto storage_info = get_contiguous_layer_storage_info(
        tensors, impl_->get_registered_layer_storage(layer_id));
    if (!storage_info.available) {
      continue;
    }
    layer_meta->set_has_contiguous_storage(true);
    layer_meta->set_storage_size(storage_info.storage_size);
    const uintptr_t base_addr =
        reinterpret_cast<uintptr_t>(storage_info.base_ptr);
    for (const auto& t : tensors) {
      auto* slice_meta = layer_meta->add_slice_metas();
      const uintptr_t tensor_addr = reinterpret_cast<uintptr_t>(t.data_ptr());
      slice_meta->set_offset(tensor_addr - base_addr);
      slice_meta->set_bytes(static_cast<uint64_t>(t.nbytes()));
      slice_meta->set_dtype(static_cast<int32_t>(t.scalar_type()));
      for (int i = 0; i < t.dim(); ++i) {
        slice_meta->add_shape(t.size(i));
      }
      slice_meta->set_npu_format(at_npu::native::get_npu_format(t));
    }
  }
}

void WeightTransferServiceImpl::TriggerWeightsSend(
    google::protobuf::RpcController* controller,
    const xllm::proto::TriggerWeightsSendRequest* request,
    xllm::proto::TriggerWeightsSendResponse* response,
    google::protobuf::Closure* done) {
  brpc::ClosureGuard done_guard(done);
  const std::string session_id = request->session_id();
  if (session_id.empty()) {
    response->set_success(false);
    response->set_error_msg("session_id is required");
    return;
  }
  if (request->use_alltoallv()) {
    std::vector<xllm::proto::AlltoAllRoundDesc> alltoall_rounds;
    alltoall_rounds.reserve(request->alltoall_rounds_size());
    for (const auto& round_desc : request->alltoall_rounds()) {
      alltoall_rounds.push_back(round_desc);
    }
    const uint32_t receiver_rank = request->receiver_rank();
    impl_->session_manager()->schedule_background_task(
        [this, alltoall_rounds, session_id, receiver_rank]() {
          uint64_t transferred_bytes = 0;
          std::string error_msg;
          bool ok =
              impl_->process_weights_alltoallv_send_request(session_id,
                                                            receiver_rank,
                                                            alltoall_rounds,
                                                            &transferred_bytes,
                                                            &error_msg);
          if (!ok) {
            LOG(ERROR) << "Async alltoallv sender task failed. session_id="
                       << session_id << ", error=" << error_msg;
          }
        });
    response->set_success(true);
    response->set_transferred_bytes(0);
    return;
  }

  std::vector<int32_t> layer_ids;
  for (int32_t id : request->layer_ids()) {
    layer_ids.push_back(id);
  }
  const bool has_extended_fields =
      request->include_non_expert() || request->layer_expert_ids_size() > 0 ||
      request->use_layer_storage_transfer() || request->transfer_all_experts();
  if (!has_extended_fields) {
    impl_->process_weights_send_request(session_id, layer_ids);
    response->set_success(true);
    return;
  }

  auto layer_expert_ids = parse_layer_expert_ids(*request);
  impl_->process_weights_send_request(session_id,
                                      layer_ids,
                                      layer_expert_ids,
                                      request->include_non_expert(),
                                      request->use_layer_storage_transfer(),
                                      request->transfer_all_experts());
  response->set_success(true);
  response->set_transferred_bytes(0);
}

}  // namespace xllm
