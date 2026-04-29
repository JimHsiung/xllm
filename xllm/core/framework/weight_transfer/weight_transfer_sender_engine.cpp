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

#include "framework/weight_transfer/weight_transfer_sender_engine.h"

#include <glog/logging.h>

#include <iomanip>

#include "framework/weight_transfer/weight_transfer_common.h"

namespace xllm {

WeightTransferSenderEngine::WeightTransferSenderEngine(
    const ModelContext& context,
    CausalLM* model,
    int32_t device_id,
    const std::string& local_addr,
    WeightTransferSessionManager* session_manager,
    std::unordered_map<int32_t, std::vector<at::Tensor>>* layer_registry)
    : context_(context),
      model_(model),
      device_id_(device_id),
      local_addr_(local_addr),
      session_manager_(session_manager),
      layer_registry_(layer_registry) {}

bool WeightTransferSenderEngine::handle_init_comm(
    const std::string& remote_addr,
    const void* root_info_ptr,
    uint32_t n_ranks,
    uint32_t rank,
    const std::string& session_id,
    xllm::proto::CommMode comm_mode) {
  return init_collective_comm_as_sender(
      remote_addr, root_info_ptr, n_ranks, rank, session_id, comm_mode);
}

void WeightTransferSenderEngine::register_layer(
    int32_t layer_id,
    const std::vector<at::Tensor>& tensors) {
  (*layer_registry_)[layer_id] = tensors;
}

const std::vector<at::Tensor>&
WeightTransferSenderEngine::get_registered_tensors(int32_t layer_id) const {
  static const std::vector<at::Tensor> k_empty_tensors;
  auto it = layer_registry_->find(layer_id);
  if (it == layer_registry_->end()) {
    return k_empty_tensors;
  }
  return it->second;
}

bool WeightTransferSenderEngine::init_collective_comm_as_sender(
    const std::string& remote_addr,
    const void* root_info_ptr,
    uint32_t n_ranks,
    uint32_t rank,
    const std::string& session_id,
    xllm::proto::CommMode comm_mode) {
  if (n_ranks < 2) {
    LOG(ERROR) << "Invalid n_ranks for sender init: " << n_ranks;
    return false;
  }
  if (rank >= n_ranks) {
    LOG(ERROR) << "Invalid sender rank " << rank << " for n_ranks " << n_ranks;
    return false;
  }
  auto session_ctx =
      session_manager_->create_session_context(session_id, comm_mode);
  if (session_ctx == nullptr) {
    return false;
  }
  std::lock_guard<std::mutex> session_lock(session_ctx->operation_mutex);

  LOG(INFO) << "Sender: Initializing HCCL Comm with peer " << remote_addr
            << ", n_ranks=" << n_ranks << ", rank=" << rank
            << ", mode=" << comm_mode << ", session_id=" << session_id;
  aclrtSetDevice(device_id_);

  HcclRootInfo root_info;
  memcpy(&root_info, root_info_ptr, sizeof(HcclRootInfo));

  auto ret =
      HcclCommInitRootInfo(n_ranks, &root_info, rank, &session_ctx->hccl_comm);
  if (ret != HCCL_SUCCESS) {
    LOG(ERROR) << "HcclCommInitRootInfo failed: " << ret;
    session_manager_->destroy_session_context_async(session_id, true);
    return false;
  }
  session_ctx->n_ranks = n_ranks;
  session_ctx->rank = rank;
  session_ctx->comm_mode = comm_mode;
  session_ctx->is_comm_initialized.store(true);
  return true;
}

void WeightTransferSenderEngine::process_weights_send_request(
    const std::string& session_id,
    const std::vector<int32_t>& layer_ids) {
  process_weights_send_request_internal(
      session_id, layer_ids, LayerExpertIdsMap{}, true, true);
}

void WeightTransferSenderEngine::process_weights_send_request(
    const std::string& session_id,
    const std::vector<int32_t>& layer_ids,
    const LayerExpertIdsMap& layer_expert_ids,
    bool include_non_expert) {
  process_weights_send_request_internal(
      session_id, layer_ids, layer_expert_ids, include_non_expert, false);
}

void WeightTransferSenderEngine::process_weights_send_request_internal(
    const std::string& session_id,
    const std::vector<int32_t>& layer_ids,
    const LayerExpertIdsMap& layer_expert_ids,
    bool include_non_expert,
    bool transfer_all_experts) {
  if (session_id.empty()) {
    LOG(ERROR) << "Sender batch transfer requires non-empty session_id.";
    return;
  }
  SenderSessionLogGuard sender_session_log(session_id, "p2p");

  auto cleanup_session_async = [this](const std::string& cleanup_session_id) {
    if (cleanup_session_id.empty()) {
      return;
    }
    session_manager_->destroy_session_context_async(cleanup_session_id, true);
  };

  auto session_ctx = session_manager_->get_session_context(session_id);
  if (session_ctx == nullptr) {
    LOG(ERROR) << "Sender batch transfer session not found: " << session_id;
    return;
  }
  if (!session_manager_->wait_for_session_ready(session_ctx,
                                                "Sender batch transfer")) {
    cleanup_session_async(session_id);
    return;
  }
  HcclComm target_hccl_comm = session_ctx->hccl_comm;
  aclrtStream target_stream = session_ctx->stream;

  aclrtSetDevice(device_id_);
  auto normalized_layer_expert_ids =
      normalize_layer_expert_ids_map(layer_expert_ids);
  if (!validate_layer_expert_ids_map(
          layer_ids, normalized_layer_expert_ids, "Sender layer expert map")) {
    LOG(ERROR) << "Invalid sender layer expert map.";
    cleanup_session_async(session_id);
    return;
  }

  auto expert_indices_set_ptr = get_cached_expert_indices_set(model_);
  int32_t ep_size = get_ep_size(context_.get_parallel_args());
  int32_t ep_rank = get_ep_rank(context_.get_parallel_args());

  auto promise = std::make_shared<std::promise<bool>>();
  std::future<bool> future = promise->get_future();
  auto session_ctx_for_thread = session_ctx;

  session_manager_->schedule_hccl_task([&,
                                        layer_ids,
                                        normalized_layer_expert_ids,
                                        include_non_expert,
                                        transfer_all_experts,
                                        target_hccl_comm,
                                        target_stream,
                                        session_ctx_for_thread,
                                        expert_indices_set_ptr,
                                        ep_size,
                                        ep_rank]() mutable {
    std::unique_lock<std::mutex> session_lock;
    if (session_ctx_for_thread != nullptr) {
      session_lock =
          std::unique_lock<std::mutex>(session_ctx_for_thread->operation_mutex);
    }
    aclrtSetDevice(device_id_);

    const auto& expert_indices_set = *expert_indices_set_ptr;

    absl::Time start_time = absl::Now();
    size_t total_nbytes = 0;

    std::vector<HcclSendRecvItem> items;
    for (int32_t layer_id : layer_ids) {
      const auto& tensors = get_registered_tensors(layer_id);
      auto it = normalized_layer_expert_ids.find(layer_id);
      const std::vector<int32_t> layer_ids_for_experts =
          it == normalized_layer_expert_ids.end() ? std::vector<int32_t>{}
                                                  : it->second;
      if (!append_layer_transfer_items(tensors,
                                       layer_id,
                                       layer_ids_for_experts,
                                       expert_indices_set,
                                       include_non_expert,
                                       transfer_all_experts,
                                       ep_rank,
                                       ep_size,
                                       HCCL_SEND,
                                       0,
                                       &items,
                                       &total_nbytes)) {
        promise->set_value(false);
        return;
      }
    }

    if (!items.empty()) {
      auto hccl_ret = HcclBatchSendRecv(
          items.data(), items.size(), target_hccl_comm, target_stream);
      if (hccl_ret != HCCL_SUCCESS) {
        LOG(ERROR)
            << "[Sender Thread] HcclBatchSendRecv (Multiple Layers) Failed.";
        promise->set_value(false);
        return;
      }
    }
    auto sync_ret = aclrtSynchronizeStream(target_stream);

    absl::Time end_time = absl::Now();
    double duration_s = absl::ToDoubleSeconds(end_time - start_time);
    double duration_ms = absl::ToDoubleMilliseconds(end_time - start_time);
    double total_gb = total_nbytes / (1024.0 * 1024.0 * 1024.0);
    double bandwidth_gb_s = total_gb / duration_s;

    LOG(INFO) << "[Sender Thread] Batch transfer (layers: " << layer_ids.size()
              << ", include_non_expert=" << include_non_expert
              << ", transfer_all_experts=" << transfer_all_experts
              << ", requested_expert_ids="
              << count_expert_ids(normalized_layer_expert_ids)
              << "): " << std::fixed << std::setprecision(2) << total_gb
              << " GB, "
              << "Time: " << duration_ms << " ms, "
              << "Bandwidth: " << bandwidth_gb_s << " GB/s";

    promise->set_value(sync_ret == ACL_SUCCESS);
  });
  bool result = future.get();
  if (!result) {
    LOG(ERROR) << "Sender batch transfer failed.";
  } else {
    sender_session_log.mark_success();
  }
  cleanup_session_async(session_id);
}

bool WeightTransferSenderEngine::process_weights_alltoallv_send_request(
    const std::string& session_id,
    uint32_t receiver_rank,
    const std::vector<xllm::proto::AlltoAllRoundDesc>& alltoall_rounds,
    uint64_t* transferred_bytes,
    std::string* error_msg) {
  SenderSessionLogGuard sender_session_log(session_id, "alltoallv");

  if (transferred_bytes != nullptr) {
    *transferred_bytes = 0;
  }
  if (error_msg != nullptr) {
    error_msg->clear();
  }

  auto session_ctx = session_manager_->get_session_context(session_id);
  if (session_ctx == nullptr) {
    if (error_msg != nullptr) {
      *error_msg = "session not found";
    }
    return false;
  }

  if (!session_manager_->wait_for_session_ready(session_ctx,
                                                "Sender alltoallv transfer")) {
    if (error_msg != nullptr) {
      *error_msg = "comm not ready";
    }
    session_manager_->destroy_session_context_async(session_id, true);
    return false;
  }
  std::unique_lock<std::mutex> session_lock(session_ctx->operation_mutex);

  uint32_t active_n_ranks = session_ctx->n_ranks;
  uint32_t active_rank = session_ctx->rank;
  xllm::proto::CommMode active_comm_mode = session_ctx->comm_mode;
  if (active_comm_mode != xllm::proto::COMM_MODE_EXPERT_ALLTOALLV) {
    if (error_msg != nullptr) {
      *error_msg = "active comm mode is not alltoallv";
    }
    session_manager_->destroy_session_context_async(session_id, true);
    return false;
  }
  if (receiver_rank >= active_n_ranks) {
    if (error_msg != nullptr) {
      *error_msg = "invalid receiver rank";
    }
    LOG(ERROR) << "Invalid receiver rank " << receiver_rank << " for n_ranks "
               << active_n_ranks;
    session_manager_->destroy_session_context_async(session_id, true);
    return false;
  }
  if (alltoall_rounds.empty()) {
    sender_session_log.mark_success();
    session_manager_->destroy_session_context_async(session_id, true);
    return true;
  }

  aclrtSetDevice(device_id_);
  auto dummy_options = torch::TensorOptions()
                           .dtype(torch::kUInt8)
                           .device("npu:" + std::to_string(device_id_));
  at::Tensor dummy_send_tensor = at::empty({1}, dummy_options);
  at::Tensor dummy_recv_tensor = at::empty({1}, dummy_options);
  void* default_send_buf = dummy_send_tensor.data_ptr();
  void* default_recv_buf = dummy_recv_tensor.data_ptr();
  absl::Time start_time = absl::Now();
  uint64_t total_bytes = 0;
  for (const auto& round_desc : alltoall_rounds) {
    std::vector<uint64_t> send_counts(active_n_ranks, 0);
    std::vector<uint64_t> sdispls(active_n_ranks, 0);
    std::vector<uint64_t> recv_counts(active_n_ranks, 0);
    std::vector<uint64_t> rdispls(active_n_ranks, 0);

    void* send_buf = default_send_buf;
    void* recv_buf = default_recv_buf;

    if (round_desc.local_expert_count() > 0) {
      const auto& tensors = get_registered_tensors(round_desc.layer_id());
      if (round_desc.tensor_idx() < 0 ||
          round_desc.tensor_idx() >= static_cast<int32_t>(tensors.size())) {
        if (error_msg != nullptr) {
          *error_msg = "invalid tensor index in alltoall round";
        }
        LOG(ERROR) << "Invalid tensor index " << round_desc.tensor_idx()
                   << " at layer " << round_desc.layer_id();
        session_manager_->destroy_session_context_async(session_id, true);
        return false;
      }
      const auto& tensor = tensors[round_desc.tensor_idx()];
      if (tensor.dim() != 3) {
        if (error_msg != nullptr) {
          *error_msg = "alltoall round tensor is not 3D";
        }
        LOG(ERROR) << "Alltoall source tensor is not 3D at layer "
                   << round_desc.layer_id() << ", tensor index "
                   << round_desc.tensor_idx();
        session_manager_->destroy_session_context_async(session_id, true);
        return false;
      }

      int64_t local_expert_num = tensor.size(0);
      int32_t local_start = round_desc.local_expert_start();
      int32_t local_count = round_desc.local_expert_count();
      if (local_start < 0 || local_count < 0 ||
          local_start + local_count > local_expert_num) {
        if (error_msg != nullptr) {
          *error_msg = "alltoall round range out of tensor local expert";
        }
        LOG(ERROR) << "Invalid alltoall source range at layer "
                   << round_desc.layer_id() << ", tensor index "
                   << round_desc.tensor_idx() << ", local_start=" << local_start
                   << ", local_count=" << local_count
                   << ", local_expert_num=" << local_expert_num;
        session_manager_->destroy_session_context_async(session_id, true);
        return false;
      }

      uint64_t expert_unit_bytes = get_expert_unit_bytes(
          tensor, round_desc.layer_id(), round_desc.tensor_idx());
      if (expert_unit_bytes == 0) {
        if (error_msg != nullptr) {
          *error_msg = "invalid expert tensor byte shape";
        }
        session_manager_->destroy_session_context_async(session_id, true);
        return false;
      }
      send_buf = tensor.data_ptr();
      send_counts[receiver_rank] = expert_unit_bytes * local_count;
      sdispls[receiver_rank] = expert_unit_bytes * local_start;
      total_bytes += send_counts[receiver_rank];
    }

    auto ret = HcclAlltoAllV(send_buf,
                             send_counts.data(),
                             sdispls.data(),
                             HCCL_DATA_TYPE_UINT8,
                             recv_buf,
                             recv_counts.data(),
                             rdispls.data(),
                             HCCL_DATA_TYPE_UINT8,
                             session_ctx->hccl_comm,
                             session_ctx->stream);
    if (ret != HCCL_SUCCESS) {
      if (error_msg != nullptr) {
        *error_msg = "HcclAlltoAllV failed on sender";
      }
      LOG(ERROR) << "HcclAlltoAllV failed on sender rank " << active_rank
                 << ", session_id=" << session_id << ", ret=" << ret;
      session_manager_->destroy_session_context_async(session_id, true);
      return false;
    }
  }

  auto sync_ret = aclrtSynchronizeStream(session_ctx->stream);
  if (sync_ret != ACL_SUCCESS) {
    if (error_msg != nullptr) {
      *error_msg = "aclrtSynchronizeStream failed on sender";
    }
    LOG(ERROR) << "aclrtSynchronizeStream failed on sender alltoallv session "
               << session_id << ", ret=" << sync_ret;
    session_manager_->destroy_session_context_async(session_id, true);
    return false;
  }
  if (transferred_bytes != nullptr) {
    *transferred_bytes = total_bytes;
  }
  absl::Time end_time = absl::Now();
  double duration_s = absl::ToDoubleSeconds(end_time - start_time);
  double duration_ms = absl::ToDoubleMilliseconds(end_time - start_time);
  double total_gb =
      static_cast<double>(total_bytes) / (1024.0 * 1024.0 * 1024.0);
  double bandwidth_gb_s = duration_s > 0 ? total_gb / duration_s : 0.0;

  LOG(INFO) << "[Sender Thread] Alltoallv transfer (rounds: "
            << alltoall_rounds.size() << ", rank=" << active_rank
            << ", session_id=" << session_id << "): " << std::fixed
            << std::setprecision(2) << total_gb << " GB, "
            << "Time: " << duration_ms << " ms, "
            << "Bandwidth: " << bandwidth_gb_s << " GB/s";
  sender_session_log.mark_success();
  session_manager_->destroy_session_context_async(session_id, true);
  return true;
}

}  // namespace xllm
