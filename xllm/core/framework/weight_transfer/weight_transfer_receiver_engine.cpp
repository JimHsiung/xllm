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

#include "framework/weight_transfer/weight_transfer_receiver_engine.h"

#include <gflags/gflags.h>
#include <glog/logging.h>

#include <algorithm>
#include <iomanip>
#include <unordered_set>
#include <utility>

#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "acl/acl_rt.h"
#include "common/global_flags.h"
#include "core/layers/npu/loader/base_loader.h"
#include "framework/model/model_args.h"
#include "framework/weight_transfer/weight_transfer_common.h"

namespace xllm {
namespace {

bool is_valid_layer_storage_meta(
    const xllm::proto::LayerWeightsMeta& layer_meta) {
  if (!layer_meta.has_contiguous_storage() || layer_meta.storage_size() == 0 ||
      layer_meta.metas_size() == 0 ||
      layer_meta.slice_metas_size() != layer_meta.metas_size()) {
    return false;
  }

  for (int j = 0; j < layer_meta.slice_metas_size(); ++j) {
    const auto& tensor_meta = layer_meta.metas(j);
    const auto& slice_meta = layer_meta.slice_metas(j);
    if (slice_meta.bytes() == 0 ||
        slice_meta.offset() + slice_meta.bytes() < slice_meta.offset() ||
        slice_meta.offset() + slice_meta.bytes() > layer_meta.storage_size() ||
        slice_meta.dtype() != tensor_meta.dtype() ||
        slice_meta.npu_format() != tensor_meta.npu_format() ||
        slice_meta.shape_size() != tensor_meta.shape_size()) {
      return false;
    }
    for (int k = 0; k < slice_meta.shape_size(); ++k) {
      if (slice_meta.shape(k) != tensor_meta.shape(k)) {
        return false;
      }
    }
  }
  return true;
}

void release_receiver_layer_storage_map(
    std::unordered_map<int32_t, ReceiverLayerStorage>* layer_storages) {
  if (layer_storages == nullptr) {
    return;
  }
  for (auto& entry : *layer_storages) {
    auto& storage = entry.second;
    storage.base_ptr = nullptr;
    storage.available = false;
    storage.storage_size = 0;
  }
  layer_storages->clear();
}

uint64_t sum_layer_storage_payload_nbytes(
    const xllm::proto::LayerWeightsMeta& layer_meta) {
  uint64_t payload_nbytes = 0;
  for (int j = 0; j < layer_meta.slice_metas_size(); ++j) {
    const auto& slice_meta = layer_meta.slice_metas(j);
    payload_nbytes += slice_meta.bytes();
  }
  return payload_nbytes;
}

void allocate_tensors_from_tensor_meta(
    const xllm::proto::LayerWeightsMeta& layer_meta,
    int32_t device_id,
    std::vector<at::Tensor>* tensors) {
  CHECK(tensors != nullptr);
  tensors->resize(layer_meta.metas_size());

  for (int j = 0; j < layer_meta.metas_size(); ++j) {
    const auto& meta = layer_meta.metas(j);
    std::vector<int64_t> shape;
    shape.reserve(meta.shape_size());
    for (int64_t dim : meta.shape()) {
      shape.push_back(dim);
    }

    auto options = torch::TensorOptions()
                       .dtype(static_cast<at::ScalarType>(meta.dtype()))
                       .device("npu:" + std::to_string(device_id));

    (*tensors)[j] =
        at_npu::native::empty_with_format(shape, options, meta.npu_format());
  }
}

}  // namespace

WeightTransferReceiverEngine::WeightTransferReceiverEngine(
    const ModelContext& context,
    CausalLM* model,
    int32_t device_id,
    const std::string& local_addr,
    WeightTransferSessionManager* session_manager,
    const WeightTransferAlltoallPlanner* planner)
    : context_(context),
      model_(model),
      device_id_(device_id),
      local_addr_(local_addr),
      session_manager_(session_manager),
      planner_(planner) {}

WeightTransferReceiverEngine::~WeightTransferReceiverEngine() {
  release_layer_storage_views();
}

bool WeightTransferReceiverEngine::init_collective_comm_as_receiver(
    const std::vector<std::string>& source_addrs,
    uint32_t receiver_rank,
    const std::string& session_id) {
  if (source_addrs.empty()) {
    LOG(ERROR) << "No source addresses for collective comm init.";
    return false;
  }
  auto session_ctx = session_manager_->create_session_context(
      session_id, xllm::proto::COMM_MODE_EXPERT_ALLTOALLV);
  if (session_ctx == nullptr) {
    return false;
  }
  std::lock_guard<std::mutex> session_lock(session_ctx->operation_mutex);
  aclrtSetDevice(device_id_);

  const uint32_t n_ranks = static_cast<uint32_t>(source_addrs.size() + 1);
  if (receiver_rank >= n_ranks) {
    LOG(ERROR) << "Invalid receiver rank " << receiver_rank << " for n_ranks "
               << n_ranks;
    session_manager_->destroy_session_context_async(session_id, true);
    return false;
  }

  HcclRootInfo root_info;
  auto ret = HcclGetRootInfo(&root_info);
  if (ret != HCCL_SUCCESS) {
    LOG(ERROR) << "HcclGetRootInfo failed for collective init.";
    session_manager_->destroy_session_context_async(session_id, true);
    return false;
  }

  std::vector<std::shared_ptr<CachedRpcEndpoint>> endpoints(
      source_addrs.size());
  for (size_t i = 0; i < source_addrs.size(); ++i) {
    if (!get_cached_rpc_endpoint(source_addrs[i], &endpoints[i])) {
      session_manager_->destroy_session_context_async(session_id, true);
      return false;
    }
  }

  std::vector<std::future<bool>> init_comm_futures;
  init_comm_futures.reserve(source_addrs.size());
  for (size_t i = 0; i < source_addrs.size(); ++i) {
    xllm::proto::InitCommRequest req;
    req.set_addr(local_addr_);
    req.set_root_info(&root_info, sizeof(HcclRootInfo));
    req.set_n_ranks(n_ranks);
    req.set_rank(static_cast<uint32_t>(i + 1));
    req.set_session_id(session_id);
    req.set_comm_mode(xllm::proto::COMM_MODE_EXPERT_ALLTOALLV);
    auto endpoint = endpoints[i];
    const std::string source_addr = source_addrs[i];
    init_comm_futures.emplace_back(
        std::async(std::launch::async, [endpoint, req, source_addr]() {
          return call_init_comm_with_retry(
              endpoint->stub.get(), req, source_addr);
        }));
  }
  for (auto& future : init_comm_futures) {
    if (!future.get()) {
      session_manager_->destroy_session_context_async(session_id, true);
      return false;
    }
  }

  ret = HcclCommInitRootInfo(
      n_ranks, &root_info, receiver_rank, &session_ctx->hccl_comm);
  if (ret != HCCL_SUCCESS) {
    LOG(ERROR) << "Receiver HcclCommInitRootInfo failed for alltoallv, ret="
               << ret;
    session_manager_->destroy_session_context_async(session_id, true);
    return false;
  }
  session_ctx->n_ranks = n_ranks;
  session_ctx->rank = receiver_rank;
  session_ctx->comm_mode = xllm::proto::COMM_MODE_EXPERT_ALLTOALLV;
  session_ctx->is_comm_initialized.store(true);
  LOG(INFO) << "Receiver collective comm initialized. n_ranks=" << n_ranks
            << ", receiver_rank=" << receiver_rank
            << ", session_id=" << session_id;
  return true;
}

bool WeightTransferReceiverEngine::pull_expert_weights_alltoallv(
    const std::vector<std::string>& source_addrs,
    const std::vector<xllm::proto::AlltoAllRoundDesc>& receiver_rounds,
    const std::vector<xllm::proto::TriggerWeightsSendRequest>& source_requests,
    const std::string& session_id,
    bool comm_initialized) {
  if (source_addrs.empty()) {
    LOG(ERROR) << "No source addresses for pull_expert_weights_alltoallv.";
    return false;
  }
  if (source_addrs.size() != source_requests.size()) {
    LOG(ERROR) << "Source address size " << source_addrs.size()
               << " mismatches source request size " << source_requests.size();
    return false;
  }
  if (!comm_initialized) {
    if (!init_collective_comm_as_receiver(source_addrs, 0, session_id)) {
      return false;
    }
  }
  auto session_ctx = session_manager_->get_session_context(session_id);
  if (session_ctx == nullptr) {
    LOG(ERROR) << "Receiver alltoall session not found. session_id="
               << session_id;
    return false;
  }
  std::unique_lock<std::mutex> session_lock(session_ctx->operation_mutex);

  std::vector<std::shared_ptr<CachedRpcEndpoint>> endpoints(
      source_addrs.size());
  for (size_t i = 0; i < source_addrs.size(); ++i) {
    if (!get_cached_rpc_endpoint(source_addrs[i], &endpoints[i])) {
      session_manager_->destroy_session_context_async(session_id, true);
      return false;
    }
  }

  struct TriggerAckResult {
    bool ok = false;
    std::string error;
  };
  std::vector<std::future<TriggerAckResult>> trigger_futures;
  trigger_futures.reserve(source_addrs.size());
  for (size_t i = 0; i < source_addrs.size(); ++i) {
    auto endpoint = endpoints[i];
    const auto req = source_requests[i];
    const std::string source_addr = source_addrs[i];
    trigger_futures.emplace_back(std::async(
        std::launch::async, [endpoint, req, source_addr]() -> TriggerAckResult {
          brpc::Controller cntl;
          cntl.set_timeout_ms(30000);
          xllm::proto::TriggerWeightsSendResponse resp;
          endpoint->stub->TriggerWeightsSend(&cntl, &req, &resp, nullptr);
          if (cntl.Failed()) {
            return TriggerAckResult{false,
                                    "RPC failed, source=" + source_addr +
                                        ", error=" + cntl.ErrorText()};
          }
          if (!resp.success()) {
            return TriggerAckResult{false,
                                    "source rejected, source=" + source_addr +
                                        ", error=" + resp.error_msg()};
          }
          return TriggerAckResult{true, ""};
        }));
  }
  for (auto& future : trigger_futures) {
    auto ack_result = future.get();
    if (!ack_result.ok) {
      LOG(ERROR) << "TriggerWeightsSend ack failed: " << ack_result.error;
      session_manager_->destroy_session_context_async(session_id, true);
      return false;
    }
  }

  uint32_t active_n_ranks = session_ctx->n_ranks;
  if (active_n_ranks != source_addrs.size() + 1) {
    LOG(ERROR) << "Unexpected n_ranks for receiver alltoallv: "
               << active_n_ranks << ", expected " << source_addrs.size() + 1;
    session_manager_->destroy_session_context_async(session_id, true);
    return false;
  }

  aclrtSetDevice(device_id_);
  auto dummy_options = torch::TensorOptions()
                           .dtype(torch::kUInt8)
                           .device("npu:" + std::to_string(device_id_));
  at::Tensor dummy_send_tensor = at::empty({1}, dummy_options);
  void* default_send_buf = dummy_send_tensor.data_ptr();
  absl::Time start_time = absl::Now();
  uint64_t total_bytes = 0;
  for (size_t round_idx = 0; round_idx < receiver_rounds.size(); ++round_idx) {
    const auto& receiver_round = receiver_rounds[round_idx];
    if (receiver_round.layer_id() < 0 ||
        receiver_round.layer_id() >= context_.get_model_args().n_layers()) {
      LOG(ERROR) << "Invalid receiver layer id " << receiver_round.layer_id();
      session_manager_->destroy_session_context_async(session_id, true);
      return false;
    }
    auto tensors = model_->get_decoder_layer_weight(receiver_round.layer_id());
    if (receiver_round.tensor_idx() < 0 ||
        receiver_round.tensor_idx() >= static_cast<int32_t>(tensors.size())) {
      LOG(ERROR) << "Invalid receiver tensor index "
                 << receiver_round.tensor_idx() << " at layer "
                 << receiver_round.layer_id();
      session_manager_->destroy_session_context_async(session_id, true);
      return false;
    }
    auto& tensor = tensors[receiver_round.tensor_idx()];
    if (tensor.dim() != 3) {
      LOG(ERROR) << "Receiver alltoall tensor is not 3D at layer "
                 << receiver_round.layer_id() << ", tensor index "
                 << receiver_round.tensor_idx();
      session_manager_->destroy_session_context_async(session_id, true);
      return false;
    }
    uint64_t expert_unit_bytes = get_expert_unit_bytes(
        tensor, receiver_round.layer_id(), receiver_round.tensor_idx());
    if (expert_unit_bytes == 0) {
      session_manager_->destroy_session_context_async(session_id, true);
      return false;
    }

    std::vector<uint64_t> send_counts(active_n_ranks, 0);
    std::vector<uint64_t> sdispls(active_n_ranks, 0);
    std::vector<uint64_t> recv_counts(active_n_ranks, 0);
    std::vector<uint64_t> rdispls(active_n_ranks, 0);
    for (size_t source_idx = 0; source_idx < source_requests.size();
         ++source_idx) {
      const auto& req = source_requests[source_idx];
      if (round_idx >= static_cast<size_t>(req.alltoall_rounds_size())) {
        LOG(ERROR) << "Alltoall round size mismatch for source "
                   << source_addrs[source_idx] << ", round_idx=" << round_idx;
        session_manager_->destroy_session_context_async(session_id, true);
        return false;
      }
      const auto& source_round =
          req.alltoall_rounds(static_cast<int>(round_idx));
      int32_t local_start = source_round.local_expert_start();
      int32_t local_count = source_round.local_expert_count();
      if (local_count == 0) {
        continue;
      }
      if (local_start < 0 || local_count < 0 ||
          local_start + local_count > tensor.size(0)) {
        LOG(ERROR) << "Invalid receiver alltoall range at layer "
                   << source_round.layer_id() << ", tensor index "
                   << source_round.tensor_idx()
                   << ", local_start=" << local_start
                   << ", local_count=" << local_count
                   << ", local_expert_num=" << tensor.size(0);
        session_manager_->destroy_session_context_async(session_id, true);
        return false;
      }
      uint32_t source_rank = static_cast<uint32_t>(source_idx + 1);
      recv_counts[source_rank] = expert_unit_bytes * local_count;
      rdispls[source_rank] = expert_unit_bytes * local_start;
      total_bytes += recv_counts[source_rank];
    }

    auto ret = HcclAlltoAllV(default_send_buf,
                             send_counts.data(),
                             sdispls.data(),
                             HCCL_DATA_TYPE_UINT8,
                             tensor.data_ptr(),
                             recv_counts.data(),
                             rdispls.data(),
                             HCCL_DATA_TYPE_UINT8,
                             session_ctx->hccl_comm,
                             session_ctx->stream);
    if (ret != HCCL_SUCCESS) {
      LOG(ERROR) << "Receiver HcclAlltoAllV failed at round " << round_idx
                 << ", ret=" << ret;
      session_manager_->destroy_session_context_async(session_id, true);
      return false;
    }
  }

  auto sync_ret = aclrtSynchronizeStream(session_ctx->stream);
  if (sync_ret != ACL_SUCCESS) {
    LOG(ERROR) << "Receiver aclrtSynchronizeStream failed for alltoallv, ret="
               << sync_ret;
    session_manager_->destroy_session_context_async(session_id, true);
    return false;
  }

  absl::Time end_time = absl::Now();
  double duration_s = absl::ToDoubleSeconds(end_time - start_time);
  double duration_ms = absl::ToDoubleMilliseconds(end_time - start_time);
  double total_gb =
      static_cast<double>(total_bytes) / (1024.0 * 1024.0 * 1024.0);
  double bandwidth_gb_s = duration_s > 0 ? total_gb / duration_s : 0.0;

  LOG(INFO) << "[Receiver Thread] Alltoallv transfer (sources: "
            << source_addrs.size() << ", rounds=" << receiver_rounds.size()
            << ", session_id=" << session_id << "): " << std::fixed
            << std::setprecision(2) << total_gb << " GB, "
            << "Time: " << duration_ms << " ms, "
            << "Bandwidth: " << bandwidth_gb_s << " GB/s";
  session_manager_->destroy_session_context_async(session_id, true);
  return true;
}

bool WeightTransferReceiverEngine::prepare_session_rpc_endpoint(
    const std::string& remote_addr,
    const std::string& session_id) {
  auto session_ctx = session_manager_->create_session_context(
      session_id, xllm::proto::COMM_MODE_P2P);
  if (session_ctx == nullptr) {
    return false;
  }
  std::lock_guard<std::mutex> session_lock(session_ctx->operation_mutex);
  if (!init_weight_transfer_stub(
          remote_addr, &session_ctx->channel, &session_ctx->stub)) {
    LOG(ERROR) << "Failed to prepare rpc endpoint for session " << session_id
               << ", remote_addr=" << remote_addr;
    return false;
  }
  session_ctx->comm_mode = xllm::proto::COMM_MODE_P2P;
  session_ctx->is_comm_initialized.store(false);
  return true;
}

bool WeightTransferReceiverEngine::init_session_p2p_comm(
    const std::string& remote_addr,
    const std::string& session_id) {
  auto session_ctx = session_manager_->get_session_context(session_id);
  if (session_ctx == nullptr) {
    LOG(ERROR) << "Session not found for p2p comm init: " << session_id;
    return false;
  }

  xllm::proto::WeightTransferService_Stub* target_stub = nullptr;
  {
    std::lock_guard<std::mutex> session_lock(session_ctx->operation_mutex);
    target_stub = session_ctx->stub.get();
  }
  if (target_stub == nullptr) {
    LOG(ERROR) << "Session stub is null for p2p comm init: " << session_id;
    return false;
  }

  aclrtSetDevice(device_id_);
  HcclRootInfo root_info;
  auto ret = HcclGetRootInfo(&root_info);
  if (ret != HCCL_SUCCESS) {
    LOG(ERROR) << "HcclGetRootInfo failed for session " << session_id;
    return false;
  }

  xllm::proto::InitCommRequest req;
  req.set_addr(local_addr_);
  req.set_root_info(&root_info, sizeof(HcclRootInfo));
  req.set_n_ranks(2);
  req.set_rank(1);
  req.set_session_id(session_id);
  req.set_comm_mode(xllm::proto::COMM_MODE_P2P);
  if (!call_init_comm_with_retry(target_stub, req, remote_addr)) {
    LOG(ERROR) << "InitComm failed for session " << session_id
               << ", remote_addr=" << remote_addr;
    return false;
  }

  HcclComm session_comm = nullptr;
  ret = HcclCommInitRootInfo(2, &root_info, 0, &session_comm);
  if (ret != HCCL_SUCCESS) {
    LOG(ERROR) << "HcclCommInitRootInfo failed for session " << session_id
               << ", ret=" << ret;
    return false;
  }
  {
    std::lock_guard<std::mutex> session_lock(session_ctx->operation_mutex);
    session_ctx->hccl_comm = session_comm;
    session_ctx->n_ranks = 2;
    session_ctx->rank = 0;
    session_ctx->comm_mode = xllm::proto::COMM_MODE_P2P;
    session_ctx->is_comm_initialized.store(true);
  }
  return true;
}

bool WeightTransferReceiverEngine::prepare_local_tensors_from_meta(
    const std::string& session_id,
    const std::vector<int32_t>& layer_ids,
    const std::vector<std::vector<at::Tensor>*>& local_tensors_ptrs,
    bool enable_layer_storage_allocation,
    std::unordered_map<int32_t, ReceiverLayerStorage>* layer_storages,
    MetaAllocateStats* prepare_stats) {
  if (session_id.empty()) {
    LOG(ERROR)
        << "Session id should not be empty when preparing local tensors.";
    return false;
  }

  auto session_ctx = session_manager_->get_session_context(session_id);
  if (session_ctx == nullptr) {
    LOG(ERROR) << "Session not found for metadata preparation: " << session_id;
    return false;
  }

  xllm::proto::WeightTransferService_Stub* target_stub = nullptr;
  {
    std::lock_guard<std::mutex> session_lock(session_ctx->operation_mutex);
    target_stub = session_ctx->stub.get();
  }
  if (target_stub == nullptr) {
    LOG(ERROR) << "Session stub is null for metadata preparation. session_id="
               << session_id;
    return false;
  }

  MetaAllocateStats stats;
  if (!fetch_weights_meta_and_allocate_tensors(
          target_stub,
          layer_ids,
          local_tensors_ptrs,
          enable_layer_storage_allocation,
          layer_storages,
          "prepare_local_tensors_from_meta",
          &stats)) {
    return false;
  }
  if (prepare_stats != nullptr) {
    *prepare_stats = stats;
  }
  return true;
}

bool WeightTransferReceiverEngine::fetch_weights_meta_and_allocate_tensors(
    xllm::proto::WeightTransferService_Stub* target_stub,
    const std::vector<int32_t>& layer_ids,
    const std::vector<std::vector<at::Tensor>*>& local_tensors_ptrs,
    bool enable_layer_storage_allocation,
    std::unordered_map<int32_t, ReceiverLayerStorage>* layer_storages,
    const std::string& stage_name,
    MetaAllocateStats* stats) {
  if (target_stub == nullptr) {
    LOG(ERROR) << stage_name << ": Weight transfer stub is null.";
    return false;
  }
  if (layer_ids.size() != local_tensors_ptrs.size()) {
    LOG(ERROR) << stage_name
               << ": Layer ids size and local tensors ptr size mismatch: "
               << layer_ids.size() << " vs " << local_tensors_ptrs.size();
    return false;
  }
  for (auto* tensors_ptr : local_tensors_ptrs) {
    if (tensors_ptr == nullptr) {
      LOG(ERROR) << stage_name
                 << ": Local tensor pointer should not be nullptr.";
      return false;
    }
  }

  MetaAllocateStats local_stats;
  const absl::Time meta_alloc_start = absl::Now();
  brpc::Controller cntl_meta;
  xllm::proto::GetWeightsMetaRequest req_meta;
  xllm::proto::GetWeightsMetaResponse resp_meta;
  for (int32_t id : layer_ids) {
    req_meta.add_layer_ids(id);
  }

  const absl::Time meta_rpc_start = absl::Now();
  target_stub->GetWeightsMeta(&cntl_meta, &req_meta, &resp_meta, nullptr);
  local_stats.rpc_ms = elapsed_ms_since(meta_rpc_start);
  if (cntl_meta.Failed()) {
    LOG(ERROR) << stage_name
               << ": GetWeightsMeta failed: " << cntl_meta.ErrorText();
    return false;
  }
  if (resp_meta.layer_metas_size() != static_cast<int>(layer_ids.size())) {
    LOG(ERROR) << stage_name << ": GetWeightsMeta size mismatch. expected="
               << layer_ids.size()
               << ", actual=" << resp_meta.layer_metas_size();
    return false;
  }

  const int32_t num_layers = context_.get_model_args().n_layers();
  std::unordered_set<int32_t> available_storage_layers;
  available_storage_layers.reserve(static_cast<size_t>(num_layers));
  bool can_allocate_layer_storage = FLAGS_enable_manual_loader &&
                                    enable_layer_storage_allocation &&
                                    layer_storages != nullptr;
  if (can_allocate_layer_storage) {
    for (int i = 0; i < resp_meta.layer_metas_size(); ++i) {
      const auto& layer_meta = resp_meta.layer_metas(i);
      const int32_t layer_id = layer_meta.layer_id();
      if (layer_id < 0) {
        continue;
      }
      if (layer_id >= num_layers || !is_valid_layer_storage_meta(layer_meta)) {
        can_allocate_layer_storage = false;
        LOG(INFO) << "Disable contiguous layer storage allocation. layer_id="
                  << layer_id << ", num_layers=" << num_layers
                  << ", has_storage=" << layer_meta.has_contiguous_storage()
                  << ", storage_size=" << layer_meta.storage_size()
                  << ", tensor_metas=" << layer_meta.metas_size()
                  << ", slice_metas=" << layer_meta.slice_metas_size();
        break;
      }
      available_storage_layers.insert(layer_id);
    }
    if (can_allocate_layer_storage &&
        available_storage_layers.size() != static_cast<size_t>(num_layers)) {
      can_allocate_layer_storage = false;
      LOG(INFO) << "Disable contiguous layer storage allocation because not "
                   "all decoder layers have storage metadata. expected="
                << num_layers << ", actual=" << available_storage_layers.size();
    }
  }
  if (!can_allocate_layer_storage && layer_storages != nullptr) {
    release_receiver_layer_storage_map(layer_storages);
  }

  aclrtSetDevice(device_id_);
  const absl::Time tensor_alloc_start = absl::Now();
  bool all_layer_storage_available = false;
  if (can_allocate_layer_storage) {
    auto decoder_loaders = model_->get_decoder_loaders();
    bool storage_alloc_ok =
        decoder_loaders.size() >= static_cast<size_t>(num_layers);
    if (!storage_alloc_ok) {
      LOG(INFO) << "Disable loader-native receiver storage because decoder "
                   "loaders are incomplete. expected="
                << num_layers << ", actual=" << decoder_loaders.size();
    }
    for (int i = 0; i < resp_meta.layer_metas_size(); ++i) {
      const auto& layer_meta = resp_meta.layer_metas(i);
      if (layer_meta.layer_id() < 0) {
        continue;
      }
      if (!storage_alloc_ok) {
        break;
      }
      layer::BaseLoader* loader = decoder_loaders[layer_meta.layer_id()];
      if (loader == nullptr || loader->mode() != layer::LoadMode::kManual ||
          loader->uses_rolling_buffer()) {
        storage_alloc_ok = false;
        LOG(INFO) << "Disable loader-native receiver storage. layer_id="
                  << layer_meta.layer_id()
                  << ", has_loader=" << (loader != nullptr)
                  << ", uses_rolling_buffer="
                  << (loader != nullptr && loader->uses_rolling_buffer());
        break;
      }
    }

    if (storage_alloc_ok) {
      for (int i = 0; i < resp_meta.layer_metas_size(); ++i) {
        const auto& layer_meta = resp_meta.layer_metas(i);
        const int32_t layer_id = layer_meta.layer_id();
        if (layer_id < 0) {
          continue;
        }
        layer::BaseLoader* loader = decoder_loaders[layer_id];
        std::vector<layer::BaseLoader::DeviceWeightSliceSpec> slice_specs;
        slice_specs.reserve(layer_meta.slice_metas_size());
        for (const auto& slice_meta : layer_meta.slice_metas()) {
          layer::BaseLoader::DeviceWeightSliceSpec spec;
          spec.offset = slice_meta.offset();
          spec.bytes = slice_meta.bytes();
          spec.dtype = static_cast<at::ScalarType>(slice_meta.dtype());
          spec.acl_format = static_cast<int>(slice_meta.npu_format());
          spec.sizes.reserve(slice_meta.shape_size());
          for (int64_t dim : slice_meta.shape()) {
            spec.sizes.push_back(dim);
          }
          slice_specs.push_back(std::move(spec));
        }
        const absl::Time storage_alloc_start = absl::Now();
        loader->prepare_device_storage_from_slices(layer_meta.storage_size(),
                                                   slice_specs,
                                                   /*initialize_views=*/false);
        local_stats.storage_alloc_ms += elapsed_ms_since(storage_alloc_start);

        ReceiverLayerStorage storage;
        storage.available = loader->get_device_storage() != nullptr;
        storage.base_ptr = loader->get_device_storage();
        storage.storage_size = layer_meta.storage_size();
        storage.payload_nbytes = sum_layer_storage_payload_nbytes(layer_meta);
        storage.views_initialized = false;
        storage.loader = loader;
        if (!storage.available || storage.base_ptr == nullptr ||
            storage.storage_size == 0 || storage.payload_nbytes == 0) {
          LOG(WARNING) << "Failed to setup loader-native receiver storage. "
                       << "layer_id=" << layer_id;
          storage_alloc_ok = false;
          break;
        }
        (*layer_storages)[layer_id] = std::move(storage);
      }
    }

    if (storage_alloc_ok) {
      for (int i = 0; i < resp_meta.layer_metas_size(); ++i) {
        const auto& layer_meta = resp_meta.layer_metas(i);
        if (layer_meta.layer_id() >= 0) {
          continue;
        }
        allocate_tensors_from_tensor_meta(
            layer_meta, device_id_, local_tensors_ptrs[i]);
      }
    }

    if (storage_alloc_ok &&
        layer_storages->size() == static_cast<size_t>(num_layers)) {
      all_layer_storage_available = true;
    } else {
      release_receiver_layer_storage_map(layer_storages);
    }
  }

  if (!all_layer_storage_available) {
    for (int i = 0; i < resp_meta.layer_metas_size(); ++i) {
      allocate_tensors_from_tensor_meta(
          resp_meta.layer_metas(i), device_id_, local_tensors_ptrs[i]);
    }
  }
  local_stats.tensor_alloc_ms = elapsed_ms_since(tensor_alloc_start);
  local_stats.all_layer_storage_available = all_layer_storage_available;
  local_stats.total_ms = elapsed_ms_since(meta_alloc_start);
  if (stats != nullptr) {
    *stats = local_stats;
  }
  return true;
}

std::vector<int32_t> WeightTransferReceiverEngine::build_pull_layer_ids()
    const {
  const int32_t num_layers = context_.get_model_args().n_layers();
  std::vector<int32_t> layer_ids = {-1};
  layer_ids.reserve(static_cast<size_t>(num_layers) + 1);
  for (int32_t layer_id = 0; layer_id < num_layers; ++layer_id) {
    layer_ids.push_back(layer_id);
  }
  return layer_ids;
}

void WeightTransferReceiverEngine::build_pull_tensor_ptrs(
    std::vector<at::Tensor>* global_tensors,
    std::vector<std::vector<at::Tensor>*>* local_tensors_ptrs) {
  CHECK(global_tensors != nullptr);
  CHECK(local_tensors_ptrs != nullptr);
  local_tensors_ptrs->clear();
  const int32_t num_layers = context_.get_model_args().n_layers();
  local_tensors_ptrs->reserve(static_cast<size_t>(num_layers) + 1);
  local_tensors_ptrs->push_back(global_tensors);
  for (int32_t layer_id = 0; layer_id < num_layers; ++layer_id) {
    local_tensors_ptrs->push_back(&model_->get_decoder_layer_weight(layer_id));
  }
}

bool WeightTransferReceiverEngine::assign_global_tensors_after_pull(
    const std::vector<at::Tensor>& global_tensors) {
  if (global_tensors.size() < 3) {
    LOG(ERROR) << "Global tensors are incomplete after pull. size="
               << global_tensors.size();
    return false;
  }
  model_->get_word_embedding_weight()[0] = global_tensors[0];
  model_->get_norm_weight()[0] = global_tensors[1];
  model_->get_lm_head_weight()[0] = global_tensors[2];
  return true;
}

void WeightTransferReceiverEngine::release_layer_storage_views() {
  release_receiver_layer_storage_map(&receiver_layer_storages_);
}

std::unordered_map<int32_t, std::vector<int32_t>>
WeightTransferReceiverEngine::build_non_expert_only_plan(
    const std::vector<int32_t>& layer_ids) const {
  std::unordered_map<int32_t, std::vector<int32_t>> non_expert_only_plan;
  for (int32_t layer_id : layer_ids) {
    non_expert_only_plan.emplace(layer_id, std::vector<int32_t>{});
  }
  return non_expert_only_plan;
}
ModelPullPrepareStageStatus
WeightTransferReceiverEngine::run_model_pull_prepare_stage(
    const std::string& remote_addr,
    const std::string& base_session_id,
    const std::vector<int32_t>& layer_ids,
    const std::vector<std::vector<at::Tensor>*>& local_tensors_ptrs,
    bool enable_layer_storage_allocation,
    std::unordered_map<int32_t, ReceiverLayerStorage>* layer_storages,
    bool has_expert_prepare,
    const std::function<ModelPullAsyncStageStatus()>& expert_prepare_fn) {
  ModelPullPrepareStageStatus prepare_status;
  const absl::Time prepare_start = absl::Now();

  const absl::Time base_endpoint_start = absl::Now();
  if (!prepare_session_rpc_endpoint(remote_addr, base_session_id)) {
    prepare_status.ok = false;
    prepare_status.failed_stage = "connect_base_rpc";
    prepare_status.prepare_wall_ms = elapsed_ms_since(prepare_start);
    return prepare_status;
  }
  prepare_status.base_endpoint_ms = elapsed_ms_since(base_endpoint_start);

  auto base_comm_future =
      std::async(std::launch::async, [&, remote_addr, base_session_id]() {
        ModelPullAsyncStageStatus stage_status;
        const absl::Time stage_start = absl::Now();
        if (!init_session_p2p_comm(remote_addr, base_session_id)) {
          stage_status.ok = false;
          stage_status.failed_stage = "connect_base";
        }
        stage_status.elapsed_ms = elapsed_ms_since(stage_start);
        return stage_status;
      });

  std::future<ModelPullAsyncStageStatus> expert_comm_future;
  if (has_expert_prepare) {
    expert_comm_future = std::async(std::launch::async, [expert_prepare_fn]() {
      return expert_prepare_fn();
    });
  }

  MetaAllocateStats prepare_stats;
  bool meta_ok =
      prepare_local_tensors_from_meta(base_session_id,
                                      layer_ids,
                                      local_tensors_ptrs,
                                      enable_layer_storage_allocation,
                                      layer_storages,
                                      &prepare_stats);
  ModelPullAsyncStageStatus base_comm_status = base_comm_future.get();
  ModelPullAsyncStageStatus expert_comm_status;
  if (has_expert_prepare) {
    expert_comm_status = expert_comm_future.get();
  }
  prepare_status.base_comm_ms = base_comm_status.elapsed_ms;
  prepare_status.expert_comm_ms = expert_comm_status.elapsed_ms;
  prepare_status.meta_alloc_ms = prepare_stats.total_ms;
  prepare_status.meta_rpc_ms = prepare_stats.rpc_ms;
  prepare_status.tensor_alloc_ms = prepare_stats.tensor_alloc_ms;
  prepare_status.storage_alloc_ms = prepare_stats.storage_alloc_ms;
  prepare_status.storage_view_init_ms = prepare_stats.storage_view_init_ms;
  prepare_status.prepare_wall_ms = elapsed_ms_since(prepare_start);

  if (!meta_ok) {
    prepare_status.ok = false;
    prepare_status.failed_stage = "prepare_local_tensors";
  }
  if (!base_comm_status.ok) {
    prepare_status.ok = false;
    if (prepare_status.failed_stage.empty()) {
      prepare_status.failed_stage = base_comm_status.failed_stage;
    }
  }
  if (has_expert_prepare && !expert_comm_status.ok) {
    prepare_status.ok = false;
    if (prepare_status.failed_stage.empty()) {
      prepare_status.failed_stage = expert_comm_status.failed_stage;
    }
  }
  return prepare_status;
}

void WeightTransferReceiverEngine::log_model_pull_prepare_timing(
    const std::string& remote_addr,
    size_t expert_sources,
    const ModelPullPrepareStageStatus& prepare_status) const {
  const double overlap_saved_ms = std::max(
      0.0,
      prepare_status.base_comm_ms + prepare_status.expert_comm_ms +
          prepare_status.meta_alloc_ms - prepare_status.prepare_wall_ms);
  LOG(INFO) << "[WeightPullPrepareTiming] remote_addr=" << remote_addr
            << ", expert_sources=" << expert_sources
            << ", base_comm_ms=" << std::fixed << std::setprecision(2)
            << prepare_status.base_comm_ms
            << ", expert_comm_ms=" << prepare_status.expert_comm_ms
            << ", meta_alloc_ms=" << prepare_status.meta_alloc_ms
            << ", meta_rpc_ms=" << prepare_status.meta_rpc_ms
            << ", tensor_alloc_ms=" << prepare_status.tensor_alloc_ms
            << ", storage_alloc_ms=" << prepare_status.storage_alloc_ms
            << ", storage_view_init_ms=" << prepare_status.storage_view_init_ms
            << ", prepare_wall_ms=" << prepare_status.prepare_wall_ms
            << ", overlap_saved_ms=" << overlap_saved_ms;
}

ModelPullStageStatus
WeightTransferReceiverEngine::run_prepared_expert_transfer_serial_stage(
    bool has_two_stage_expert,
    bool use_alltoallv_expert_transfer,
    const std::vector<std::string>& alltoall_source_addrs,
    const std::vector<xllm::proto::AlltoAllRoundDesc>& alltoall_receiver_rounds,
    const std::vector<xllm::proto::TriggerWeightsSendRequest>&
        alltoall_source_requests,
    const std::string& expert_session_id,
    const std::vector<int32_t>& single_source_layer_ids,
    const std::vector<std::vector<at::Tensor>*>& single_source_tensors_ptrs,
    const std::unordered_map<int32_t, std::vector<int32_t>>&
        single_source_layer_expert_ids,
    double* transfer_ms) {
  if (!has_two_stage_expert) {
    return ModelPullStageStatus{};
  }
  const absl::Time transfer_start = absl::Now();
  if (use_alltoallv_expert_transfer) {
    LOG(INFO) << "Pulling expert weights by alltoallv. sources="
              << alltoall_source_addrs.size()
              << ", rounds=" << alltoall_receiver_rounds.size()
              << ", session_id=" << expert_session_id;
    if (!pull_expert_weights_alltoallv(alltoall_source_addrs,
                                       alltoall_receiver_rounds,
                                       alltoall_source_requests,
                                       expert_session_id,
                                       true)) {
      LOG(ERROR) << "Failed to pull expert weights by alltoallv.";
      return ModelPullStageStatus{false, "transfer_expert_alltoall"};
    }
    if (transfer_ms != nullptr) {
      *transfer_ms = elapsed_ms_since(transfer_start);
    }
    return ModelPullStageStatus{};
  }

  if (!pull_weight_internal(expert_session_id,
                            single_source_layer_ids,
                            single_source_tensors_ptrs,
                            receiver_layer_storages_,
                            single_source_layer_expert_ids,
                            false,
                            false,
                            false)) {
    LOG(ERROR) << "Failed to pull expert weights by selective p2p. session_id="
               << expert_session_id;
    return ModelPullStageStatus{false, "transfer_expert"};
  }
  if (transfer_ms != nullptr) {
    *transfer_ms = elapsed_ms_since(transfer_start);
  }
  session_manager_->destroy_session_context_async(expert_session_id, true);
  return ModelPullStageStatus{};
}

bool WeightTransferReceiverEngine::pull_model_from_instance(
    const std::string& remote_addr,
    const RankExpertTransferPlanData& rank_expert_transfer_plan) {
  const absl::Time overall_start_time = absl::Now();
  WeightPullTimingStats timing_stats;
  release_layer_storage_views();
  use_layer_storage_transfer_ = false;

  auto finalize_with_timing = [&](bool success,
                                  const std::string& failed_stage) -> bool {
    timing_stats.success = success;
    timing_stats.failed_stage = success ? "" : failed_stage;
    timing_stats.total_ms = elapsed_ms_since(overall_start_time);
    timing_stats.other_ms =
        std::max(0.0,
                 timing_stats.total_ms - timing_stats.comm_create_ms -
                     timing_stats.weight_transfer_ms);
    log_weight_pull_timing(remote_addr, timing_stats);
    return success;
  };

  int32_t num_layers = context_.get_model_args().n_layers();
  std::vector<int32_t> layer_ids = build_pull_layer_ids();

  auto source_tasks = planner_->build_source_expert_transfer_tasks(
      rank_expert_transfer_plan, context_.get_model_args());
  size_t total_source_experts = 0;
  for (const auto& task : source_tasks) {
    total_source_experts +=
        planner_->count_task_expert_ids(task.layer_expert_ids);
  }
  timing_stats.expert_sources = source_tasks.size();
  LOG(INFO) << "Expert transfer tasks prepared: sources=" << source_tasks.size()
            << ", expert_ids=" << total_source_experts;

  std::vector<at::Tensor> global_tensors;
  std::vector<std::vector<at::Tensor>*> local_tensors_ptrs;
  std::unordered_map<int32_t, ReceiverLayerStorage> layer_storages;
  build_pull_tensor_ptrs(&global_tensors, &local_tensors_ptrs);

  const bool use_single_phase_pull =
      source_tasks.size() == 1 &&
      source_tasks.front().source_addr == remote_addr;
  const bool has_two_stage_expert =
      !source_tasks.empty() && !use_single_phase_pull;
  const bool allow_layer_storage_transfer =
      FLAGS_enable_manual_loader &&
      (source_tasks.empty() || use_single_phase_pull) && !has_two_stage_expert;
  const bool enable_parallel_pull =
      FLAGS_enable_parallel_weight_pull && has_two_stage_expert;
  timing_stats.parallel_mode = enable_parallel_pull;
  if (use_single_phase_pull) {
    LOG(INFO) << "Use single-phase weight pull because expert source matches "
              << "base source. remote_addr=" << remote_addr;
  }

  std::string expert_session_id;
  std::vector<SourceExpertTransferTask> sorted_source_tasks_for_alltoall;
  std::vector<std::string> alltoall_source_addrs;
  std::vector<xllm::proto::AlltoAllRoundDesc> alltoall_receiver_rounds;
  std::vector<xllm::proto::TriggerWeightsSendRequest> alltoall_source_requests;
  std::vector<int32_t> single_source_layer_ids;
  std::vector<std::vector<at::Tensor>*> single_source_tensors_ptrs;
  LayerExpertIdsMap single_source_layer_expert_ids;
  bool use_alltoallv_expert_transfer = false;

  auto prepare_expert_comm_only = [&]() -> ModelPullStageStatus {
    if (!has_two_stage_expert) {
      return ModelPullStageStatus{};
    }

    expert_session_id.clear();
    sorted_source_tasks_for_alltoall.clear();
    alltoall_source_addrs.clear();
    alltoall_receiver_rounds.clear();
    alltoall_source_requests.clear();
    single_source_layer_ids.clear();
    single_source_tensors_ptrs.clear();
    single_source_layer_expert_ids.clear();
    use_alltoallv_expert_transfer = false;

    if (source_tasks.size() > 1) {
      sorted_source_tasks_for_alltoall = source_tasks;
      std::sort(sorted_source_tasks_for_alltoall.begin(),
                sorted_source_tasks_for_alltoall.end(),
                [&](const SourceExpertTransferTask& lhs,
                    const SourceExpertTransferTask& rhs) {
                  return planner_->compare_source_task_by_addr(lhs, rhs);
                });
      alltoall_source_addrs.reserve(sorted_source_tasks_for_alltoall.size());
      for (const auto& source_task : sorted_source_tasks_for_alltoall) {
        alltoall_source_addrs.push_back(source_task.source_addr);
      }
      expert_session_id = generate_session_id();
      if (!init_collective_comm_as_receiver(
              alltoall_source_addrs, 0, expert_session_id)) {
        LOG(ERROR) << "Failed to initialize alltoall session on receiver. "
                   << "session_id=" << expert_session_id;
        return ModelPullStageStatus{false, "init_alltoall"};
      }
      use_alltoallv_expert_transfer = true;
      LOG(INFO) << "Prepared expert alltoall communication. sources="
                << alltoall_source_addrs.size()
                << ", session_id=" << expert_session_id;
      return ModelPullStageStatus{};
    }

    const auto& source_task = source_tasks.front();
    single_source_layer_ids =
        planner_->collect_sorted_layer_ids(source_task.layer_expert_ids);
    single_source_tensors_ptrs.reserve(single_source_layer_ids.size());
    for (int32_t layer_id : single_source_layer_ids) {
      if (layer_id < 0 || layer_id >= num_layers) {
        LOG(ERROR) << "Invalid layer id " << layer_id
                   << " in expert transfer plan for source "
                   << source_task.source_addr;
        return ModelPullStageStatus{false, "validate_expert_layers"};
      }
      single_source_tensors_ptrs.push_back(
          &model_->get_decoder_layer_weight(layer_id));
    }
    single_source_layer_expert_ids = source_task.layer_expert_ids;
    expert_session_id = generate_session_id();
    if (!prepare_session_rpc_endpoint(source_task.source_addr,
                                      expert_session_id)) {
      LOG(ERROR) << "Failed to prepare expert source rpc endpoint "
                 << source_task.source_addr
                 << ", session_id=" << expert_session_id;
      return ModelPullStageStatus{false, "prepare_expert_source_rpc"};
    }
    if (!init_session_p2p_comm(source_task.source_addr, expert_session_id)) {
      LOG(ERROR) << "Failed to connect to expert source "
                 << source_task.source_addr
                 << ", session_id=" << expert_session_id;
      return ModelPullStageStatus{false, "connect_expert_source"};
    }
    LOG(INFO) << "Prepared expert selective transfer. source="
              << source_task.source_addr
              << ", layers=" << single_source_layer_ids.size()
              << ", expert_ids="
              << planner_->count_task_expert_ids(single_source_layer_expert_ids)
              << ", session_id=" << expert_session_id;
    return ModelPullStageStatus{};
  };

  auto finalize_expert_transfer_after_tensor_ready =
      [&]() -> ModelPullStageStatus {
    if (!has_two_stage_expert || !use_alltoallv_expert_transfer) {
      return ModelPullStageStatus{};
    }
    std::vector<AlltoallRoundPlan> round_plans;
    if (!planner_->build_alltoall_round_plans(model_,
                                              context_.get_parallel_args(),
                                              sorted_source_tasks_for_alltoall,
                                              &round_plans)) {
      LOG(ERROR) << "Failed to build alltoall round plans.";
      return ModelPullStageStatus{false, "build_alltoall_rounds"};
    }
    planner_->build_alltoall_requests(round_plans,
                                      sorted_source_tasks_for_alltoall,
                                      /*receiver_rank=*/0,
                                      expert_session_id,
                                      &alltoall_receiver_rounds,
                                      &alltoall_source_requests);
    LOG(INFO) << "Prepared expert alltoall transfer. sources="
              << alltoall_source_addrs.size()
              << ", rounds=" << alltoall_receiver_rounds.size()
              << ", session_id=" << expert_session_id;
    return ModelPullStageStatus{};
  };

  const std::string base_session_id = generate_session_id();
  ModelPullPrepareStageStatus prepare_status = run_model_pull_prepare_stage(
      remote_addr,
      base_session_id,
      layer_ids,
      local_tensors_ptrs,
      allow_layer_storage_transfer &&
          FLAGS_enable_layer_storage_weight_transfer,
      &layer_storages,
      has_two_stage_expert,
      [&]() -> ModelPullAsyncStageStatus {
        ModelPullAsyncStageStatus stage_status;
        const absl::Time stage_start = absl::Now();
        ModelPullStageStatus expert_status = prepare_expert_comm_only();
        stage_status.ok = expert_status.ok;
        stage_status.failed_stage = expert_status.failed_stage;
        stage_status.elapsed_ms = elapsed_ms_since(stage_start);
        return stage_status;
      });
  log_model_pull_prepare_timing(
      remote_addr, source_tasks.size(), prepare_status);
  timing_stats.comm_create_ms += prepare_status.base_endpoint_ms +
                                 prepare_status.base_comm_ms +
                                 prepare_status.expert_comm_ms;
  if (!prepare_status.ok) {
    receiver_layer_storages_ = std::move(layer_storages);
    session_manager_->destroy_session_context_async(base_session_id, true);
    if (!expert_session_id.empty()) {
      session_manager_->destroy_session_context_async(expert_session_id, true);
    }
    const std::string failed_stage = prepare_status.failed_stage.empty()
                                         ? "prepare_stage"
                                         : prepare_status.failed_stage;
    return finalize_with_timing(false, failed_stage);
  }

  ModelPullStageStatus finalize_expert_status =
      finalize_expert_transfer_after_tensor_ready();
  if (!finalize_expert_status.ok) {
    receiver_layer_storages_ = std::move(layer_storages);
    session_manager_->destroy_session_context_async(base_session_id, true);
    if (!expert_session_id.empty()) {
      session_manager_->destroy_session_context_async(expert_session_id, true);
    }
    return finalize_with_timing(false, finalize_expert_status.failed_stage);
  }

  const bool can_use_layer_storage_transfer =
      allow_layer_storage_transfer &&
      FLAGS_enable_layer_storage_weight_transfer &&
      layer_storages.size() == static_cast<size_t>(num_layers);
  if (can_use_layer_storage_transfer) {
    receiver_layer_storages_ = std::move(layer_storages);
    use_layer_storage_transfer_ = true;
    LOG(INFO) << "Use contiguous layer storage transfer for full single-source "
                 "pull. decoder_layers="
              << receiver_layer_storages_.size();
  } else {
    receiver_layer_storages_ = std::move(layer_storages);
    use_layer_storage_transfer_ = false;
  }

  if (source_tasks.empty() || use_single_phase_pull) {
    const absl::Time transfer_start = absl::Now();
    if (!pull_weight_internal(base_session_id,
                              layer_ids,
                              local_tensors_ptrs,
                              receiver_layer_storages_,
                              LayerExpertIdsMap{},
                              true,
                              true,
                              false)) {
      LOG(ERROR) << "Failed to pull full weights from base source "
                 << remote_addr << ", session_id=" << base_session_id;
      session_manager_->destroy_session_context_async(base_session_id, true);
      return finalize_with_timing(false,
                                  use_single_phase_pull
                                      ? "transfer_single_phase_weights"
                                      : "transfer_full_weights");
    }
    timing_stats.weight_transfer_ms += elapsed_ms_since(transfer_start);
    session_manager_->destroy_session_context_async(base_session_id, true);
    if (!assign_global_tensors_after_pull(global_tensors)) {
      return finalize_with_timing(false, "validate_global_tensors");
    }
    return finalize_with_timing(true, "");
  }

  auto non_expert_only_plan = build_non_expert_only_plan(layer_ids);

  if (!enable_parallel_pull) {
    const absl::Time transfer_start = absl::Now();
    if (!pull_weight_internal(base_session_id,
                              layer_ids,
                              local_tensors_ptrs,
                              receiver_layer_storages_,
                              non_expert_only_plan,
                              true,
                              false,
                              false)) {
      LOG(ERROR) << "Failed to pull non-expert weights from base source "
                 << remote_addr << ", session_id=" << base_session_id;
      session_manager_->destroy_session_context_async(base_session_id, true);
      if (!expert_session_id.empty()) {
        session_manager_->destroy_session_context_async(expert_session_id,
                                                        true);
      }
      return finalize_with_timing(false, "transfer_non_expert");
    }
    timing_stats.weight_transfer_ms += elapsed_ms_since(transfer_start);
    session_manager_->destroy_session_context_async(base_session_id, true);
    if (!assign_global_tensors_after_pull(global_tensors)) {
      return finalize_with_timing(false, "validate_global_tensors");
    }
    double expert_transfer_ms = 0.0;
    ModelPullStageStatus expert_transfer_status =
        run_prepared_expert_transfer_serial_stage(
            has_two_stage_expert,
            use_alltoallv_expert_transfer,
            alltoall_source_addrs,
            alltoall_receiver_rounds,
            alltoall_source_requests,
            expert_session_id,
            single_source_layer_ids,
            single_source_tensors_ptrs,
            single_source_layer_expert_ids,
            &expert_transfer_ms);
    timing_stats.weight_transfer_ms += expert_transfer_ms;
    if (!expert_transfer_status.ok) {
      if (!expert_session_id.empty()) {
        session_manager_->destroy_session_context_async(expert_session_id,
                                                        true);
      }
      return finalize_with_timing(false, expert_transfer_status.failed_stage);
    }
    return finalize_with_timing(true, "");
  }

  LOG(INFO) << "Pull model in parallel mode. remote_addr=" << remote_addr
            << ", expert_sources=" << source_tasks.size();

  const absl::Time parallel_transfer_start = absl::Now();
  auto non_expert_future =
      std::async(std::launch::async, [&]() -> ModelPullAsyncTransferStatus {
        const absl::Time transfer_start = absl::Now();
        bool ok = pull_weight_internal(base_session_id,
                                       layer_ids,
                                       local_tensors_ptrs,
                                       receiver_layer_storages_,
                                       non_expert_only_plan,
                                       true,
                                       false,
                                       false);
        double transfer_ms = elapsed_ms_since(transfer_start);
        if (!ok) {
          LOG(ERROR) << "Failed to pull non-expert weights from base source "
                     << remote_addr << ", session_id=" << base_session_id;
        }
        session_manager_->destroy_session_context_async(base_session_id, true);
        return ModelPullAsyncTransferStatus{
            ok, transfer_ms, ok ? "" : "transfer_non_expert"};
      });

  auto expert_future =
      std::async(std::launch::async, [&]() -> ModelPullAsyncTransferStatus {
        const absl::Time transfer_start = absl::Now();
        if (use_alltoallv_expert_transfer) {
          bool ok = pull_expert_weights_alltoallv(alltoall_source_addrs,
                                                  alltoall_receiver_rounds,
                                                  alltoall_source_requests,
                                                  expert_session_id,
                                                  true);
          double transfer_ms = elapsed_ms_since(transfer_start);
          if (!ok) {
            LOG(ERROR)
                << "Failed to pull expert weights by alltoallv. session_id="
                << expert_session_id;
          }
          return ModelPullAsyncTransferStatus{
              ok, transfer_ms, ok ? "" : "transfer_expert_alltoall"};
        }

        bool ok = pull_weight_internal(expert_session_id,
                                       single_source_layer_ids,
                                       single_source_tensors_ptrs,
                                       receiver_layer_storages_,
                                       single_source_layer_expert_ids,
                                       false,
                                       false,
                                       false);
        double transfer_ms = elapsed_ms_since(transfer_start);
        if (!ok) {
          LOG(ERROR)
              << "Failed to pull expert weights by selective p2p. session_id="
              << expert_session_id;
        }
        session_manager_->destroy_session_context_async(expert_session_id,
                                                        true);
        return ModelPullAsyncTransferStatus{
            ok, transfer_ms, ok ? "" : "transfer_expert"};
      });

  ModelPullAsyncTransferStatus non_expert_status = non_expert_future.get();
  ModelPullAsyncTransferStatus expert_status = expert_future.get();
  const double parallel_transfer_wall_ms =
      elapsed_ms_since(parallel_transfer_start);
  timing_stats.weight_transfer_ms += parallel_transfer_wall_ms;
  LOG(INFO) << "[WeightPullTimingParallel] non_expert_ms="
            << non_expert_status.transfer_ms
            << ", expert_ms=" << expert_status.transfer_ms
            << ", wall_ms=" << parallel_transfer_wall_ms;
  if (!non_expert_status.ok || !expert_status.ok) {
    LOG(ERROR) << "Parallel pull failed. non_expert_ok=" << non_expert_status.ok
               << ", expert_ok=" << expert_status.ok;
    if (!expert_session_id.empty()) {
      session_manager_->destroy_session_context_async(expert_session_id, true);
    }
    if (!non_expert_status.ok) {
      return finalize_with_timing(false, non_expert_status.failed_stage);
    }
    return finalize_with_timing(false, expert_status.failed_stage);
  }

  if (!assign_global_tensors_after_pull(global_tensors)) {
    return finalize_with_timing(false, "validate_global_tensors");
  }
  return finalize_with_timing(true, "");
}

std::future<TriggerRpcResult>
WeightTransferReceiverEngine::launch_trigger_rpc_stage(
    const std::shared_ptr<CommSessionContext>& session_ctx,
    const std::vector<int32_t>& layer_ids,
    const std::unordered_map<int32_t, std::vector<int32_t>>&
        normalized_layer_expert_ids,
    bool include_non_expert,
    bool transfer_all_experts,
    bool use_layer_storage_transfer,
    const std::string& session_id) {
  auto trigger_promise = std::make_shared<std::promise<TriggerRpcResult>>();
  std::future<TriggerRpcResult> trigger_future = trigger_promise->get_future();
  const absl::Time trigger_task_enqueued_time = absl::Now();
  auto session_ctx_for_rpc = session_ctx;

  session_manager_->schedule_rpc_task([this,
                                       layer_ids,
                                       normalized_layer_expert_ids,
                                       include_non_expert,
                                       transfer_all_experts,
                                       use_layer_storage_transfer,
                                       session_id,
                                       session_ctx_for_rpc,
                                       trigger_task_enqueued_time,
                                       trigger_promise]() {
    TriggerRpcResult trigger_result;
    const absl::Time trigger_thread_start = absl::Now();
    trigger_result.queue_wait_ms = absl::ToDoubleMilliseconds(
        trigger_thread_start - trigger_task_enqueued_time);
    auto* stub_for_rpc = session_ctx_for_rpc == nullptr
                             ? nullptr
                             : session_ctx_for_rpc->stub.get();
    if (stub_for_rpc == nullptr) {
      LOG(ERROR) << "TriggerWeightsSend skipped due to null stub. session_id="
                 << session_id;
      trigger_promise->set_value(trigger_result);
      return;
    }

    brpc::Controller cntl_trig;
    xllm::proto::TriggerWeightsSendRequest req_trig;
    xllm::proto::TriggerWeightsSendResponse resp_trig;
    if (transfer_all_experts) {
      for (int32_t id : layer_ids) {
        req_trig.add_layer_ids(id);
      }
      req_trig.set_include_non_expert(include_non_expert);
      req_trig.set_transfer_all_experts(true);
    } else {
      fill_trigger_weights_send_request(layer_ids,
                                        normalized_layer_expert_ids,
                                        include_non_expert,
                                        &req_trig);
    }
    req_trig.set_session_id(session_id);
    req_trig.set_use_layer_storage_transfer(use_layer_storage_transfer);

    const absl::Time rpc_start = absl::Now();
    stub_for_rpc->TriggerWeightsSend(
        &cntl_trig, &req_trig, &resp_trig, nullptr);
    trigger_result.rpc_ms = elapsed_ms_since(rpc_start);
    if (cntl_trig.Failed() || !resp_trig.success()) {
      LOG(ERROR) << "TriggerWeightsSend failed: " << cntl_trig.ErrorText()
                 << ", session_id=" << session_id;
      trigger_promise->set_value(trigger_result);
      return;
    }
    trigger_result.ok = true;
    trigger_promise->set_value(trigger_result);
  });
  return trigger_future;
}

std::future<ReceiverTransferResult>
WeightTransferReceiverEngine::launch_receiver_exec_stage(
    const std::shared_ptr<CommSessionContext>& session_ctx,
    const std::vector<int32_t>& layer_ids,
    const std::vector<std::vector<at::Tensor>*>& local_tensors_ptrs,
    const std::unordered_map<int32_t, ReceiverLayerStorage>& layer_storages,
    const std::unordered_map<int32_t, std::vector<int32_t>>&
        normalized_layer_expert_ids,
    bool include_non_expert,
    bool transfer_all_experts,
    bool use_layer_storage_transfer) {
  auto promise = std::make_shared<std::promise<ReceiverTransferResult>>();
  std::future<ReceiverTransferResult> future = promise->get_future();
  if (session_ctx == nullptr) {
    LOG(ERROR) << "Session context is null in receiver stage.";
    promise->set_value(ReceiverTransferResult{});
    return future;
  }

  HcclComm target_hccl_comm = session_ctx->hccl_comm;
  aclrtStream target_stream = session_ctx->stream;
  auto expert_indices_set_ptr = get_cached_expert_indices_set(model_);
  int32_t ep_size = get_ep_size(context_.get_parallel_args());
  int32_t ep_rank = get_ep_rank(context_.get_parallel_args());
  auto session_ctx_for_thread = session_ctx;
  const absl::Time receiver_task_enqueued_time = absl::Now();

  session_manager_->schedule_hccl_task([&,
                                        local_tensors_ptrs,
                                        layer_ids,
                                        normalized_layer_expert_ids,
                                        include_non_expert,
                                        transfer_all_experts,
                                        target_hccl_comm,
                                        target_stream,
                                        session_ctx_for_thread,
                                        expert_indices_set_ptr,
                                        ep_size,
                                        ep_rank,
                                        layer_storages,
                                        use_layer_storage_transfer,
                                        receiver_task_enqueued_time,
                                        promise]() mutable {
    ReceiverTransferResult receiver_result;
    const absl::Time thread_start_time = absl::Now();
    receiver_result.queue_wait_ms = absl::ToDoubleMilliseconds(
        thread_start_time - receiver_task_enqueued_time);
    std::unique_lock<std::mutex> session_lock(
        session_ctx_for_thread->operation_mutex);
    aclError ret = aclrtSetDevice(device_id_);
    if (ret != ACL_SUCCESS) {
      LOG(ERROR) << "[Receiver Thread] SetContext Failed: " << ret;
      promise->set_value(receiver_result);
      return;
    }

    const auto& expert_indices_set = *expert_indices_set_ptr;

    const absl::Time start_time = absl::Now();
    size_t total_nbytes = 0;
    size_t layer_storage_items = 0;
    size_t tensor_items = 0;
    uint64_t storage_padding_nbytes = 0;
    std::vector<HcclSendRecvItem> items;
    const absl::Time build_items_start = absl::Now();

    for (size_t i = 0; i < local_tensors_ptrs.size(); ++i) {
      auto* tensors_ptr = local_tensors_ptrs[i];
      const int32_t layer_id = layer_ids[i];
      auto it = normalized_layer_expert_ids.find(layer_id);
      const std::vector<int32_t> layer_ids_for_experts =
          it == normalized_layer_expert_ids.end() ? std::vector<int32_t>{}
                                                  : it->second;
      bool appended_layer_storage = false;
      if (use_layer_storage_transfer && FLAGS_enable_manual_loader &&
          transfer_all_experts && include_non_expert &&
          layer_ids_for_experts.empty() && layer_id >= 0) {
        auto storage_it = layer_storages.find(layer_id);
        if (storage_it != layer_storages.end() &&
            storage_it->second.available &&
            storage_it->second.base_ptr != nullptr &&
            storage_it->second.storage_size > 0) {
          append_contiguous_storage_transfer_items(
              storage_it->second.base_ptr,
              storage_it->second.storage_size,
              HCCL_RECV,
              1,
              FLAGS_layer_storage_weight_transfer_chunk_bytes,
              &items,
              &total_nbytes);
          ++layer_storage_items;
          const uint64_t payload_nbytes =
              storage_it->second.payload_nbytes > 0
                  ? storage_it->second.payload_nbytes
                  : sum_tensor_nbytes(*tensors_ptr);
          if (storage_it->second.storage_size > payload_nbytes) {
            storage_padding_nbytes +=
                storage_it->second.storage_size - payload_nbytes;
          }
          appended_layer_storage = true;
        } else {
          LOG(WARNING) << "Layer storage transfer requested but receiver layer "
                          "storage is unavailable. Fallback to tensor items. "
                       << "layer_id=" << layer_id;
        }
      }
      if (!appended_layer_storage) {
        const size_t before_items = items.size();
        if (!append_layer_transfer_items(*tensors_ptr,
                                         layer_id,
                                         layer_ids_for_experts,
                                         expert_indices_set,
                                         include_non_expert,
                                         transfer_all_experts,
                                         ep_rank,
                                         ep_size,
                                         HCCL_RECV,
                                         1,
                                         &items,
                                         &total_nbytes)) {
          promise->set_value(receiver_result);
          return;
        }
        tensor_items += items.size() - before_items;
      }
    }
    receiver_result.build_items_ms = elapsed_ms_since(build_items_start);
    receiver_result.item_count = items.size();
    receiver_result.layer_storage_items = layer_storage_items;
    receiver_result.tensor_items = tensor_items;
    receiver_result.storage_padding_nbytes = storage_padding_nbytes;

    const absl::Time hccl_exec_start = absl::Now();
    if (!items.empty()) {
      auto hccl_ret = HcclBatchSendRecv(
          items.data(), items.size(), target_hccl_comm, target_stream);
      if (hccl_ret != HCCL_SUCCESS) {
        LOG(ERROR)
            << "[Receiver Thread] HcclBatchSendRecv (Multiple Layers) Failed.";
        promise->set_value(receiver_result);
        return;
      }
    }

    bool storage_view_init_ok = true;
    if (use_layer_storage_transfer && layer_storage_items > 0) {
      const absl::Time view_init_start = absl::Now();
      for (size_t i = 0; i < local_tensors_ptrs.size(); ++i) {
        const int32_t layer_id = layer_ids[i];
        auto storage_it = layer_storages.find(layer_id);
        if (storage_it == layer_storages.end() ||
            !storage_it->second.available ||
            storage_it->second.views_initialized) {
          continue;
        }
        if (storage_it->second.loader == nullptr) {
          storage_view_init_ok = false;
          break;
        }
        storage_it->second.loader->init_device_at_weights();
      }
      receiver_result.storage_view_init_ms = elapsed_ms_since(view_init_start);
    }

    auto sync_ret = aclrtSynchronizeStream(target_stream);
    receiver_result.hccl_exec_ms = elapsed_ms_since(hccl_exec_start);
    if (!storage_view_init_ok) {
      LOG(ERROR) << "[Receiver Thread] Failed to initialize tensor views from "
                    "layer storage after HCCL enqueue.";
      promise->set_value(receiver_result);
      return;
    }

    absl::Time end_time = absl::Now();
    double duration_s = absl::ToDoubleSeconds(end_time - start_time);
    double duration_ms = absl::ToDoubleMilliseconds(end_time - start_time);
    double total_gb = total_nbytes / (1024.0 * 1024.0 * 1024.0);
    double bandwidth_gb_s = total_gb / duration_s;

    LOG(INFO) << "[Receiver Thread] Batch transfer (layers: "
              << layer_ids.size()
              << ", include_non_expert=" << include_non_expert
              << ", transfer_all_experts=" << transfer_all_experts
              << ", use_layer_storage_transfer=" << use_layer_storage_transfer
              << ", requested_expert_ids="
              << count_expert_ids(normalized_layer_expert_ids) << ", "
              << summarize_hccl_transfer_items(items) << ", "
              << summarize_hccl_transfer_item_addresses(items)
              << ", layer_storage_items=" << layer_storage_items
              << ", tensor_items=" << tensor_items << ", storage_padding_mb="
              << (static_cast<double>(storage_padding_nbytes) /
                  (1024.0 * 1024.0))
              << "): " << std::fixed << std::setprecision(2) << total_gb
              << " GB, "
              << "Time: " << duration_ms << " ms, "
              << "Bandwidth: " << bandwidth_gb_s << " GB/s";

    receiver_result.ok = (sync_ret == ACL_SUCCESS);
    receiver_result.thread_total_ms =
        absl::ToDoubleMilliseconds(end_time - thread_start_time);
    receiver_result.total_nbytes = total_nbytes;
    promise->set_value(receiver_result);
  });
  return future;
}

bool WeightTransferReceiverEngine::pull_weight_internal(
    const std::string& session_id,
    const std::vector<int32_t>& layer_ids,
    const std::vector<std::vector<at::Tensor>*>& local_tensors_ptrs,
    const std::unordered_map<int32_t, ReceiverLayerStorage>& layer_storages,
    const std::unordered_map<int32_t, std::vector<int32_t>>& layer_expert_ids,
    bool include_non_expert,
    bool transfer_all_experts,
    bool allocate_tensors) {
  const absl::Time pull_internal_start = absl::Now();
  double meta_alloc_ms = 0.0;
  double meta_rpc_ms = 0.0;
  double tensor_alloc_ms = 0.0;

  if (session_id.empty()) {
    LOG(ERROR) << "Session id is empty in pull_weight_internal.";
    return false;
  }
  auto session_ctx = session_manager_->get_session_context(session_id);
  if (session_ctx == nullptr) {
    LOG(ERROR) << "Session not found for pull_weight_internal: " << session_id;
    return false;
  }
  if (!session_manager_->wait_for_session_ready(session_ctx,
                                                "Receiver batch transfer")) {
    return false;
  }
  const std::string effective_session_id = session_id;
  xllm::proto::WeightTransferService_Stub* target_stub =
      session_ctx->stub.get();
  if (target_stub == nullptr) {
    LOG(ERROR) << "Weight transfer stub is null for session " << session_id;
    return false;
  }
  if (layer_ids.size() != local_tensors_ptrs.size()) {
    LOG(ERROR) << "Layer ids size and local tensors ptr size mismatch: "
               << layer_ids.size() << " vs " << local_tensors_ptrs.size();
    return false;
  }
  for (auto* tensors_ptr : local_tensors_ptrs) {
    if (tensors_ptr == nullptr) {
      LOG(ERROR) << "Local tensor pointer should not be nullptr.";
      return false;
    }
  }

  auto normalized_layer_expert_ids =
      normalize_layer_expert_ids_map(layer_expert_ids);
  if (!validate_layer_expert_ids_map(layer_ids,
                                     normalized_layer_expert_ids,
                                     "Receiver layer expert map")) {
    LOG(ERROR) << "Invalid receiver layer expert map.";
    return false;
  }

  if (allocate_tensors) {
    MetaAllocateStats meta_stats;
    if (!fetch_weights_meta_and_allocate_tensors(target_stub,
                                                 layer_ids,
                                                 local_tensors_ptrs,
                                                 false,
                                                 nullptr,
                                                 "pull_weight_internal",
                                                 &meta_stats)) {
      return false;
    }
    meta_alloc_ms = meta_stats.total_ms;
    meta_rpc_ms = meta_stats.rpc_ms;
    tensor_alloc_ms = meta_stats.tensor_alloc_ms;
  }

  std::future<TriggerRpcResult> trigger_future =
      launch_trigger_rpc_stage(session_ctx,
                               layer_ids,
                               normalized_layer_expert_ids,
                               include_non_expert,
                               transfer_all_experts,
                               use_layer_storage_transfer_,
                               effective_session_id);
  std::future<ReceiverTransferResult> receiver_future =
      launch_receiver_exec_stage(session_ctx,
                                 layer_ids,
                                 local_tensors_ptrs,
                                 layer_storages,
                                 normalized_layer_expert_ids,
                                 include_non_expert,
                                 transfer_all_experts,
                                 use_layer_storage_transfer_);

  const absl::Time receiver_wait_start = absl::Now();
  ReceiverTransferResult receiver_result = receiver_future.get();
  double receiver_wait_ms = elapsed_ms_since(receiver_wait_start);

  const absl::Time trigger_wait_start = absl::Now();
  TriggerRpcResult trigger_result = trigger_future.get();
  double trigger_wait_ms = elapsed_ms_since(trigger_wait_start);

  bool result = receiver_result.ok;
  if (!trigger_result.ok) {
    result = false;
  }

  double pull_internal_total_ms = elapsed_ms_since(pull_internal_start);
  double receiver_total_gb = static_cast<double>(receiver_result.total_nbytes) /
                             (1024.0 * 1024.0 * 1024.0);
  LOG(INFO) << "[WeightPullTimingDetail] session_id=" << effective_session_id
            << ", layers=" << layer_ids.size()
            << ", include_non_expert=" << include_non_expert
            << ", transfer_all_experts=" << transfer_all_experts
            << ", allocate_tensors=" << allocate_tensors
            << ", total_ms=" << std::fixed << std::setprecision(2)
            << pull_internal_total_ms << ", meta_alloc_ms=" << meta_alloc_ms
            << ", meta_rpc_ms=" << meta_rpc_ms
            << ", tensor_alloc_ms=" << tensor_alloc_ms
            << ", trigger_queue_wait_ms=" << trigger_result.queue_wait_ms
            << ", trigger_rpc_ms=" << trigger_result.rpc_ms
            << ", trigger_wait_ms=" << trigger_wait_ms
            << ", receiver_queue_wait_ms=" << receiver_result.queue_wait_ms
            << ", receiver_build_items_ms=" << receiver_result.build_items_ms
            << ", receiver_storage_view_init_ms="
            << receiver_result.storage_view_init_ms
            << ", receiver_hccl_exec_ms=" << receiver_result.hccl_exec_ms
            << ", receiver_thread_total_ms=" << receiver_result.thread_total_ms
            << ", receiver_wait_ms=" << receiver_wait_ms
            << ", receiver_gb=" << receiver_total_gb
            << ", receiver_items=" << receiver_result.item_count
            << ", receiver_layer_storage_items="
            << receiver_result.layer_storage_items
            << ", receiver_tensor_items=" << receiver_result.tensor_items
            << ", receiver_storage_padding_mb="
            << (static_cast<double>(receiver_result.storage_padding_nbytes) /
                (1024.0 * 1024.0))
            << ", success=" << result;

  if (!result) {
    LOG(ERROR) << "Batch pull weight failed!";
  }
  return result;
}

}  // namespace xllm
