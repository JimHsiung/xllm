#include "hccl_weight_transfer.h"

#include <glog/logging.h>
#include <torch_npu/csrc/core/npu/NPUFormat.h>

#include <iomanip>

#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "util/net.h"

namespace xllm {

class WeightTransferServiceImpl : public xllm::proto::WeightTransferService {
 public:
  explicit WeightTransferServiceImpl(HcclWeightTransfer* hccl_weight_transfer)
      : hccl_weight_transfer_(hccl_weight_transfer) {}

  void InitComm(google::protobuf::RpcController* controller,
                const xllm::proto::InitCommRequest* request,
                xllm::proto::InitCommResponse* response,
                google::protobuf::Closure* done) override {
    brpc::ClosureGuard done_guard(done);

    std::string remote_addr = request->addr();
    std::string root_info_str = request->root_info();

    std::thread([this, remote_addr, root_info_str]() {
      LOG(INFO) << "Sender Async Thread: Start Waiting for Receiver to join "
                   "HCCL group...";

      hccl_weight_transfer_->handle_init_comm(remote_addr,
                                              root_info_str.data());

      LOG(INFO) << "Sender Async Thread: HCCL Init DONE! Handshake complete.";
    }).detach();
    response->set_success(true);
    LOG(INFO)
        << "Sender: RootInfo generated and sent back. Async Init triggered.";
  }

  void GetLayerMeta(google::protobuf::RpcController* controller,
                    const xllm::proto::GetLayerMetaRequest* request,
                    xllm::proto::GetLayerMetaResponse* response,
                    google::protobuf::Closure* done) override {
    brpc::ClosureGuard done_guard(done);
    auto tensors =
        hccl_weight_transfer_->get_registered_tensors(request->layer_id());

    for (const auto& t : tensors) {
      auto* meta = response->add_metas();
      meta->set_dtype(static_cast<int32_t>(t.scalar_type()));
      for (int i = 0; i < t.dim(); ++i) {
        meta->add_shape(t.size(i));
      }
      meta->set_npu_format(at_npu::native::get_npu_format(t));
    }
  }

  void TriggerSend(google::protobuf::RpcController* controller,
                   const xllm::proto::TriggerSendRequest* request,
                   xllm::proto::TriggerSendResponse* response,
                   google::protobuf::Closure* done) override {
    brpc::ClosureGuard done_guard(done);
    hccl_weight_transfer_->process_send_request(request->layer_id());
    response->set_success(true);
  }

  void GetWeightsMeta(google::protobuf::RpcController* controller,
                      const xllm::proto::GetWeightsMetaRequest* request,
                      xllm::proto::GetWeightsMetaResponse* response,
                      google::protobuf::Closure* done) override {
    brpc::ClosureGuard done_guard(done);
    for (int32_t layer_id : request->layer_ids()) {
      auto* layer_meta = response->add_layer_metas();
      layer_meta->set_layer_id(layer_id);
      auto tensors = hccl_weight_transfer_->get_registered_tensors(layer_id);
      for (const auto& t : tensors) {
        auto* meta = layer_meta->add_metas();
        meta->set_dtype(static_cast<int32_t>(t.scalar_type()));
        for (int i = 0; i < t.dim(); ++i) {
          meta->add_shape(t.size(i));
        }
        meta->set_npu_format(at_npu::native::get_npu_format(t));
      }
    }
  }

  void TriggerWeightsSend(google::protobuf::RpcController* controller,
                          const xllm::proto::TriggerWeightsSendRequest* request,
                          xllm::proto::TriggerWeightsSendResponse* response,
                          google::protobuf::Closure* done) override {
    brpc::ClosureGuard done_guard(done);
    std::vector<int32_t> layer_ids;
    for (int32_t id : request->layer_ids()) layer_ids.push_back(id);
    hccl_weight_transfer_->process_weights_send_request(layer_ids);
    response->set_success(true);
  }

 private:
  HcclWeightTransfer* hccl_weight_transfer_;
};

HcclWeightTransfer::HcclWeightTransfer(const ModelContext& context,
                                       CausalLM* model,
                                       const int32_t device_id,
                                       const int32_t listen_port)
    : context_(context),
      model_(model),
      device_id_(device_id),
      listen_port_(listen_port) {
  aclrtSetDevice(device_id_);
  aclrtCreateStream(&stream_);

  std::string ip = net::get_local_ip_addr();
  local_addr_ = ip + ":" + std::to_string(listen_port);
  hccl_thread_pool_ = std::make_shared<ThreadPool>();
  rpc_thread_pool_ = std::make_shared<ThreadPool>();
}

HcclWeightTransfer::~HcclWeightTransfer() {
  if (server_.IsRunning()) server_.Stop(0);
  server_.Join();
  if (hccl_comm_) HcclCommDestroy(hccl_comm_);
  aclrtDestroyStream(stream_);
}

void HcclWeightTransfer::register_layer(
    int32_t layer_id,
    const std::vector<at::Tensor>& tensors) {
  layer_registry_[layer_id] = tensors;
}

std::vector<at::Tensor> HcclWeightTransfer::get_registered_tensors(
    int32_t layer_id) {
  if (layer_registry_.find(layer_id) == layer_registry_.end()) {
    return {};
  }
  return layer_registry_[layer_id];
}

void HcclWeightTransfer::start_serving() {
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

bool HcclWeightTransfer::handle_init_comm(const std::string& remote_addr,
                                          const void* root_info_ptr) {
  if (is_comm_initialized_) return true;

  LOG(INFO) << "Sender: Initializing HCCL Comm with Receiver " << remote_addr;
  aclrtSetDevice(device_id_);

  HcclRootInfo root_info;
  memcpy(&root_info, root_info_ptr, sizeof(HcclRootInfo));

  auto ret = HcclCommInitRootInfo(2, &root_info, 1, &hccl_comm_);
  if (ret != HCCL_SUCCESS) {
    LOG(ERROR) << "HcclCommInitRootInfo failed: " << ret;
    return false;
  }
  is_comm_initialized_ = true;
  return true;
}

void HcclWeightTransfer::process_send_request(int32_t layer_id) {
  int wait_retry = 0;
  while (!is_comm_initialized_) {
    if (wait_retry % 100 == 0) {
      LOG(WARNING) << "Sender: Waiting for HCCL Init to "
                      "finish before sending layer "
                   << layer_id << "...";
    }
    usleep(10000);
    wait_retry++;

    if (wait_retry > 2000) {
      LOG(ERROR) << "Sender: FATAL TIMEOUT waiting for HCCL Init.";
      return;
    }
  }

  aclrtSetDevice(device_id_);
  auto tensors = get_registered_tensors(layer_id);
  if (tensors.empty()) {
    LOG(ERROR) << "Request to send unknown layer " << layer_id;
    return;
  }

  auto promise = std::make_shared<std::promise<bool>>();
  std::future<bool> future = promise->get_future();

  hccl_thread_pool_->schedule([&, layer_id, tensors]() mutable {
    aclrtSetDevice(device_id_);
    // LOG(INFO) << "[Sender Thread] Task started for Layer " << layer_id;

    auto expert_indices = model_->get_expert_weight_indices();
    std::unordered_set<int> expert_indices_set(expert_indices.begin(),
                                               expert_indices.end());

    absl::Time start_time = absl::Now();
    size_t total_nbytes = 0;

    std::vector<HcclSendRecvItem> items;
    for (size_t i = 0; i < tensors.size(); ++i) {
      const auto& tensor = tensors[i];
      size_t nbytes = tensor.nbytes();
      total_nbytes += nbytes;

      if (expert_indices_set.count(i) && tensor.dim() == 3) {
        int64_t expert_num = tensor.size(0);
        size_t expert_nbytes = nbytes / expert_num;
        for (int64_t e = 0; e < expert_num; ++e) {
          void* data_ptr =
              static_cast<uint8_t*>(tensor.data_ptr()) + e * expert_nbytes;
          items.push_back({HCCL_SEND,
                           data_ptr,
                           (uint64_t)expert_nbytes,
                           HCCL_DATA_TYPE_UINT8,
                           0});
        }
      } else {
        items.push_back({HCCL_SEND,
                         tensor.data_ptr(),
                         (uint64_t)nbytes,
                         HCCL_DATA_TYPE_UINT8,
                         0});
      }
    }

    if (!items.empty()) {
      auto hccl_ret =
          HcclBatchSendRecv(items.data(), items.size(), hccl_comm_, stream_);
      if (hccl_ret != HCCL_SUCCESS) {
        LOG(ERROR) << "[Sender Thread] HcclBatchSendRecv Failed.";
        promise->set_value(false);
        return;
      }
    }
    auto sync_ret = aclrtSynchronizeStream(stream_);

    absl::Time end_time = absl::Now();
    double duration_s = absl::ToDoubleSeconds(end_time - start_time);
    double duration_ms = absl::ToDoubleMilliseconds(end_time - start_time);
    double total_gb = total_nbytes / (1024.0 * 1024.0 * 1024.0);
    double bandwidth_gb_s = total_gb / duration_s;

    LOG(INFO) << "[Sender Thread] Layer " << layer_id
              << " transfer: " << std::fixed << std::setprecision(2) << total_gb
              << " GB, "
              << "Time: " << duration_ms << " ms, "
              << "Bandwidth: " << bandwidth_gb_s << " GB/s";

    if (sync_ret != ACL_SUCCESS) {
      promise->set_value(false);
    } else {
      promise->set_value(true);
    }
  });
  bool result = future.get();

  if (!result) {
    LOG(ERROR) << "Sender process failed for layer " << layer_id;
  }
}

void HcclWeightTransfer::process_weights_send_request(
    const std::vector<int32_t>& layer_ids) {
  int wait_retry = 0;
  while (!is_comm_initialized_) {
    if (wait_retry % 100 == 0) {
      LOG(WARNING) << "Sender: Waiting for HCCL Init to "
                      "finish before sending weights ";
    }
    usleep(10000);
    wait_retry++;

    if (wait_retry > 2000) {
      LOG(ERROR) << "Sender: FATAL TIMEOUT waiting for HCCL Init.";
      return;
    }
  }

  aclrtSetDevice(device_id_);

  auto promise = std::make_shared<std::promise<bool>>();
  std::future<bool> future = promise->get_future();

  hccl_thread_pool_->schedule([&, layer_ids]() mutable {
    aclrtSetDevice(device_id_);

    auto expert_indices = model_->get_expert_weight_indices();
    std::unordered_set<int> expert_indices_set(expert_indices.begin(),
                                               expert_indices.end());

    absl::Time start_time = absl::Now();
    size_t total_nbytes = 0;

    std::vector<HcclSendRecvItem> items;
    for (int32_t layer_id : layer_ids) {
      auto tensors = get_registered_tensors(layer_id);
      for (size_t i = 0; i < tensors.size(); ++i) {
        const auto& tensor = tensors[i];
        size_t nbytes = tensor.nbytes();
        total_nbytes += nbytes;

        if (expert_indices_set.count(i) && tensor.dim() == 3) {
          int64_t expert_num = tensor.size(0);
          size_t expert_nbytes = nbytes / expert_num;
          for (int64_t e = 0; e < expert_num; ++e) {
            void* data_ptr =
                static_cast<uint8_t*>(tensor.data_ptr()) + e * expert_nbytes;
            items.push_back({HCCL_SEND,
                             data_ptr,
                             (uint64_t)expert_nbytes,
                             HCCL_DATA_TYPE_UINT8,
                             0});
          }
        } else {
          items.push_back({HCCL_SEND,
                           tensor.data_ptr(),
                           (uint64_t)nbytes,
                           HCCL_DATA_TYPE_UINT8,
                           0});
        }
      }
    }

    if (!items.empty()) {
      auto hccl_ret =
          HcclBatchSendRecv(items.data(), items.size(), hccl_comm_, stream_);
      if (hccl_ret != HCCL_SUCCESS) {
        LOG(ERROR)
            << "[Sender Thread] HcclBatchSendRecv (Multiple Layers) Failed.";
        promise->set_value(false);
        return;
      }
    }
    auto sync_ret = aclrtSynchronizeStream(stream_);

    absl::Time end_time = absl::Now();
    double duration_s = absl::ToDoubleSeconds(end_time - start_time);
    double duration_ms = absl::ToDoubleMilliseconds(end_time - start_time);
    double total_gb = total_nbytes / (1024.0 * 1024.0 * 1024.0);
    double bandwidth_gb_s = total_gb / duration_s;

    LOG(INFO) << "[Sender Thread] Batch transfer (layers: " << layer_ids.size()
              << "): " << std::fixed << std::setprecision(2) << total_gb
              << " GB, "
              << "Time: " << duration_ms << " ms, "
              << "Bandwidth: " << bandwidth_gb_s << " GB/s";

    if (sync_ret != ACL_SUCCESS) {
      promise->set_value(false);
    } else {
      promise->set_value(true);
    }
  });
  future.wait();
}

bool HcclWeightTransfer::connect_to_remote(const std::string& remote_addr) {
  aclrtSetDevice(device_id_);
  channel_ = std::make_unique<brpc::Channel>();
  brpc::ChannelOptions options;
  options.timeout_ms = 10000;
  options.connect_timeout_ms = 2000;
  options.max_retry = 3;

  // Initialize Channel
  if (channel_->Init(remote_addr.c_str(), &options) != 0) {
    LOG(ERROR) << "BRPC Channel init failed";
    return false;
  }
  stub_ =
      std::make_unique<xllm::proto::WeightTransferService_Stub>(channel_.get());

  // Prepare HCCL Root Info
  aclrtSetDevice(device_id_);
  HcclRootInfo root_info;
  auto ret = HcclGetRootInfo(&root_info);
  if (ret != HCCL_SUCCESS) {
    LOG(ERROR) << "HcclGetRootInfo failed";
    return false;
  }

  // Loop to try connecting (Polling)
  // Sender may be loading weights, we need to keep trying until its port opens
  int max_wait_retries = 100;  // Wait at most 100 times
  bool connect_success = false;

  xllm::proto::InitCommRequest req;
  xllm::proto::InitCommResponse resp;
  req.set_addr(local_addr_);
  req.set_root_info(&root_info, sizeof(HcclRootInfo));

  for (int i = 0; i < max_wait_retries; ++i) {
    brpc::Controller cntl;

    // Try to initiate RPC
    stub_->InitComm(&cntl, &req, &resp, nullptr);

    if (!cntl.Failed()) {
      // RPC succeeded, meaning connection established!
      if (resp.success()) {
        LOG(INFO) << "Receiver: Successfully connected to Sender at "
                  << remote_addr;
        connect_success = true;
        break;
      } else {
        LOG(ERROR) << "Receiver: Connected, but Sender returned logic error.";
        return false;
      }
    }

    if (i % 5 == 0) {
      LOG(WARNING) << "Receiver: Waiting for Sender (" << remote_addr
                   << ") to come online... (Attempt " << i + 1 << "/"
                   << max_wait_retries << ")";
    }

    // Sleep 1 second before retry
    sleep(1);
  }

  if (!connect_success) {
    LOG(ERROR)
        << "Receiver: FATAL - Timed out waiting for Sender to start after "
        << max_wait_retries << " seconds.";
    return false;
  }

  // Receiver initializes HCCL (Rank 1)
  LOG(INFO) << "Receiver: Initializing HCCL Comm (Rank 1)...";
  ret = HcclCommInitRootInfo(2, &root_info, 0, &hccl_comm_);
  if (ret != HCCL_SUCCESS) {
    LOG(ERROR) << "HcclCommInitRootInfo (Client) failed: " << ret;
    return false;
  }

  is_comm_initialized_ = true;
  return true;
}

bool HcclWeightTransfer::pull_layer(int32_t layer_id,
                                    std::vector<at::Tensor>& local_tensors) {
  if (!is_comm_initialized_) return false;

  brpc::Controller cntl_meta;
  xllm::proto::GetLayerMetaRequest req_meta;
  xllm::proto::GetLayerMetaResponse resp_meta;
  req_meta.set_layer_id(layer_id);

  stub_->GetLayerMeta(&cntl_meta, &req_meta, &resp_meta, nullptr);
  if (cntl_meta.Failed()) {
    LOG(ERROR) << "GetLayerMeta failed: " << cntl_meta.ErrorText();
    return false;
  }

  int tensor_count = resp_meta.metas_size();
  local_tensors.resize(tensor_count);

  aclrtSetDevice(device_id_);

  for (int i = 0; i < tensor_count; ++i) {
    const auto& meta = resp_meta.metas(i);
    std::vector<int64_t> shape;
    for (int64_t d : meta.shape()) shape.push_back(d);

    auto options = torch::TensorOptions()
                       .dtype(static_cast<at::ScalarType>(meta.dtype()))
                       .device("npu:" + std::to_string(device_id_));

    local_tensors[i] =
        at_npu::native::empty_with_format(shape, options, meta.npu_format());
  }

  rpc_thread_pool_->schedule([this, layer_id]() {
    brpc::Controller cntl_trig;
    xllm::proto::TriggerSendRequest req_trig;
    xllm::proto::TriggerSendResponse resp_trig;
    req_trig.set_layer_id(layer_id);

    stub_->TriggerSend(&cntl_trig, &req_trig, &resp_trig, nullptr);

    if (cntl_trig.Failed() || !resp_trig.success()) {
      LOG(ERROR) << "TriggerSend failed: " << cntl_trig.ErrorText();
    }
  });
  auto promise = std::make_shared<std::promise<bool>>();
  std::future<bool> future = promise->get_future();

  hccl_thread_pool_->schedule([&]() mutable {
    aclError ret;
    ret = aclrtSetDevice(device_id_);

    if (ret != ACL_SUCCESS) {
      LOG(ERROR) << "[Receiver Thread] SetContext Failed: " << ret;
      promise->set_value(false);
      return;
    }

    auto expert_indices = model_->get_expert_weight_indices();
    std::unordered_set<int> expert_indices_set(expert_indices.begin(),
                                               expert_indices.end());

    absl::Time start_time = absl::Now();
    size_t total_nbytes = 0;

    std::vector<HcclSendRecvItem> items;
    for (size_t i = 0; i < local_tensors.size(); ++i) {
      auto& tensor = local_tensors[i];
      size_t nbytes = tensor.nbytes();
      total_nbytes += nbytes;

      if (expert_indices_set.count(i) && tensor.dim() == 3) {
        int64_t expert_num = tensor.size(0);
        size_t expert_nbytes = nbytes / expert_num;
        for (int64_t e = 0; e < expert_num; ++e) {
          void* data_ptr =
              static_cast<uint8_t*>(tensor.data_ptr()) + e * expert_nbytes;
          items.push_back({HCCL_RECV,
                           data_ptr,
                           (uint64_t)expert_nbytes,
                           HCCL_DATA_TYPE_UINT8,
                           1});
        }
      } else {
        items.push_back({HCCL_RECV,
                         tensor.data_ptr(),
                         (uint64_t)nbytes,
                         HCCL_DATA_TYPE_UINT8,
                         1});
      }
    }

    if (!items.empty()) {
      auto hccl_ret =
          HcclBatchSendRecv(items.data(), items.size(), hccl_comm_, stream_);
      if (hccl_ret != HCCL_SUCCESS) {
        LOG(ERROR) << "[Receiver Thread] HcclBatchSendRecv Failed.";
        promise->set_value(false);
        return;
      }
    }

    auto sync_ret = aclrtSynchronizeStream(stream_);

    absl::Time end_time = absl::Now();
    double duration_s = absl::ToDoubleSeconds(end_time - start_time);
    double duration_ms = absl::ToDoubleMilliseconds(end_time - start_time);
    double total_gb = total_nbytes / (1024.0 * 1024.0 * 1024.0);
    double bandwidth_gb_s = total_gb / duration_s;

    LOG(INFO) << "[Receiver Thread] Layer " << layer_id
              << " transfer: " << std::fixed << std::setprecision(2) << total_gb
              << " GB, "
              << "Time: " << duration_ms << " ms, "
              << "Bandwidth: " << bandwidth_gb_s << " GB/s";

    promise->set_value(sync_ret == ACL_SUCCESS);
  });

  bool result = future.get();
  if (!result) {
    LOG(ERROR) << "Push layer failed!";
  }

  return result;
}

bool HcclWeightTransfer::pull_weight(
    const std::vector<int32_t>& layer_ids,
    const std::vector<std::vector<at::Tensor>*>& local_tensors_ptrs) {
  if (!is_comm_initialized_) return false;

  brpc::Controller cntl_meta;
  xllm::proto::GetWeightsMetaRequest req_meta;
  xllm::proto::GetWeightsMetaResponse resp_meta;
  for (int32_t id : layer_ids) req_meta.add_layer_ids(id);

  stub_->GetWeightsMeta(&cntl_meta, &req_meta, &resp_meta, nullptr);
  if (cntl_meta.Failed()) {
    LOG(ERROR) << "GetWeightsMeta failed: " << cntl_meta.ErrorText();
    return false;
  }

  aclrtSetDevice(device_id_);

  for (int i = 0; i < resp_meta.layer_metas_size(); ++i) {
    const auto& layer_meta = resp_meta.layer_metas(i);
    auto& tensors = *local_tensors_ptrs[i];
    tensors.resize(layer_meta.metas_size());

    for (int j = 0; j < layer_meta.metas_size(); ++j) {
      const auto& meta = layer_meta.metas(j);
      std::vector<int64_t> shape;
      for (int64_t d : meta.shape()) shape.push_back(d);

      auto options = torch::TensorOptions()
                         .dtype(static_cast<at::ScalarType>(meta.dtype()))
                         .device("npu:" + std::to_string(device_id_));

      tensors[j] =
          at_npu::native::empty_with_format(shape, options, meta.npu_format());
    }
  }

  rpc_thread_pool_->schedule([this, layer_ids]() {
    brpc::Controller cntl_trig;
    xllm::proto::TriggerWeightsSendRequest req_trig;
    xllm::proto::TriggerWeightsSendResponse resp_trig;
    for (int32_t id : layer_ids) req_trig.add_layer_ids(id);

    stub_->TriggerWeightsSend(&cntl_trig, &req_trig, &resp_trig, nullptr);

    if (cntl_trig.Failed() || !resp_trig.success()) {
      LOG(ERROR) << "TriggerWeightsSend failed: " << cntl_trig.ErrorText();
    }
  });

  auto promise = std::make_shared<std::promise<bool>>();
  std::future<bool> future = promise->get_future();

  hccl_thread_pool_->schedule([&, local_tensors_ptrs, layer_ids]() mutable {
    aclError ret = aclrtSetDevice(device_id_);
    if (ret != ACL_SUCCESS) {
      LOG(ERROR) << "[Receiver Thread] SetContext Failed: " << ret;
      promise->set_value(false);
      return;
    }

    auto expert_indices = model_->get_expert_weight_indices();
    std::unordered_set<int> expert_indices_set(expert_indices.begin(),
                                               expert_indices.end());

    absl::Time start_time = absl::Now();
    size_t total_nbytes = 0;

    std::vector<HcclSendRecvItem> items;
    for (auto* tensors_ptr : local_tensors_ptrs) {
      auto& local_tensors = *tensors_ptr;
      for (size_t i = 0; i < local_tensors.size(); ++i) {
        auto& tensor = local_tensors[i];
        size_t nbytes = tensor.nbytes();
        total_nbytes += nbytes;

        if (expert_indices_set.count(i) && tensor.dim() == 3) {
          int64_t expert_num = tensor.size(0);
          size_t expert_nbytes = nbytes / expert_num;
          for (int64_t e = 0; e < expert_num; ++e) {
            void* data_ptr =
                static_cast<uint8_t*>(tensor.data_ptr()) + e * expert_nbytes;
            items.push_back({HCCL_RECV,
                             data_ptr,
                             (uint64_t)expert_nbytes,
                             HCCL_DATA_TYPE_UINT8,
                             1});
          }
        } else {
          items.push_back({HCCL_RECV,
                           tensor.data_ptr(),
                           (uint64_t)nbytes,
                           HCCL_DATA_TYPE_UINT8,
                           1});
        }
      }
    }

    if (!items.empty()) {
      auto hccl_ret =
          HcclBatchSendRecv(items.data(), items.size(), hccl_comm_, stream_);
      if (hccl_ret != HCCL_SUCCESS) {
        LOG(ERROR)
            << "[Receiver Thread] HcclBatchSendRecv (Multiple Layers) Failed.";
        promise->set_value(false);
        return;
      }
    }

    auto sync_ret = aclrtSynchronizeStream(stream_);

    absl::Time end_time = absl::Now();
    double duration_s = absl::ToDoubleSeconds(end_time - start_time);
    double duration_ms = absl::ToDoubleMilliseconds(end_time - start_time);
    double total_gb = total_nbytes / (1024.0 * 1024.0 * 1024.0);
    double bandwidth_gb_s = total_gb / duration_s;

    LOG(INFO) << "[Receiver Thread] Batch transfer (layers: "
              << layer_ids.size() << "): " << std::fixed << std::setprecision(2)
              << total_gb << " GB, "
              << "Time: " << duration_ms << " ms, "
              << "Bandwidth: " << bandwidth_gb_s << " GB/s";

    promise->set_value(sync_ret == ACL_SUCCESS);
  });

  bool result = future.get();
  if (!result) {
    LOG(ERROR) << "Batch pull weight failed!";
  }

  return result;
}

}  // namespace xllm