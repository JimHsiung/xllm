#pragma once

#include <acl/acl.h>
#include <brpc/channel.h>
#include <brpc/server.h>
#include <hccl/hccl.h>
#include <torch/torch.h>

#include <future>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "framework/model/causal_lm.h"
#include "framework/model_context.h"
#include "hccl_weight_transfer.pb.h"
#include "util/threadpool.h"

namespace xllm {

class WeightTransferServiceImpl;

class HcclWeightTransfer {
 public:
  HcclWeightTransfer(const ModelContext& context,
                     CausalLM* model,
                     const int32_t device_id,
                     const int32_t listen_port);
  ~HcclWeightTransfer();

  void register_layer(int32_t layer_id, const std::vector<at::Tensor>& tensors);

  void start_serving();

  void process_send_request(int32_t layer_id);

  bool connect_to_remote(const std::string& remote_addr);

  bool pull_layer(int32_t layer_id, std::vector<at::Tensor>& local_tensors);

  bool handle_init_comm(const std::string& remote_addr,
                        const void* root_info_ptr);

  std::vector<at::Tensor> get_registered_tensors(int32_t layer_id);

 private:
  ModelContext context_;
  CausalLM* model_;

  int32_t device_id_;
  int32_t listen_port_;
  std::string local_addr_;

  aclrtStream stream_ = nullptr;
  HcclComm hccl_comm_ = nullptr;
  bool is_comm_initialized_ = false;

  brpc::Server server_;
  std::unique_ptr<WeightTransferServiceImpl> service_;

  std::unique_ptr<brpc::Channel> channel_;
  std::unique_ptr<xllm::proto::WeightTransferService_Stub> stub_;

  std::unordered_map<int32_t, std::vector<at::Tensor>> layer_registry_;

  std::shared_ptr<ThreadPool> hccl_thread_pool_;
  std::shared_ptr<ThreadPool> rpc_thread_pool_;
};

}  // namespace xllm