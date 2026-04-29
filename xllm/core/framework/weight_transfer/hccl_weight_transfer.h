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

#include <torch/torch.h>

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "common/types.h"
#include "hccl_weight_transfer.pb.h"

namespace xllm {

class CausalLM;
class ModelContext;
class HcclWeightTransferImpl;

class HcclWeightTransfer {
 public:
  HcclWeightTransfer(const ModelContext& context,
                     CausalLM* model,
                     int32_t device_id,
                     int32_t listen_port);
  ~HcclWeightTransfer();

  void register_layer(int32_t layer_id, const std::vector<at::Tensor>& tensors);

  void start_serving();

  void process_weights_send_request(const std::string& session_id,
                                    const std::vector<int32_t>& layer_ids);
  void process_weights_send_request(
      const std::string& session_id,
      const std::vector<int32_t>& layer_ids,
      const std::unordered_map<int32_t, std::vector<int32_t>>& layer_expert_ids,
      bool include_non_expert);
  bool process_weights_alltoallv_send_request(
      const std::string& session_id,
      uint32_t receiver_rank,
      const std::vector<xllm::proto::AlltoAllRoundDesc>& alltoall_rounds,
      uint64_t* transferred_bytes,
      std::string* error_msg);

  bool pull_model_from_instance(
      const std::string& remote_addr,
      const RankExpertTransferPlanData& rank_expert_transfer_plan);

  bool handle_init_comm(const std::string& remote_addr,
                        const void* root_info_ptr,
                        uint32_t n_ranks,
                        uint32_t rank,
                        const std::string& session_id,
                        xllm::proto::CommMode comm_mode);

  const std::vector<at::Tensor>& get_registered_tensors(int32_t layer_id) const;

  std::string get_weight_transfer_addr() const;

 private:
  std::unique_ptr<HcclWeightTransferImpl> impl_;
};

}  // namespace xllm
