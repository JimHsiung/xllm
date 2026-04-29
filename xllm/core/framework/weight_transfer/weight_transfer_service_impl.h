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

#include "hccl_weight_transfer.pb.h"

namespace xllm {

class HcclWeightTransferImpl;

class WeightTransferServiceImpl : public xllm::proto::WeightTransferService {
 public:
  explicit WeightTransferServiceImpl(HcclWeightTransferImpl* impl);

  void InitComm(google::protobuf::RpcController* controller,
                const xllm::proto::InitCommRequest* request,
                xllm::proto::InitCommResponse* response,
                google::protobuf::Closure* done) override;

  void GetWeightsMeta(google::protobuf::RpcController* controller,
                      const xllm::proto::GetWeightsMetaRequest* request,
                      xllm::proto::GetWeightsMetaResponse* response,
                      google::protobuf::Closure* done) override;

  void TriggerWeightsSend(google::protobuf::RpcController* controller,
                          const xllm::proto::TriggerWeightsSendRequest* request,
                          xllm::proto::TriggerWeightsSendResponse* response,
                          google::protobuf::Closure* done) override;

 private:
  HcclWeightTransferImpl* impl_;
};

}  // namespace xllm
