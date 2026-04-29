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

#include <string>
#include <vector>

#include "framework/model/causal_lm.h"
#include "framework/weight_transfer/weight_transfer_types.h"

namespace xllm {

class WeightTransferAlltoallPlanner {
 public:
  std::vector<SourceExpertTransferTask> build_source_expert_transfer_tasks(
      const RankExpertTransferPlanData& rank_plan,
      const ModelArgs& model_args) const;

  std::vector<int32_t> collect_sorted_layer_ids(
      const LayerExpertIdsMap& layer_expert_ids) const;

  size_t count_task_expert_ids(const LayerExpertIdsMap& layer_expert_ids) const;

  bool compare_source_task_by_addr(const SourceExpertTransferTask& lhs,
                                   const SourceExpertTransferTask& rhs) const;

  bool build_alltoall_round_plans(
      CausalLM* model,
      const ParallelArgs& parallel_args,
      const std::vector<SourceExpertTransferTask>& source_tasks,
      std::vector<AlltoallRoundPlan>* round_plans) const;

  void build_alltoall_requests(
      const std::vector<AlltoallRoundPlan>& round_plans,
      const std::vector<SourceExpertTransferTask>& source_tasks,
      uint32_t receiver_rank,
      const std::string& session_id,
      std::vector<xllm::proto::AlltoAllRoundDesc>* receiver_rounds,
      std::vector<xllm::proto::TriggerWeightsSendRequest>* source_requests)
      const;
};

}  // namespace xllm
