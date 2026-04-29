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

#include "framework/weight_transfer/weight_transfer_alltoall_planner.h"

#include <glog/logging.h>

#include <algorithm>
#include <sstream>
#include <unordered_map>
#include <unordered_set>

#include "framework/weight_transfer/weight_transfer_common.h"

namespace xllm {
namespace {

std::string format_int32_list_limited(const std::vector<int32_t>& values,
                                      size_t max_items) {
  std::ostringstream oss;
  oss << "[";
  size_t limit = std::min(values.size(), max_items);
  for (size_t i = 0; i < limit; ++i) {
    if (i > 0) {
      oss << ",";
    }
    oss << values[i];
  }
  if (values.size() > limit) {
    if (limit > 0) {
      oss << ",";
    }
    oss << "...(" << (values.size() - limit) << " more)";
  }
  oss << "]";
  return oss.str();
}

std::string format_local_runs_limited(const std::vector<LocalExpertRun>& runs,
                                      size_t max_items) {
  std::ostringstream oss;
  oss << "[";
  size_t limit = std::min(runs.size(), max_items);
  for (size_t i = 0; i < limit; ++i) {
    if (i > 0) {
      oss << ",";
    }
    oss << "{start=" << runs[i].local_expert_start
        << ",count=" << runs[i].local_expert_count << "}";
  }
  if (runs.size() > limit) {
    if (limit > 0) {
      oss << ",";
    }
    oss << "...(" << (runs.size() - limit) << " more)";
  }
  oss << "]";
  return oss.str();
}

std::string format_source_run_counts(
    const std::vector<SourceExpertTransferTask>& source_tasks,
    const std::vector<std::vector<LocalExpertRun>>& runs_per_source) {
  std::ostringstream oss;
  oss << "[";
  for (size_t i = 0; i < source_tasks.size(); ++i) {
    if (i > 0) {
      oss << ",";
    }
    size_t run_count =
        i < runs_per_source.size() ? runs_per_source[i].size() : 0;
    oss << source_tasks[i].source_addr << ":" << run_count;
  }
  oss << "]";
  return oss.str();
}

std::string format_round_active_sources(
    const std::vector<SourceExpertTransferTask>& source_tasks,
    const AlltoallRoundPlan& round_plan) {
  std::ostringstream oss;
  oss << "[";
  bool first = true;
  for (size_t source_idx = 0; source_idx < source_tasks.size(); ++source_idx) {
    if (source_idx >= round_plan.source_runs.size() ||
        !round_plan.source_runs[source_idx].active) {
      continue;
    }
    if (!first) {
      oss << ",";
    }
    first = false;
    const auto& run = round_plan.source_runs[source_idx].run;
    oss << source_tasks[source_idx].source_addr
        << "{start=" << run.local_expert_start
        << ",count=" << run.local_expert_count << "}";
  }
  oss << "]";
  return oss.str();
}

bool build_local_expert_runs(const std::vector<int32_t>& expert_ids,
                             int64_t local_expert_num,
                             int32_t ep_rank,
                             int32_t layer_id,
                             int32_t tensor_idx,
                             std::vector<LocalExpertRun>* runs) {
  CHECK(runs != nullptr);
  runs->clear();
  if (expert_ids.empty()) {
    return true;
  }
  if (local_expert_num <= 0) {
    LOG(ERROR) << "Invalid local expert number " << local_expert_num
               << " at layer " << layer_id << ", tensor " << tensor_idx;
    return false;
  }

  const int64_t local_expert_begin =
      static_cast<int64_t>(ep_rank) * local_expert_num;
  const int64_t local_expert_end = local_expert_begin + local_expert_num;
  std::vector<int32_t> local_indices;
  local_indices.reserve(expert_ids.size());
  for (int32_t expert_id : expert_ids) {
    if (expert_id < local_expert_begin || expert_id >= local_expert_end) {
      LOG(ERROR) << "Expert id " << expert_id << " is outside local slice ["
                 << local_expert_begin << ", " << local_expert_end
                 << ") at layer " << layer_id << ", tensor " << tensor_idx
                 << ", ep_rank=" << ep_rank;
      return false;
    }
    local_indices.push_back(
        static_cast<int32_t>(expert_id - local_expert_begin));
  }
  std::sort(local_indices.begin(), local_indices.end());
  local_indices.erase(std::unique(local_indices.begin(), local_indices.end()),
                      local_indices.end());
  VLOG(1) << "[AlltoallRoundBuild] local index mapping. layer=" << layer_id
          << ", tensor=" << tensor_idx << ", ep_rank=" << ep_rank
          << ", local_expert_num=" << local_expert_num
          << ", global_expert_ids=" << format_int32_list_limited(expert_ids, 64)
          << ", local_indices=" << format_int32_list_limited(local_indices, 64);

  int32_t start = local_indices.front();
  int32_t prev = local_indices.front();
  for (size_t i = 1; i < local_indices.size(); ++i) {
    int32_t cur = local_indices[i];
    if (cur == prev + 1) {
      prev = cur;
      continue;
    }
    runs->push_back(LocalExpertRun{start, prev - start + 1});
    start = cur;
    prev = cur;
  }
  runs->push_back(LocalExpertRun{start, prev - start + 1});
  VLOG(1) << "[AlltoallRoundBuild] local runs generated. layer=" << layer_id
          << ", tensor=" << tensor_idx << ", run_count=" << runs->size()
          << ", runs=" << format_local_runs_limited(*runs, 64);
  return true;
}

std::vector<int32_t> build_moe_layer_ids(const ModelArgs& args) {
  bool is_deepseek_moe =
      args.n_routed_experts() > 0 && args.num_experts_per_tok() > 0;
  bool is_qwen_moe = args.num_experts() > 0 && args.num_experts_per_tok() > 0;
  std::vector<int32_t> moe_layer_ids;

  if (is_deepseek_moe) {
    const int32_t moe_layer_count = std::max<int32_t>(
        static_cast<int32_t>(args.n_layers() - args.first_k_dense_replace()),
        0);
    moe_layer_ids.reserve(static_cast<size_t>(moe_layer_count));
    for (int32_t layer_id = args.first_k_dense_replace();
         layer_id < args.n_layers();
         ++layer_id) {
      moe_layer_ids.push_back(layer_id);
    }
    return moe_layer_ids;
  }

  if (!is_qwen_moe) {
    return moe_layer_ids;
  }

  auto mlp_only_layers = args.mlp_only_layers();
  const int32_t stride = std::max<int32_t>(args.decoder_sparse_step(), 1);
  for (int32_t layer = 0; layer < args.n_layers(); ++layer) {
    bool mlp_only =
        std::find(mlp_only_layers.begin(), mlp_only_layers.end(), layer) !=
        mlp_only_layers.end();
    bool is_moe_layer = !mlp_only && ((layer + 1) % stride == 0);
    if (is_moe_layer) {
      moe_layer_ids.push_back(layer);
    }
  }
  return moe_layer_ids;
}

}  // namespace

std::vector<SourceExpertTransferTask>
WeightTransferAlltoallPlanner::build_source_expert_transfer_tasks(
    const RankExpertTransferPlanData& rank_plan,
    const ModelArgs& model_args) const {
  std::vector<SourceExpertTransferTask> tasks;
  std::unordered_map<std::string, size_t> source_to_task_index;
  auto moe_layer_ids = build_moe_layer_ids(model_args);

  for (size_t moe_layer_idx = 0; moe_layer_idx < rank_plan.layer_plans.size();
       ++moe_layer_idx) {
    const auto& layer_plan = rank_plan.layer_plans[moe_layer_idx];
    if (layer_plan.source_experts.empty()) {
      continue;
    }
    if (moe_layer_idx >= moe_layer_ids.size()) {
      LOG(ERROR) << "Expert transfer plan layer index " << moe_layer_idx
                 << " exceeds local MoE layer count " << moe_layer_ids.size();
      return {};
    }
    const int32_t layer_id = moe_layer_ids[moe_layer_idx];
    for (const auto& source_experts : layer_plan.source_experts) {
      if (source_experts.source_addr.empty()) {
        LOG(ERROR) << "Expert transfer source address is empty at MoE layer "
                   << moe_layer_idx;
        return {};
      }
      size_t task_index = 0;
      auto it = source_to_task_index.find(source_experts.source_addr);
      if (it == source_to_task_index.end()) {
        task_index = tasks.size();
        source_to_task_index.emplace(source_experts.source_addr, task_index);
        tasks.push_back(SourceExpertTransferTask{source_experts.source_addr,
                                                 LayerExpertIdsMap{}});
      } else {
        task_index = it->second;
      }

      auto& expert_ids = tasks[task_index].layer_expert_ids[layer_id];
      expert_ids.insert(expert_ids.end(),
                        source_experts.expert_ids.begin(),
                        source_experts.expert_ids.end());
    }
  }

  for (auto& task : tasks) {
    std::vector<int32_t> empty_layer_ids;
    for (auto& [layer_id, expert_ids] : task.layer_expert_ids) {
      std::sort(expert_ids.begin(), expert_ids.end());
      expert_ids.erase(std::unique(expert_ids.begin(), expert_ids.end()),
                       expert_ids.end());
      if (expert_ids.empty()) {
        empty_layer_ids.push_back(layer_id);
      }
    }
    for (int32_t layer_id : empty_layer_ids) {
      task.layer_expert_ids.erase(layer_id);
    }
  }

  tasks.erase(std::remove_if(tasks.begin(),
                             tasks.end(),
                             [](const SourceExpertTransferTask& task) {
                               return task.layer_expert_ids.empty();
                             }),
              tasks.end());
  return tasks;
}

std::vector<int32_t> WeightTransferAlltoallPlanner::collect_sorted_layer_ids(
    const LayerExpertIdsMap& layer_expert_ids) const {
  std::vector<int32_t> layer_ids;
  layer_ids.reserve(layer_expert_ids.size());
  for (const auto& [layer_id, expert_ids] : layer_expert_ids) {
    (void)expert_ids;
    layer_ids.push_back(layer_id);
  }
  std::sort(layer_ids.begin(), layer_ids.end());
  return layer_ids;
}

size_t WeightTransferAlltoallPlanner::count_task_expert_ids(
    const LayerExpertIdsMap& layer_expert_ids) const {
  size_t count = 0;
  for (const auto& [layer_id, expert_ids] : layer_expert_ids) {
    (void)layer_id;
    count += expert_ids.size();
  }
  return count;
}

bool WeightTransferAlltoallPlanner::compare_source_task_by_addr(
    const SourceExpertTransferTask& lhs,
    const SourceExpertTransferTask& rhs) const {
  return lhs.source_addr < rhs.source_addr;
}

bool WeightTransferAlltoallPlanner::build_alltoall_round_plans(
    CausalLM* model,
    const ParallelArgs& parallel_args,
    const std::vector<SourceExpertTransferTask>& source_tasks,
    std::vector<AlltoallRoundPlan>* round_plans) const {
  CHECK(model != nullptr);
  CHECK(round_plans != nullptr);
  round_plans->clear();
  if (source_tasks.empty()) {
    return true;
  }

  size_t total_requested_expert_ids = 0;
  for (const auto& source_task : source_tasks) {
    size_t source_expert_ids =
        count_task_expert_ids(source_task.layer_expert_ids);
    total_requested_expert_ids += source_expert_ids;
    LOG(INFO) << "[AlltoallRoundBuild] source summary. source="
              << source_task.source_addr
              << ", layers=" << source_task.layer_expert_ids.size()
              << ", expert_ids=" << source_expert_ids;
  }

  std::unordered_map<int32_t, std::unordered_map<int32_t, std::string>>
      expert_owner_by_layer;
  for (const auto& source_task : source_tasks) {
    for (const auto& [layer_id, expert_ids] : source_task.layer_expert_ids) {
      for (int32_t expert_id : expert_ids) {
        auto& owner_map = expert_owner_by_layer[layer_id];
        auto it = owner_map.find(expert_id);
        if (it == owner_map.end()) {
          owner_map.emplace(expert_id, source_task.source_addr);
          continue;
        }
        if (it->second != source_task.source_addr) {
          LOG(ERROR) << "Conflict expert ownership at layer " << layer_id
                     << ", expert_id " << expert_id << ", source "
                     << source_task.source_addr << " vs " << it->second;
          return false;
        }
      }
    }
  }

  std::unordered_set<int32_t> requested_layers_set;
  for (const auto& source_task : source_tasks) {
    for (const auto& [layer_id, expert_ids] : source_task.layer_expert_ids) {
      (void)expert_ids;
      requested_layers_set.insert(layer_id);
    }
  }
  std::vector<int32_t> requested_layers(requested_layers_set.begin(),
                                        requested_layers_set.end());
  std::sort(requested_layers.begin(), requested_layers.end());

  auto expert_indices = model->get_expert_weight_indices();
  std::vector<int32_t> sorted_expert_indices(expert_indices.begin(),
                                             expert_indices.end());
  std::sort(sorted_expert_indices.begin(), sorted_expert_indices.end());
  const int32_t ep_rank = get_ep_rank(parallel_args);
  const int32_t ep_size = get_ep_size(parallel_args);
  LOG(INFO) << "[AlltoallRoundBuild] start. sources=" << source_tasks.size()
            << ", requested_layers=" << requested_layers.size()
            << ", requested_expert_ids=" << total_requested_expert_ids
            << ", expert_tensors=" << sorted_expert_indices.size()
            << ", ep_rank=" << ep_rank << ", ep_size=" << ep_size;

  size_t candidate_tensor_count = 0;
  size_t skipped_non_3d_tensor_count = 0;
  size_t tensor_with_run_count = 0;

  for (int32_t layer_id : requested_layers) {
    auto tensors = model->get_decoder_layer_weight(layer_id);
    size_t source_count_for_layer = 0;
    size_t expert_ids_for_layer = 0;
    for (const auto& source_task : source_tasks) {
      auto layer_it = source_task.layer_expert_ids.find(layer_id);
      if (layer_it == source_task.layer_expert_ids.end()) {
        continue;
      }
      ++source_count_for_layer;
      expert_ids_for_layer += layer_it->second.size();
    }
    LOG(INFO) << "[AlltoallRoundBuild] layer start. layer=" << layer_id
              << ", sources=" << source_count_for_layer
              << ", expert_ids=" << expert_ids_for_layer
              << ", tensor_size=" << tensors.size();
    for (int32_t tensor_idx : sorted_expert_indices) {
      ++candidate_tensor_count;
      if (tensor_idx < 0 ||
          tensor_idx >= static_cast<int32_t>(tensors.size())) {
        LOG(ERROR) << "Expert tensor index " << tensor_idx
                   << " is out of range for layer " << layer_id
                   << ", tensor_size=" << tensors.size();
        return false;
      }
      const auto& tensor = tensors[tensor_idx];
      if (tensor.dim() != 3) {
        ++skipped_non_3d_tensor_count;
        VLOG(1) << "Skip non-3D expert tensor in alltoall plan build at layer "
                << layer_id << ", tensor index " << tensor_idx
                << ", dim=" << tensor.dim();
        continue;
      }

      std::vector<std::vector<LocalExpertRun>> runs_per_source(
          source_tasks.size());
      size_t max_runs = 0;
      for (size_t source_idx = 0; source_idx < source_tasks.size();
           ++source_idx) {
        auto it = source_tasks[source_idx].layer_expert_ids.find(layer_id);
        if (it == source_tasks[source_idx].layer_expert_ids.end()) {
          continue;
        }
        if (!build_local_expert_runs(it->second,
                                     tensor.size(0),
                                     ep_rank,
                                     layer_id,
                                     tensor_idx,
                                     &runs_per_source[source_idx])) {
          return false;
        }
        VLOG(1) << "[AlltoallRoundBuild] source runs. layer=" << layer_id
                << ", tensor=" << tensor_idx
                << ", source=" << source_tasks[source_idx].source_addr
                << ", run_count=" << runs_per_source[source_idx].size()
                << ", runs="
                << format_local_runs_limited(runs_per_source[source_idx], 64);
        max_runs = std::max(max_runs, runs_per_source[source_idx].size());
      }

      if (max_runs == 0) {
        VLOG(1) << "[AlltoallRoundBuild] no active run. layer=" << layer_id
                << ", tensor=" << tensor_idx;
        continue;
      }
      ++tensor_with_run_count;
      LOG(INFO) << "[AlltoallRoundBuild] tensor run summary. layer=" << layer_id
                << ", tensor=" << tensor_idx << ", max_runs=" << max_runs
                << ", source_run_counts="
                << format_source_run_counts(source_tasks, runs_per_source);

      for (size_t run_slot = 0; run_slot < max_runs; ++run_slot) {
        AlltoallRoundPlan round_plan;
        round_plan.layer_id = layer_id;
        round_plan.tensor_idx = tensor_idx;
        round_plan.run_slot = static_cast<int32_t>(run_slot);
        round_plan.source_runs.resize(source_tasks.size());
        bool has_active_run = false;
        for (size_t source_idx = 0; source_idx < source_tasks.size();
             ++source_idx) {
          if (run_slot >= runs_per_source[source_idx].size()) {
            continue;
          }
          round_plan.source_runs[source_idx].active = true;
          round_plan.source_runs[source_idx].run =
              runs_per_source[source_idx][run_slot];
          has_active_run = true;
        }
        if (has_active_run) {
          VLOG(1) << "[AlltoallRoundBuild] round generated. layer="
                  << round_plan.layer_id << ", tensor=" << round_plan.tensor_idx
                  << ", run_slot=" << round_plan.run_slot << ", active_sources="
                  << format_round_active_sources(source_tasks, round_plan);
          round_plans->push_back(std::move(round_plan));
        }
      }
    }
  }

  LOG(INFO) << "[AlltoallRoundBuild] done. rounds=" << round_plans->size()
            << ", requested_layers=" << requested_layers.size()
            << ", candidate_tensors=" << candidate_tensor_count
            << ", tensors_with_runs=" << tensor_with_run_count
            << ", skipped_non_3d_tensors=" << skipped_non_3d_tensor_count;
  return true;
}

void WeightTransferAlltoallPlanner::build_alltoall_requests(
    const std::vector<AlltoallRoundPlan>& round_plans,
    const std::vector<SourceExpertTransferTask>& source_tasks,
    uint32_t receiver_rank,
    const std::string& session_id,
    std::vector<xllm::proto::AlltoAllRoundDesc>* receiver_rounds,
    std::vector<xllm::proto::TriggerWeightsSendRequest>* source_requests)
    const {
  CHECK(receiver_rounds != nullptr);
  CHECK(source_requests != nullptr);
  receiver_rounds->clear();
  source_requests->clear();
  source_requests->resize(source_tasks.size());
  for (auto& req : *source_requests) {
    req.set_use_alltoallv(true);
    req.set_receiver_rank(receiver_rank);
    req.set_session_id(session_id);
    req.set_include_non_expert(false);
  }

  for (const auto& round_plan : round_plans) {
    xllm::proto::AlltoAllRoundDesc receiver_desc;
    receiver_desc.set_layer_id(round_plan.layer_id);
    receiver_desc.set_tensor_idx(round_plan.tensor_idx);
    receiver_desc.set_local_expert_start(0);
    receiver_desc.set_local_expert_count(0);
    receiver_rounds->push_back(receiver_desc);

    for (size_t source_idx = 0; source_idx < source_tasks.size();
         ++source_idx) {
      auto* desc = (*source_requests)[source_idx].add_alltoall_rounds();
      desc->set_layer_id(round_plan.layer_id);
      desc->set_tensor_idx(round_plan.tensor_idx);
      if (!round_plan.source_runs[source_idx].active) {
        desc->set_local_expert_start(0);
        desc->set_local_expert_count(0);
        continue;
      }
      desc->set_local_expert_start(
          round_plan.source_runs[source_idx].run.local_expert_start);
      desc->set_local_expert_count(
          round_plan.source_runs[source_idx].run.local_expert_count);
    }
  }
}

}  // namespace xllm
