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

#include "framework/weight_transfer/weight_transfer_common.h"

#include <absl/time/clock.h>
#include <glog/logging.h>

#include <algorithm>
#include <atomic>
#include <iomanip>
#include <limits>
#include <sstream>
#include <unordered_map>
#include <unordered_set>

#include "layers/npu/loader/base_loader.h"

namespace xllm {
namespace {

struct SenderSessionState {
  absl::Time start_time;
  absl::Time last_update_time;
  double overlap_ms = 0.0;
};

struct SenderSessionStartInfo {
  int32_t active_sessions = 0;
  int32_t overlap_with = 0;
  int64_t start_unix_us = 0;
};

struct SenderSessionEndInfo {
  double duration_ms = 0.0;
  double overlap_ms = 0.0;
  double overlap_ratio = 0.0;
  int32_t remaining_sessions = 0;
  int64_t end_unix_us = 0;
};

std::mutex g_sender_session_tracking_mutex;
std::unordered_map<std::string, SenderSessionState> g_active_sender_sessions;

void update_sender_overlap_locked(
    const absl::Time& now,
    std::unordered_map<std::string, SenderSessionState>* active_sessions) {
  CHECK(active_sessions != nullptr);
  if (active_sessions->empty()) {
    return;
  }
  if (active_sessions->size() >= 2) {
    for (auto& entry : *active_sessions) {
      entry.second.overlap_ms +=
          absl::ToDoubleMilliseconds(now - entry.second.last_update_time);
    }
  }
  for (auto& entry : *active_sessions) {
    entry.second.last_update_time = now;
  }
}

SenderSessionStartInfo begin_sender_session_tracking(
    const std::string& session_id) {
  CHECK(!session_id.empty());
  std::lock_guard<std::mutex> lock(g_sender_session_tracking_mutex);
  const absl::Time now = absl::Now();
  update_sender_overlap_locked(now, &g_active_sender_sessions);

  auto existing_it = g_active_sender_sessions.find(session_id);
  if (existing_it != g_active_sender_sessions.end()) {
    LOG(WARNING) << "Sender session already tracked, replacing stale entry. "
                 << "session_id=" << session_id;
    g_active_sender_sessions.erase(existing_it);
  }
  g_active_sender_sessions.emplace(
      session_id, SenderSessionState{now, now, /*overlap_ms=*/0.0});
  const int32_t active_count =
      static_cast<int32_t>(g_active_sender_sessions.size());
  return SenderSessionStartInfo{
      active_count, std::max(0, active_count - 1), absl::ToUnixMicros(now)};
}

SenderSessionEndInfo end_sender_session_tracking(
    const std::string& session_id) {
  CHECK(!session_id.empty());
  std::lock_guard<std::mutex> lock(g_sender_session_tracking_mutex);
  const absl::Time now = absl::Now();
  update_sender_overlap_locked(now, &g_active_sender_sessions);

  auto it = g_active_sender_sessions.find(session_id);
  if (it == g_active_sender_sessions.end()) {
    LOG(WARNING) << "Sender session tracking end without active entry. "
                 << "session_id=" << session_id;
    return SenderSessionEndInfo{
        /*duration_ms=*/0.0,
        /*overlap_ms=*/0.0,
        /*overlap_ratio=*/0.0,
        static_cast<int32_t>(g_active_sender_sessions.size()),
        absl::ToUnixMicros(now)};
  }
  const SenderSessionState state = it->second;
  const double duration_ms = absl::ToDoubleMilliseconds(now - state.start_time);
  const double overlap_ms = state.overlap_ms;
  const double overlap_ratio =
      duration_ms > 0.0 ? overlap_ms / duration_ms : 0.0;
  g_active_sender_sessions.erase(it);
  return SenderSessionEndInfo{
      duration_ms,
      overlap_ms,
      overlap_ratio,
      static_cast<int32_t>(g_active_sender_sessions.size()),
      absl::ToUnixMicros(now)};
}

}  // namespace

LayerStorageInfo get_contiguous_layer_storage_info(
    const std::vector<at::Tensor>& tensors,
    layer::BaseLoader* loader) {
  LayerStorageInfo info;
  if (loader == nullptr || loader->mode() != layer::LoadMode::kManual ||
      loader->uses_rolling_buffer()) {
    return info;
  }

  void* base = loader->get_device_storage();
  uint64_t storage_size = loader->get_storage_size();
  if (base == nullptr || storage_size == 0 || tensors.empty()) {
    return info;
  }

  const uintptr_t base_addr = reinterpret_cast<uintptr_t>(base);
  const uintptr_t end_addr = base_addr + storage_size;
  if (end_addr < base_addr) {
    return info;
  }

  bool has_payload = false;
  for (const auto& tensor : tensors) {
    if (!tensor.defined()) {
      return LayerStorageInfo{};
    }
    const size_t nbytes = tensor.nbytes();
    if (nbytes == 0) {
      return LayerStorageInfo{};
    }
    void* data_ptr = tensor.data_ptr();
    if (data_ptr == nullptr) {
      return LayerStorageInfo{};
    }
    const uintptr_t tensor_addr = reinterpret_cast<uintptr_t>(data_ptr);
    if (tensor_addr < base_addr || tensor_addr + nbytes < tensor_addr ||
        tensor_addr + nbytes > end_addr) {
      return LayerStorageInfo{};
    }
    has_payload = true;
  }

  if (!has_payload) {
    return info;
  }
  info.available = true;
  info.base_ptr = base;
  info.storage_size = storage_size;
  return info;
}

SenderSessionLogGuard::SenderSessionLogGuard(const std::string& session_id,
                                             const std::string& mode)
    : session_id_(session_id), mode_(mode), enabled_(!session_id.empty()) {
  if (!enabled_) {
    return;
  }
  auto start_info = begin_sender_session_tracking(session_id_);
  LOG(INFO) << "[SenderSessionTiming] phase=start"
            << ", mode=" << mode_ << ", session_id=" << session_id_
            << ", start_unix_us=" << start_info.start_unix_us
            << ", active_sessions=" << start_info.active_sessions
            << ", overlap_with=" << start_info.overlap_with;
}

SenderSessionLogGuard::~SenderSessionLogGuard() {
  if (!enabled_) {
    return;
  }
  auto end_info = end_sender_session_tracking(session_id_);
  LOG(INFO) << "[SenderSessionTiming] phase=end"
            << ", mode=" << mode_ << ", session_id=" << session_id_
            << ", success=" << success_
            << ", end_unix_us=" << end_info.end_unix_us
            << ", duration_ms=" << std::fixed << std::setprecision(2)
            << end_info.duration_ms << ", overlap_ms=" << end_info.overlap_ms
            << ", overlap_ratio=" << end_info.overlap_ratio
            << ", active_sessions_after=" << end_info.remaining_sessions;
}

void SenderSessionLogGuard::mark_success() { success_ = true; }

double elapsed_ms_since(const absl::Time& start_time) {
  return absl::ToDoubleMilliseconds(absl::Now() - start_time);
}

void log_weight_pull_timing(const std::string& remote_addr,
                            const WeightPullTimingStats& timing_stats) {
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(2)
      << "[WeightPullTiming] remote_addr=" << remote_addr
      << ", parallel_mode=" << timing_stats.parallel_mode
      << ", expert_sources=" << timing_stats.expert_sources
      << ", success=" << timing_stats.success
      << ", comm_create_ms=" << timing_stats.comm_create_ms
      << ", weight_transfer_ms=" << timing_stats.weight_transfer_ms
      << ", other_ms=" << timing_stats.other_ms
      << ", total_ms=" << timing_stats.total_ms;
  if (!timing_stats.success && !timing_stats.failed_stage.empty()) {
    oss << ", failed_stage=" << timing_stats.failed_stage;
  }
  LOG(INFO) << oss.str();
}

std::string generate_session_id() {
  static std::atomic<uint64_t> counter{0};
  std::ostringstream oss;
  oss << "wt-" << absl::ToUnixMicros(absl::Now()) << "-"
      << counter.fetch_add(1, std::memory_order_relaxed);
  return oss.str();
}

bool get_cached_rpc_endpoint(const std::string& remote_addr,
                             std::shared_ptr<CachedRpcEndpoint>* endpoint_out) {
  CHECK(endpoint_out != nullptr);
  static std::mutex cache_mutex;
  static std::unordered_map<std::string, std::weak_ptr<CachedRpcEndpoint>>
      endpoint_cache;

  {
    std::lock_guard<std::mutex> lock(cache_mutex);
    auto it = endpoint_cache.find(remote_addr);
    if (it != endpoint_cache.end()) {
      auto cached = it->second.lock();
      if (cached != nullptr && cached->channel != nullptr &&
          cached->stub != nullptr) {
        *endpoint_out = std::move(cached);
        return true;
      }
    }
  }

  auto endpoint = std::make_shared<CachedRpcEndpoint>();
  endpoint->channel = std::make_shared<brpc::Channel>();
  brpc::ChannelOptions options;
  options.timeout_ms = 10000;
  options.connect_timeout_ms = 2000;
  options.max_retry = 3;
  if (endpoint->channel->Init(remote_addr.c_str(), &options) != 0) {
    LOG(ERROR) << "BRPC Channel init failed for " << remote_addr;
    return false;
  }
  endpoint->stub = std::make_shared<xllm::proto::WeightTransferService_Stub>(
      endpoint->channel.get());

  {
    std::lock_guard<std::mutex> lock(cache_mutex);
    endpoint_cache[remote_addr] = endpoint;
  }
  *endpoint_out = std::move(endpoint);
  return true;
}

std::shared_ptr<const std::unordered_set<int32_t>>
get_cached_expert_indices_set(CausalLM* model) {
  CHECK(model != nullptr);
  static std::mutex cache_mutex;
  static std::unordered_map<CausalLM*,
                            std::weak_ptr<const std::unordered_set<int32_t>>>
      expert_indices_cache;

  {
    std::lock_guard<std::mutex> lock(cache_mutex);
    auto it = expert_indices_cache.find(model);
    if (it != expert_indices_cache.end()) {
      auto cached = it->second.lock();
      if (cached != nullptr) {
        return cached;
      }
    }
  }

  auto expert_indices = model->get_expert_weight_indices();
  auto cached_set = std::make_shared<std::unordered_set<int32_t>>(
      expert_indices.begin(), expert_indices.end());
  {
    std::lock_guard<std::mutex> lock(cache_mutex);
    expert_indices_cache[model] = cached_set;
  }
  return cached_set;
}

bool init_weight_transfer_stub(
    const std::string& remote_addr,
    std::unique_ptr<brpc::Channel>* channel,
    std::unique_ptr<xllm::proto::WeightTransferService_Stub>* stub) {
  CHECK(channel != nullptr);
  CHECK(stub != nullptr);
  auto tmp_channel = std::make_unique<brpc::Channel>();
  brpc::ChannelOptions options;
  options.timeout_ms = 10000;
  options.connect_timeout_ms = 2000;
  options.max_retry = 3;
  if (tmp_channel->Init(remote_addr.c_str(), &options) != 0) {
    LOG(ERROR) << "BRPC Channel init failed for " << remote_addr;
    return false;
  }
  auto tmp_stub = std::make_unique<xllm::proto::WeightTransferService_Stub>(
      tmp_channel.get());
  *stub = std::move(tmp_stub);
  *channel = std::move(tmp_channel);
  return true;
}

bool call_init_comm_with_retry(xllm::proto::WeightTransferService_Stub* stub,
                               const xllm::proto::InitCommRequest& req,
                               const std::string& remote_addr) {
  CHECK(stub != nullptr);
  constexpr int64_t k_retry_deadline_ms = 30000;
  constexpr int32_t k_initial_backoff_ms = 20;
  constexpr int32_t k_max_backoff_ms = 500;
  const absl::Time deadline =
      absl::Now() + absl::Milliseconds(k_retry_deadline_ms);
  int32_t backoff_ms = k_initial_backoff_ms;
  int attempt = 0;
  while (absl::Now() < deadline) {
    ++attempt;
    brpc::Controller cntl;
    xllm::proto::InitCommResponse resp;
    stub->InitComm(&cntl, &req, &resp, nullptr);
    if (!cntl.Failed()) {
      if (resp.success()) {
        LOG(INFO) << "Receiver: InitComm succeeded for source " << remote_addr;
        return true;
      }
      LOG(ERROR) << "Receiver: InitComm logic failure from " << remote_addr;
      return false;
    }
    if (attempt % 10 == 0) {
      LOG(WARNING) << "Receiver: Waiting for source " << remote_addr
                   << " to serve InitComm... (Attempt " << attempt << ")";
    }
    absl::SleepFor(absl::Milliseconds(backoff_ms));
    backoff_ms = std::min(backoff_ms * 2, k_max_backoff_ms);
  }
  LOG(ERROR) << "Receiver: Timeout waiting InitComm for source " << remote_addr
             << ", timeout_ms=" << k_retry_deadline_ms
             << ", attempts=" << attempt;
  return false;
}

std::vector<int32_t> normalize_expert_ids(
    const std::vector<int32_t>& expert_ids) {
  std::vector<int32_t> normalized = expert_ids;
  std::sort(normalized.begin(), normalized.end());
  normalized.erase(std::unique(normalized.begin(), normalized.end()),
                   normalized.end());
  return normalized;
}

LayerExpertIdsMap normalize_layer_expert_ids_map(
    const LayerExpertIdsMap& input) {
  LayerExpertIdsMap normalized;
  for (const auto& [layer_id, expert_ids] : input) {
    auto sorted_ids = normalize_expert_ids(expert_ids);
    if (!sorted_ids.empty()) {
      normalized.emplace(layer_id, std::move(sorted_ids));
    }
  }
  return normalized;
}

size_t count_expert_ids(const LayerExpertIdsMap& layer_expert_ids) {
  size_t count = 0;
  for (const auto& [layer_id, expert_ids] : layer_expert_ids) {
    (void)layer_id;
    count += expert_ids.size();
  }
  return count;
}

bool validate_layer_expert_ids_map(const std::vector<int32_t>& layer_ids,
                                   const LayerExpertIdsMap& layer_expert_ids,
                                   const std::string& stage_name) {
  std::unordered_set<int32_t> layer_id_set(layer_ids.begin(), layer_ids.end());
  for (const auto& [layer_id, expert_ids] : layer_expert_ids) {
    (void)expert_ids;
    if (layer_id_set.count(layer_id) == 0) {
      LOG(ERROR) << stage_name
                 << " has expert ids for unknown layer id: " << layer_id;
      return false;
    }
  }
  return true;
}

int32_t get_ep_size(const ParallelArgs& parallel_args) {
  return std::max(parallel_args.ep_size(), 1);
}

int32_t get_ep_rank(const ParallelArgs& parallel_args) {
  int32_t ep_size = get_ep_size(parallel_args);
  int32_t ep_rank = parallel_args.rank() % ep_size;
  if (ep_rank < 0) {
    ep_rank += ep_size;
  }
  return ep_rank;
}

bool append_layer_transfer_items(
    const std::vector<at::Tensor>& tensors,
    int32_t layer_id,
    const std::vector<int32_t>& expert_ids,
    const std::unordered_set<int32_t>& expert_indices_set,
    bool include_non_expert,
    bool transfer_all_experts,
    int32_t ep_rank,
    int32_t ep_size,
    decltype(HCCL_SEND) operation,
    uint32_t peer_rank,
    std::vector<HcclSendRecvItem>* items,
    size_t* total_nbytes) {
  CHECK(items != nullptr);
  CHECK(total_nbytes != nullptr);

  for (size_t tensor_idx = 0; tensor_idx < tensors.size(); ++tensor_idx) {
    const auto& tensor = tensors[tensor_idx];
    size_t nbytes = tensor.nbytes();
    bool is_expert_tensor =
        expert_indices_set.count(static_cast<int32_t>(tensor_idx)) > 0;

    if (is_expert_tensor) {
      if (tensor.dim() != 3) {
        if (!expert_ids.empty() && !transfer_all_experts &&
            !include_non_expert) {
          VLOG(1)
              << "Skip non-3D expert tensor in selective expert transfer at "
              << "layer " << layer_id << ", tensor index " << tensor_idx
              << ", dim=" << tensor.dim();
          continue;
        }
        if (include_non_expert) {
          items->push_back({operation,
                            tensor.data_ptr(),
                            static_cast<uint64_t>(nbytes),
                            HCCL_DATA_TYPE_UINT8,
                            peer_rank});
          *total_nbytes += nbytes;
        }
        continue;
      }

      int64_t local_expert_num = tensor.size(0);
      if (local_expert_num <= 0) {
        LOG(ERROR) << "Invalid local expert count at layer " << layer_id
                   << ", tensor index " << tensor_idx
                   << ", expert_num=" << local_expert_num;
        return false;
      }
      size_t expert_nbytes = nbytes / local_expert_num;

      if (transfer_all_experts) {
        for (int64_t local_expert_idx = 0; local_expert_idx < local_expert_num;
             ++local_expert_idx) {
          void* data_ptr = static_cast<uint8_t*>(tensor.data_ptr()) +
                           local_expert_idx * expert_nbytes;
          items->push_back({operation,
                            data_ptr,
                            static_cast<uint64_t>(expert_nbytes),
                            HCCL_DATA_TYPE_UINT8,
                            peer_rank});
          *total_nbytes += expert_nbytes;
        }
        continue;
      }

      if (expert_ids.empty()) {
        continue;
      }

      int64_t local_expert_begin =
          static_cast<int64_t>(ep_rank) * local_expert_num;
      int64_t local_expert_end = local_expert_begin + local_expert_num;
      std::vector<int64_t> local_expert_indices;
      local_expert_indices.reserve(expert_ids.size());
      for (int32_t expert_id : expert_ids) {
        if (expert_id < local_expert_begin || expert_id >= local_expert_end) {
          LOG(ERROR) << "Expert id " << expert_id
                     << " is outside local expert slice [" << local_expert_begin
                     << ", " << local_expert_end << ") at layer " << layer_id
                     << ", ep_rank=" << ep_rank << ", ep_size=" << ep_size;
          return false;
        }
        local_expert_indices.push_back(expert_id - local_expert_begin);
      }
      if (local_expert_indices.empty()) {
        continue;
      }

      int64_t run_start = local_expert_indices.front();
      int64_t run_prev = local_expert_indices.front();
      auto flush_run = [&](int64_t start_idx, int64_t end_idx) {
        int64_t run_count = end_idx - start_idx + 1;
        void* data_ptr = static_cast<uint8_t*>(tensor.data_ptr()) +
                         start_idx * expert_nbytes;
        items->push_back({operation,
                          data_ptr,
                          static_cast<uint64_t>(run_count * expert_nbytes),
                          HCCL_DATA_TYPE_UINT8,
                          peer_rank});
        *total_nbytes += run_count * expert_nbytes;
      };
      for (size_t idx = 1; idx < local_expert_indices.size(); ++idx) {
        int64_t cur = local_expert_indices[idx];
        if (cur == run_prev + 1) {
          run_prev = cur;
          continue;
        }
        flush_run(run_start, run_prev);
        run_start = cur;
        run_prev = cur;
      }
      flush_run(run_start, run_prev);
      continue;
    }

    if (!include_non_expert) {
      continue;
    }

    items->push_back({operation,
                      tensor.data_ptr(),
                      static_cast<uint64_t>(nbytes),
                      HCCL_DATA_TYPE_UINT8,
                      peer_rank});
    *total_nbytes += nbytes;
  }

  return true;
}

void append_contiguous_storage_transfer_items(
    void* base_ptr,
    uint64_t storage_size,
    decltype(HCCL_SEND) operation,
    uint32_t peer_rank,
    uint64_t chunk_bytes,
    std::vector<HcclSendRecvItem>* items,
    size_t* total_nbytes) {
  CHECK(base_ptr != nullptr);
  CHECK(items != nullptr);
  CHECK(total_nbytes != nullptr);
  if (storage_size == 0) {
    return;
  }
  const uint64_t effective_chunk =
      chunk_bytes == 0 ? storage_size : chunk_bytes;
  uint64_t offset = 0;
  while (offset < storage_size) {
    const uint64_t bytes =
        std::min<uint64_t>(effective_chunk, storage_size - offset);
    void* data_ptr = static_cast<uint8_t*>(base_ptr) + offset;
    items->push_back(
        {operation, data_ptr, bytes, HCCL_DATA_TYPE_UINT8, peer_rank});
    *total_nbytes += bytes;
    offset += bytes;
  }
}

uint64_t sum_tensor_nbytes(const std::vector<at::Tensor>& tensors) {
  uint64_t total = 0;
  for (const auto& tensor : tensors) {
    if (!tensor.defined()) {
      continue;
    }
    total += static_cast<uint64_t>(tensor.nbytes());
  }
  return total;
}

std::string summarize_hccl_transfer_items(
    const std::vector<HcclSendRecvItem>& items) {
  if (items.empty()) {
    return "items=0, item_min_mb=0.00, item_max_mb=0.00, item_avg_mb=0.00";
  }

  uint64_t min_bytes = std::numeric_limits<uint64_t>::max();
  uint64_t max_bytes = 0;
  uint64_t total_bytes = 0;
  for (const auto& item : items) {
    min_bytes = std::min(min_bytes, item.count);
    max_bytes = std::max(max_bytes, item.count);
    total_bytes += item.count;
  }

  constexpr double kMiB = 1024.0 * 1024.0;
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(2) << "items=" << items.size()
      << ", item_min_mb=" << (static_cast<double>(min_bytes) / kMiB)
      << ", item_max_mb=" << (static_cast<double>(max_bytes) / kMiB)
      << ", item_avg_mb="
      << (static_cast<double>(total_bytes) / static_cast<double>(items.size()) /
          kMiB);
  return oss.str();
}

std::string summarize_hccl_transfer_item_addresses(
    const std::vector<HcclSendRecvItem>& items) {
  if (items.empty()) {
    return "addr_align64=0/0, addr_align512=0/0, addr_align4k=0/0, "
           "size_align512=0/0, adjacent_items=0, start_2mb_pages=0, "
           "addr_span_mb=0.00";
  }

  size_t addr_align64 = 0;
  size_t addr_align512 = 0;
  size_t addr_align4k = 0;
  size_t size_align512 = 0;
  size_t adjacent_items = 0;
  uintptr_t min_addr = std::numeric_limits<uintptr_t>::max();
  uintptr_t max_end = 0;
  std::unordered_set<uintptr_t> start_2mb_pages;
  start_2mb_pages.reserve(items.size());
  uintptr_t prev_end = 0;

  for (const auto& item : items) {
    const uintptr_t addr = reinterpret_cast<uintptr_t>(item.buf);
    const uintptr_t end = addr + static_cast<uintptr_t>(item.count);
    if (addr % 64 == 0) {
      ++addr_align64;
    }
    if (addr % 512 == 0) {
      ++addr_align512;
    }
    if (addr % 4096 == 0) {
      ++addr_align4k;
    }
    if (item.count % 512 == 0) {
      ++size_align512;
    }
    if (prev_end != 0 && addr == prev_end) {
      ++adjacent_items;
    }
    prev_end = end;
    min_addr = std::min(min_addr, addr);
    max_end = std::max(max_end, end);
    start_2mb_pages.insert(addr >> 21);
  }

  constexpr double kMiB = 1024.0 * 1024.0;
  const double span_mb =
      max_end > min_addr ? static_cast<double>(max_end - min_addr) / kMiB : 0.0;
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(2) << "addr_align64=" << addr_align64
      << "/" << items.size() << ", addr_align512=" << addr_align512 << "/"
      << items.size() << ", addr_align4k=" << addr_align4k << "/"
      << items.size() << ", size_align512=" << size_align512 << "/"
      << items.size() << ", adjacent_items=" << adjacent_items
      << ", start_2mb_pages=" << start_2mb_pages.size()
      << ", addr_span_mb=" << span_mb;
  return oss.str();
}

void fill_trigger_weights_send_request(
    const std::vector<int32_t>& layer_ids,
    const LayerExpertIdsMap& layer_expert_ids,
    bool include_non_expert,
    xllm::proto::TriggerWeightsSendRequest* req) {
  CHECK(req != nullptr);
  for (int32_t layer_id : layer_ids) {
    req->add_layer_ids(layer_id);
  }
  req->set_include_non_expert(include_non_expert);
  for (int32_t layer_id : layer_ids) {
    auto* layer_expert_entry = req->add_layer_expert_ids();
    layer_expert_entry->set_layer_id(layer_id);
    auto it = layer_expert_ids.find(layer_id);
    if (it == layer_expert_ids.end()) {
      continue;
    }
    layer_expert_entry->mutable_expert_ids()->Reserve(it->second.size());
    for (int32_t expert_id : it->second) {
      layer_expert_entry->add_expert_ids(expert_id);
    }
  }
}

LayerExpertIdsMap parse_layer_expert_ids(
    const xllm::proto::TriggerWeightsSendRequest& request) {
  LayerExpertIdsMap parsed;
  for (const auto& layer_expert_ids : request.layer_expert_ids()) {
    auto& ids = parsed[layer_expert_ids.layer_id()];
    ids.insert(ids.end(),
               layer_expert_ids.expert_ids().begin(),
               layer_expert_ids.expert_ids().end());
  }
  return normalize_layer_expert_ids_map(parsed);
}

uint64_t get_expert_unit_bytes(const at::Tensor& tensor,
                               int32_t layer_id,
                               int32_t tensor_idx) {
  if (tensor.dim() != 3) {
    return 0;
  }
  int64_t local_expert_num = tensor.size(0);
  if (local_expert_num <= 0) {
    LOG(ERROR) << "Invalid local expert number " << local_expert_num
               << " at layer " << layer_id << ", tensor " << tensor_idx;
    return 0;
  }
  if (tensor.nbytes() % local_expert_num != 0) {
    LOG(ERROR) << "Expert tensor bytes " << tensor.nbytes()
               << " not divisible by local experts " << local_expert_num
               << " at layer " << layer_id << ", tensor " << tensor_idx;
    return 0;
  }
  return static_cast<uint64_t>(tensor.nbytes() / local_expert_num);
}

}  // namespace xllm
