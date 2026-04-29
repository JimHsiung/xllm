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

#include "framework/weight_transfer/weight_transfer_session_manager.h"

#include <glog/logging.h>

#include <thread>

namespace xllm {

WeightTransferSessionManager::WeightTransferSessionManager(int32_t device_id)
    : device_id_(device_id) {
  hccl_thread_pool_ = std::make_shared<ThreadPool>();
  rpc_thread_pool_ = std::make_shared<ThreadPool>();
  const unsigned int hw_threads = std::thread::hardware_concurrency();
  const size_t bg_threads =
      std::max<size_t>(4, hw_threads == 0 ? 4 : hw_threads);
  background_thread_pool_ = std::make_shared<ThreadPool>(bg_threads);
}

void WeightTransferSessionManager::schedule_hccl_task(
    ThreadPool::Runnable runnable) {
  if (runnable == nullptr) {
    return;
  }
  if (hccl_thread_pool_ == nullptr) {
    runnable();
    return;
  }
  hccl_thread_pool_->schedule(std::move(runnable));
}

void WeightTransferSessionManager::schedule_rpc_task(
    ThreadPool::Runnable runnable) {
  if (runnable == nullptr) {
    return;
  }
  if (rpc_thread_pool_ == nullptr) {
    runnable();
    return;
  }
  rpc_thread_pool_->schedule(std::move(runnable));
}

void WeightTransferSessionManager::schedule_background_task(
    ThreadPool::Runnable runnable) {
  if (runnable == nullptr) {
    return;
  }
  if (background_thread_pool_ == nullptr) {
    runnable();
    return;
  }
  background_thread_pool_->schedule(std::move(runnable));
}

std::shared_ptr<CommSessionContext>
WeightTransferSessionManager::create_session_context(
    const std::string& session_id,
    xllm::proto::CommMode comm_mode) {
  if (session_id.empty()) {
    LOG(ERROR) << "Session id should not be empty.";
    return nullptr;
  }
  destroy_session_context_async(session_id, true);

  auto session_ctx = std::make_shared<CommSessionContext>();
  session_ctx->session_id = session_id;
  session_ctx->comm_mode = comm_mode;

  aclrtSetDevice(device_id_);
  auto stream_ret = aclrtCreateStream(&session_ctx->stream);
  if (stream_ret != ACL_SUCCESS) {
    LOG(ERROR) << "Failed to create stream for session " << session_id
               << ", ret=" << stream_ret;
    return nullptr;
  }

  {
    std::lock_guard<std::mutex> lock(session_contexts_mutex_);
    session_contexts_[session_id] = session_ctx;
  }
  return session_ctx;
}

std::shared_ptr<CommSessionContext>
WeightTransferSessionManager::get_session_context(
    const std::string& session_id) {
  std::lock_guard<std::mutex> lock(session_contexts_mutex_);
  auto it = session_contexts_.find(session_id);
  if (it == session_contexts_.end()) {
    return nullptr;
  }
  return it->second;
}

bool WeightTransferSessionManager::destroy_session_context_resources(
    const std::shared_ptr<CommSessionContext>& session_ctx,
    bool log_if_empty) {
  if (session_ctx == nullptr) {
    return false;
  }

  bool has_comm = (session_ctx->hccl_comm != nullptr);
  bool has_stream = (session_ctx->stream != nullptr);
  bool has_rpc_resource =
      (session_ctx->channel != nullptr || session_ctx->stub != nullptr);
  bool initialized = session_ctx->is_comm_initialized.load();
  if (!has_comm && !has_stream && !has_rpc_resource && !initialized) {
    if (log_if_empty) {
      LOG(INFO) << "No active session resources to reset. session_id="
                << session_ctx->session_id;
    }
    return false;
  }

  if (has_comm) {
    aclrtSetDevice(device_id_);
    auto ret = HcclCommDestroy(session_ctx->hccl_comm);
    if (ret != HCCL_SUCCESS) {
      LOG(WARNING) << "HcclCommDestroy failed while resetting session "
                   << session_ctx->session_id << ", ret=" << ret;
    }
  }
  if (has_stream) {
    aclrtSetDevice(device_id_);
    auto ret = aclrtDestroyStream(session_ctx->stream);
    if (ret != ACL_SUCCESS) {
      LOG(WARNING) << "aclrtDestroyStream failed while resetting session "
                   << session_ctx->session_id << ", ret=" << ret;
    }
  }

  session_ctx->hccl_comm = nullptr;
  session_ctx->stream = nullptr;
  session_ctx->stub.reset();
  session_ctx->channel.reset();
  session_ctx->is_comm_initialized.store(false);
  session_ctx->n_ranks = 0;
  session_ctx->rank = 0;
  session_ctx->comm_mode = xllm::proto::COMM_MODE_P2P;
  return true;
}

void WeightTransferSessionManager::destroy_session_context(
    const std::string& session_id,
    bool erase_context) {
  destroy_session_context_async(session_id, erase_context);
}

void WeightTransferSessionManager::destroy_session_context_async(
    const std::string& session_id,
    bool erase_context) {
  std::shared_ptr<CommSessionContext> session_ctx;
  {
    std::lock_guard<std::mutex> lock(session_contexts_mutex_);
    auto it = session_contexts_.find(session_id);
    if (it == session_contexts_.end()) {
      return;
    }
    session_ctx = it->second;
    if (erase_context) {
      session_contexts_.erase(it);
    }
  }
  schedule_background_task([this, session_ctx]() {
    destroy_session_context_resources(session_ctx, false);
  });
}

void WeightTransferSessionManager::destroy_all_session_contexts() {
  std::vector<std::shared_ptr<CommSessionContext>> contexts;
  {
    std::lock_guard<std::mutex> lock(session_contexts_mutex_);
    for (const auto& [session_id, session_ctx] : session_contexts_) {
      (void)session_id;
      contexts.push_back(session_ctx);
    }
    session_contexts_.clear();
  }
  for (auto& session_ctx : contexts) {
    destroy_session_context_resources(session_ctx, false);
  }
}

bool WeightTransferSessionManager::wait_for_session_ready(
    const std::shared_ptr<CommSessionContext>& session_ctx,
    const std::string& stage_name) {
  if (session_ctx == nullptr) {
    LOG(ERROR) << stage_name << ": Session context is null.";
    return false;
  }
  int wait_retry = 0;
  while (!session_ctx->is_comm_initialized.load()) {
    if (wait_retry % 100 == 0) {
      LOG(WARNING) << stage_name << ": Waiting for HCCL init. session_id="
                   << session_ctx->session_id;
    }
    usleep(10000);
    ++wait_retry;
    if (wait_retry > 2000) {
      LOG(ERROR) << stage_name << ": Timeout waiting for HCCL init. session_id="
                 << session_ctx->session_id;
      return false;
    }
  }
  return true;
}

}  // namespace xllm
