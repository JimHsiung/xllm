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

#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

#include "framework/weight_transfer/weight_transfer_types.h"
#include "util/threadpool.h"

namespace xllm {

class WeightTransferSessionManager {
 public:
  explicit WeightTransferSessionManager(int32_t device_id);

  void schedule_hccl_task(ThreadPool::Runnable runnable);
  void schedule_rpc_task(ThreadPool::Runnable runnable);
  void schedule_background_task(ThreadPool::Runnable runnable);

  std::shared_ptr<CommSessionContext> create_session_context(
      const std::string& session_id,
      xllm::proto::CommMode comm_mode);
  std::shared_ptr<CommSessionContext> get_session_context(
      const std::string& session_id);
  void destroy_session_context(const std::string& session_id,
                               bool erase_context);
  void destroy_session_context_async(const std::string& session_id,
                                     bool erase_context);
  void destroy_all_session_contexts();
  bool destroy_session_context_resources(
      const std::shared_ptr<CommSessionContext>& session_ctx,
      bool log_if_empty);
  bool wait_for_session_ready(
      const std::shared_ptr<CommSessionContext>& session_ctx,
      const std::string& stage_name);

 private:
  int32_t device_id_;
  std::mutex session_contexts_mutex_;
  std::unordered_map<std::string, std::shared_ptr<CommSessionContext>>
      session_contexts_;

  std::shared_ptr<ThreadPool> hccl_thread_pool_;
  std::shared_ptr<ThreadPool> rpc_thread_pool_;
  std::shared_ptr<ThreadPool> background_thread_pool_;
};

}  // namespace xllm
