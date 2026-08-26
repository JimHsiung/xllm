/* Copyright 2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/xLLM-AI/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "runtime/prepared_task/prepared_task_pipeline.h"

#include <glog/logging.h>

#include <atomic>
#include <condition_variable>
#include <deque>
#include <future>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

#include "common/metrics.h"
#if defined(USE_NPU)
#include "common/mspti_helper.h"
#endif

namespace xllm {

class PreparedTaskPipeline::Impl final {
 public:
  Impl(std::unique_ptr<PreparedTaskAdapter> adapter, int32_t slot_count)
      : adapter_(std::move(adapter)), slots_(slot_count) {
    CHECK(adapter_ != nullptr);
    CHECK(slot_count == 1 || slot_count == 2)
        << "PreparedTaskPipeline supports one synchronous slot or two "
           "asynchronous slots";
    for (int32_t slot_id = 0; slot_id < slot_count; ++slot_id) {
      slots_[slot_id].slot_id = slot_id;
      free_slots_.emplace_back(slot_id);
    }
    state_thread_ = std::thread([this]() { state_loop(); });
    launch_thread_ = std::thread([this]() { launch_loop(); });
    lifecycle_ = PreparedPipelineLifecycle::RUNNING;
  }

  ~Impl() { shutdown(); }

  folly::SemiFuture<std::optional<ForwardOutput>> submit(
      const ForwardInput& input) {
    ForwardInput unpacked_input;
    const ForwardInput* prepared_input = &input;
    if (input.input_host_buffer_has_layout) {
      CHECK(detail::unpack_from_input_host_buffer(
          input, torch::Device(torch::kCPU), unpacked_input))
          << "Failed to unpack Prepared ForwardInput Host payload";
      prepared_input = &unpacked_input;
    }

    auto output_promise =
        std::make_shared<folly::Promise<std::optional<ForwardOutput>>>();
    folly::SemiFuture<std::optional<ForwardOutput>> output_future =
        output_promise->getSemiFuture();
    auto prepare_ack = std::make_shared<std::promise<void>>();
    std::future<void> prepare_future = prepare_ack->get_future();

    {
      std::unique_lock<std::mutex> lock(lifecycle_mutex_);
      CHECK(lifecycle_ == PreparedPipelineLifecycle::RUNNING)
          << "PreparedTaskPipeline is not accepting tasks";
      CHECK_LT(in_flight_tasks_, slots_.size())
          << "PreparedTaskPipeline only supports one-step scheduling ahead";
      ++in_flight_tasks_;
    }

    {
      std::lock_guard<std::mutex> lock(state_mutex_);
      CHECK(!stop_requested_);
      state_commands_.emplace_back(StateCommand{StateCommandKind::PREPARE,
                                                prepared_input,
                                                std::move(prepare_ack),
                                                output_promise});
    }
    state_cv_.notify_one();

    // The caller owns ForwardInput. Do not return until Prepare has copied all
    // CPU metadata and submitted H2D from slot-owned pinned storage.
    prepare_future.get();
    return output_future;
  }

  folly::SemiFuture<std::optional<ForwardOutput>> get_last_step_result() {
    CHECK(async_mode())
        << "Only the two-slot PreparedTaskPipeline exposes last-step output";
    auto output_promise =
        std::make_shared<folly::Promise<std::optional<ForwardOutput>>>();
    folly::SemiFuture<std::optional<ForwardOutput>> output_future =
        output_promise->getSemiFuture();
    {
      std::lock_guard<std::mutex> lock(lifecycle_mutex_);
      CHECK(lifecycle_ == PreparedPipelineLifecycle::RUNNING ||
            lifecycle_ == PreparedPipelineLifecycle::QUIESCENT)
          << "PreparedTaskPipeline cannot consume output after shutdown";
      CHECK_LT(pending_result_requests_, in_flight_tasks_)
          << "No unclaimed Prepared task output is in flight";
      ++pending_result_requests_;
    }
    {
      std::lock_guard<std::mutex> lock(state_mutex_);
      CHECK(!stop_requested_);
      state_commands_.emplace_back(StateCommand{
          StateCommandKind::CONSUME, nullptr, nullptr, output_promise});
    }
    state_cv_.notify_one();
    return output_future;
  }

  void quiesce() {
    std::unique_lock<std::mutex> lock(lifecycle_mutex_);
    if (lifecycle_ == PreparedPipelineLifecycle::QUIESCENT ||
        lifecycle_ == PreparedPipelineLifecycle::STOPPED) {
      return;
    }
    CHECK(lifecycle_ == PreparedPipelineLifecycle::RUNNING);
    lifecycle_ = PreparedPipelineLifecycle::QUIESCENT;
    lifecycle_cv_.wait(lock, [this]() { return in_flight_tasks_ == 0; });
  }

  void resume() {
    std::lock_guard<std::mutex> lock(lifecycle_mutex_);
    CHECK(lifecycle_ == PreparedPipelineLifecycle::QUIESCENT);
    CHECK(!stop_requested_);
    lifecycle_ = PreparedPipelineLifecycle::RUNNING;
  }

  void shutdown() {
    {
      std::lock_guard<std::mutex> lock(lifecycle_mutex_);
      if (lifecycle_ == PreparedPipelineLifecycle::STOPPED) {
        return;
      }
    }
    quiesce();
    {
      std::lock_guard<std::mutex> lock(lifecycle_mutex_);
      stop_requested_ = true;
      lifecycle_ = PreparedPipelineLifecycle::STOPPED;
    }
    state_cv_.notify_all();
    ready_cv_.notify_all();
    completed_cv_.notify_all();
    if (state_thread_.joinable()) {
      state_thread_.join();
    }
    if (launch_thread_.joinable()) {
      launch_thread_.join();
    }
  }

  PreparedPipelineLifecycle lifecycle() const {
    std::lock_guard<std::mutex> lock(lifecycle_mutex_);
    return lifecycle_;
  }

  int32_t slot_count() const { return static_cast<int32_t>(slots_.size()); }

 private:
  enum class StateCommandKind : int8_t {
    PREPARE = 0,
    CONSUME,
  };

  struct StateCommand {
    StateCommandKind kind = StateCommandKind::PREPARE;
    const ForwardInput* input = nullptr;
    std::shared_ptr<std::promise<void>> prepare_ack;
    std::shared_ptr<folly::Promise<std::optional<ForwardOutput>>>
        output_promise;
  };

  bool async_mode() const { return slots_.size() == 2; }

  void state_loop() {
    adapter_->initialize_state_thread();
    while (true) {
      StateCommand command;
      {
        std::unique_lock<std::mutex> lock(state_mutex_);
        state_cv_.wait(lock, [this]() {
          return !state_commands_.empty() || stop_requested_;
        });
        if (state_commands_.empty() && stop_requested_) {
          return;
        }
        command = std::move(state_commands_.front());
        state_commands_.pop_front();
      }
      if (command.kind == StateCommandKind::PREPARE) {
        prepare(std::move(command));
      } else {
        consume_next(std::move(command.output_promise));
      }
    }
  }

  void prepare(StateCommand command) {
    CHECK(command.input != nullptr);
    CHECK(command.prepare_ack != nullptr);
    CHECK(command.output_promise != nullptr);
    CHECK(!free_slots_.empty());
    const int32_t slot_id = free_slots_.front();
    free_slots_.pop_front();
    ExecutionSlot& slot = slots_[slot_id];
    CHECK(slot.state == ExecutionSlotState::FREE);
    wait_for_successor_state_read(slot);
    slot.state = ExecutionSlotState::PREPARING;
    adapter_->prepare_slot_for_reuse(slot);
    slot.predecessor_slot_id.reset();
    slot.successor_consumes_state = false;
    slot.successor_read_enqueued = false;
    slot.reuse_fence.reset();
    slot.task_seq_no = next_task_seq_no_++;
    slot.task_kind = adapter_->classify(*command.input);
    if (adapter_->consumes_predecessor_state(*command.input) &&
        last_prepared_slot_id_.has_value()) {
      slot.predecessor_slot_id = last_prepared_slot_id_;
      ExecutionSlot& predecessor_slot = slots_[*last_prepared_slot_id_];
      if (predecessor_slot.slot_id != slot.slot_id) {
        CHECK(!predecessor_slot.successor_consumes_state)
            << "Prepared Slot already has a successor DeviceStepState reader";
        predecessor_slot.successor_consumes_state = true;
        predecessor_slot.successor_read_enqueued = false;
      }
    }
    last_prepared_slot_id_ = slot.slot_id;
    Timer prepare_timer;
    {
#if defined(USE_NPU)
      LLM_MSTX_HOST_RANGE("xllm.PreparedTask.Prepare");
#endif
      adapter_->prepare(*command.input, slot);
    }
    HISTOGRAM_OBSERVE(
        prepared_task_prepare_cpu_latency_microseconds,
        static_cast<int64_t>(prepare_timer.elapsed_microseconds()));
    slot.state = ExecutionSlotState::READY;

    {
      std::lock_guard<std::mutex> lock(ready_mutex_);
      CHECK_LT(ready_slots_.size(), slots_.size());
      ready_slots_.emplace_back(slot.slot_id);
    }
    ready_cv_.notify_one();
    if (async_mode()) {
      command.output_promise->setValue(std::nullopt);
    }
    command.prepare_ack->set_value();

    if (!async_mode()) {
      consume_next(std::move(command.output_promise));
    }
  }

  void consume_next(
      std::shared_ptr<folly::Promise<std::optional<ForwardOutput>>>
          output_promise) {
    CHECK(output_promise != nullptr);
    int32_t completed_slot_id = 0;
    {
      std::unique_lock<std::mutex> lock(completed_mutex_);
      completed_cv_.wait(lock, [this]() {
        return !completed_slots_.empty() || stop_requested_;
      });
      CHECK(!completed_slots_.empty());
      completed_slot_id = completed_slots_.front();
      completed_slots_.pop_front();
    }
    ExecutionSlot& slot = slots_[completed_slot_id];
    CHECK_EQ(completed_slot_id, slot.slot_id);
    CHECK(slot.state == ExecutionSlotState::COMPLETED);
    slot.state = ExecutionSlotState::CONSUMING;
    Timer consume_timer;
    std::optional<ForwardOutput> output;
    {
#if defined(USE_NPU)
      LLM_MSTX_HOST_RANGE("xllm.PreparedTask.Consume");
#endif
      output = adapter_->consume(slot);
      wait_for_successor_state_read(slot);
    }
    HISTOGRAM_OBSERVE(
        prepared_task_consume_latency_microseconds,
        static_cast<int64_t>(consume_timer.elapsed_microseconds()));
    slot.prepared_input = ForwardInput();
    slot.model_binding.reset();
    slot.output.reset();
    slot.task_output_event.reset();
    slot.state = ExecutionSlotState::FREE;
    free_slots_.emplace_back(slot.slot_id);
    output_promise->setValue(std::move(output));

    {
      std::lock_guard<std::mutex> lock(lifecycle_mutex_);
      CHECK_GT(in_flight_tasks_, 0);
      --in_flight_tasks_;
      if (async_mode()) {
        CHECK_GT(pending_result_requests_, 0);
        --pending_result_requests_;
      }
    }
    lifecycle_cv_.notify_all();
  }

  void wait_for_successor_state_read(ExecutionSlot& slot) {
    if (!slot.successor_consumes_state) {
      return;
    }
    std::unique_lock<std::mutex> lock(reuse_mutex_);
    reuse_cv_.wait(lock, [&slot, this]() {
      return slot.successor_read_enqueued || stop_requested_;
    });
    CHECK(slot.successor_read_enqueued)
        << "Prepared successor did not publish its Slot reuse fence";
  }

  void launch_loop() {
    adapter_->initialize_launch_thread();
    while (true) {
      int32_t slot_id = 0;
      {
        std::unique_lock<std::mutex> lock(ready_mutex_);
        ready_cv_.wait(lock, [this]() {
          return !ready_slots_.empty() || stop_requested_;
        });
        if (ready_slots_.empty() && stop_requested_) {
          return;
        }
        slot_id = ready_slots_.front();
        ready_slots_.pop_front();
      }
      ExecutionSlot& slot = slots_[slot_id];
      CHECK(slot.state == ExecutionSlotState::READY);
      slot.state = ExecutionSlotState::RUNNING;
      Timer launch_timer;
      {
#if defined(USE_NPU)
        LLM_MSTX_HOST_RANGE("xllm.PreparedTask.Launch");
#endif
        if (slot.predecessor_slot_id.has_value()) {
          ExecutionSlot& predecessor_slot = slots_[*slot.predecessor_slot_id];
          adapter_->launch_predecessor_continuation(slot, predecessor_slot);
          if (predecessor_slot.slot_id != slot.slot_id) {
            {
              std::lock_guard<std::mutex> lock(reuse_mutex_);
              CHECK(predecessor_slot.successor_consumes_state);
              predecessor_slot.successor_read_enqueued = true;
            }
            reuse_cv_.notify_all();
          }
        }
        adapter_->launch(slot);
      }
      HISTOGRAM_OBSERVE(
          prepared_task_launch_submission_latency_microseconds,
          static_cast<int64_t>(launch_timer.elapsed_microseconds()));
      COUNTER_INC(prepared_task_launch_submissions_total);
      if (async_mode()) {
        std::lock_guard<std::mutex> lock(ready_mutex_);
        if (ready_slots_.empty()) {
          COUNTER_INC(prepared_task_ready_queue_empty_at_launch_end_total);
        }
      }
      slot.state = ExecutionSlotState::COMPLETED;
      {
        std::lock_guard<std::mutex> lock(completed_mutex_);
        CHECK_LT(completed_slots_.size(), slots_.size());
        completed_slots_.emplace_back(slot_id);
      }
      completed_cv_.notify_one();
    }
  }

  std::unique_ptr<PreparedTaskAdapter> adapter_;
  std::vector<ExecutionSlot> slots_;

  mutable std::mutex lifecycle_mutex_;
  std::condition_variable lifecycle_cv_;
  PreparedPipelineLifecycle lifecycle_ = PreparedPipelineLifecycle::CREATED;
  uint64_t in_flight_tasks_ = 0;
  uint64_t pending_result_requests_ = 0;
  std::atomic_bool stop_requested_{false};

  std::mutex state_mutex_;
  std::condition_variable state_cv_;
  std::deque<StateCommand> state_commands_;
  std::deque<int32_t> free_slots_;

  std::mutex ready_mutex_;
  std::condition_variable ready_cv_;
  std::deque<int32_t> ready_slots_;

  std::mutex completed_mutex_;
  std::condition_variable completed_cv_;
  std::deque<int32_t> completed_slots_;

  std::mutex reuse_mutex_;
  std::condition_variable reuse_cv_;

  uint64_t next_task_seq_no_ = 0;
  std::optional<int32_t> last_prepared_slot_id_;
  std::thread state_thread_;
  std::thread launch_thread_;
};

PreparedTaskPipeline::PreparedTaskPipeline(
    std::unique_ptr<PreparedTaskAdapter> adapter,
    int32_t slot_count)
    : impl_(std::make_unique<Impl>(std::move(adapter), slot_count)) {}

PreparedTaskPipeline::~PreparedTaskPipeline() = default;

folly::SemiFuture<std::optional<ForwardOutput>> PreparedTaskPipeline::submit(
    const ForwardInput& input) {
  return impl_->submit(input);
}

folly::SemiFuture<std::optional<ForwardOutput>>
PreparedTaskPipeline::get_last_step_result() {
  return impl_->get_last_step_result();
}

void PreparedTaskPipeline::quiesce() { impl_->quiesce(); }

void PreparedTaskPipeline::resume() { impl_->resume(); }

void PreparedTaskPipeline::shutdown() { impl_->shutdown(); }

PreparedPipelineLifecycle PreparedTaskPipeline::lifecycle() const {
  return impl_->lifecycle();
}

int32_t PreparedTaskPipeline::slot_count() const { return impl_->slot_count(); }

}  // namespace xllm
