/* Copyright 2025-2026 The xLLM Authors.
Copyright 2024 The ScaleLLM Authors. All Rights Reserved.

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

#include "worker.h"

#include <folly/Unit.h>
#include <folly/futures/Future.h>
#include <glog/logging.h>
#include <torch/torch.h>

#include <limits>
#include <memory>
#include <optional>
#include <utility>

#include "common/metrics.h"
#include "core/framework/config/execution_config.h"
#include "core/framework/config/service_config.h"
#include "core/framework/config/speculative_config.h"
#include "framework/kv_cache/kv_cache.h"
#include "framework/model/model_input_params.h"
#include "framework/state_dict/state_dict.h"
#include "platform/platform.h"
#include "runtime/dflash_worker_impl.h"
#include "runtime/dit_worker_impl.h"
#include "runtime/dspark_worker_impl.h"
#include "runtime/eagle3_worker_impl.h"
#include "runtime/embed_vlm_worker_impl.h"
#include "runtime/embed_worker_impl.h"
#include "runtime/llm_worker_impl.h"
#include "runtime/mm_embed_vlm_worker_impl.h"
#include "runtime/mtp_worker_impl.h"
#include "runtime/prepared_task/block_spec_prepared_task_adapter.h"
#include "runtime/prepared_task/llm_prepared_task_adapter.h"
#include "runtime/prepared_task/mtp_prepared_task_adapter.h"
#include "runtime/rec_worker_impl.h"
#include "runtime/suffix_worker_impl.h"
#include "runtime/vlm_worker_impl.h"
#include "util/timer.h"

namespace xllm {
namespace {

// Keep speculative Prepared startup closed until real MTP/Eagle3/DFlash/
// DSpark E2E, deterministic comparison, and Graph/replay validation are
// complete. The delayed construction path below is compiled now so cache
// lifecycle and ownership can be reviewed and tested before this gate opens.
// Offline MTP/Block-Spec single/dual-Slot contracts do not count as
// real-model E2E evidence.
constexpr bool kSpeculativePreparedPipelineE2EValidated = false;
// Keep Prepared Graph startup closed in production. Validation windows may
// temporarily set this true, rebuild, run strict capture/replay E2E, and must
// restore it before the final gated build.
constexpr bool kPreparedGraphE2EValidated = false;

}  // namespace

Worker::Worker(const ParallelArgs& parallel_args,
               const torch::Device& device,
               const runtime::Options& options,
               WorkerType worker_type) {
  if (options.enable_speculative_decode()) {
    const std::string& algorithm = options.speculative_algorithm();
    LOG(INFO) << "Speculative decode is enabled, algorithm: " << algorithm;
    if (algorithm == "Eagle3") {
      impl_ = new Eagle3WorkerImpl(parallel_args, device, options);
    } else if (algorithm == "DFlash") {
      impl_ = new DFlashWorkerImpl(parallel_args, device, options);
    } else if (algorithm == "DSpark") {
      impl_ = new DSparkWorkerImpl(parallel_args, device, options);
    } else if (algorithm == "Suffix") {
      impl_ = new SuffixWorkerImpl(parallel_args, device, options);
    } else if (SpeculativeConfig::is_mtp_algorithm(algorithm)) {
      impl_ = new MTPWorkerImpl(parallel_args, device, options);
    } else {
      LOG(FATAL) << "Unsupported speculative decoding algorithm: " << algorithm;
    }
  } else if (worker_type == WorkerType::LLM) {
    impl_ = new LLMWorkerImpl(parallel_args, device, options);
  } else if (worker_type == WorkerType::VLM) {
    impl_ = new VLMWorkerImpl(parallel_args, device, options);
  } else if (worker_type == WorkerType::ELM) {
    impl_ = new EmbedWorkerImpl(parallel_args, device, options);
  } else if (worker_type == WorkerType::EVLM) {
    impl_ = new EmbedVLMWorkerImpl(parallel_args, device, options);
  } else if (worker_type == WorkerType::REC) {
    impl_ = new RecWorkerImpl(parallel_args, device, options);
  } else if (worker_type == WorkerType::MMEVLM) {
    impl_ = new MMEmbedVLMWorkerImpl(parallel_args, device, options);
  } else if (worker_type == WorkerType::DIT) {
    impl_ = new DiTWorkerImpl(parallel_args, device, options);
  } else {
    LOG(ERROR) << "Unknown worker type, please check logic";
  }

  const ExecutionConfig& execution_config = ExecutionConfig::get_instance();
  if (execution_config.enable_prepared_task_pipeline()) {
    const bool speculative_decode = options.enable_speculative_decode();
    CHECK(Platform::is_npu()) << "PreparedTaskPipeline only supports NPU";
    CHECK(worker_type == WorkerType::LLM)
        << "PreparedTaskPipeline only supports LLM workers";
    CHECK(!speculative_decode || kSpeculativePreparedPipelineE2EValidated)
        << "Speculative PreparedTaskPipeline remains gated until MTP/Eagle3/"
           "DFlash/DSpark E2E and Graph validation complete";
    CHECK_EQ(options.backend(), "llm")
        << "PreparedTaskPipeline requires the native LLM backend";
    CHECK_EQ(parallel_args.cp_size(), 1)
        << "PreparedTaskPipeline currently requires CP=1 until NpuCpPlan "
           "metadata has a fixed Slot-owned representation";
    CHECK(!execution_config.enable_graph() || kPreparedGraphE2EValidated)
        << "Prepared Graph remains gated until replay validation completes";
    CHECK(!options.enable_disagg_pd())
        << "PreparedTaskPipeline does not support disaggregated PD";
    CHECK(!execution_config.enable_shm())
        << "PreparedTaskPipeline does not support shared-memory input";
    CHECK(!ServiceConfig::get_instance().enable_json_object_output())
        << "PreparedTaskPipeline does not support JSON grammar";
    if (speculative_decode) {
      const std::string& algorithm = options.speculative_algorithm();
      CHECK(SpeculativeConfig::requires_prepared_predecessor_rows(algorithm))
          << "Prepared speculative binding only supports MTP, Eagle3, "
             "DFlash, and DSpark";
      if (SpeculativeConfig::is_mtp_algorithm(algorithm) ||
          algorithm == "Eagle3") {
        prepared_adapter_kind_ = PreparedAdapterKind::MTP;
      } else if (algorithm == "DFlash") {
        prepared_adapter_kind_ = PreparedAdapterKind::DFLASH;
      } else {
        CHECK_EQ(algorithm, "DSpark");
        prepared_adapter_kind_ = PreparedAdapterKind::DSPARK;
      }
      CHECK_GT(options.num_speculative_tokens(), 0);
      prepared_num_speculative_tokens_ = options.num_speculative_tokens();
      prepared_pipeline_activator_.configure(
          PreparedPipelineActivationStage::AFTER_CACHE_ALLOCATION);
    } else {
      prepared_adapter_kind_ = PreparedAdapterKind::LLM;
      prepared_pipeline_activator_.configure(
          PreparedPipelineActivationStage::IMMEDIATE);
    }
    constexpr uint64_t kBytesPerMib = 1024 * 1024;
    CHECK_GT(execution_config.prepared_task_input_buffer_size(), 0);
    CHECK_LE(execution_config.prepared_task_input_buffer_size(),
             std::numeric_limits<uint64_t>::max() / kBytesPerMib);
    prepared_input_arena_capacity_bytes_ =
        execution_config.prepared_task_input_buffer_size() * kBytesPerMib;
    prepared_slot_count_ = options.enable_schedule_overlap() ? 2 : 1;
    if (prepared_adapter_kind_ == PreparedAdapterKind::LLM) {
      prepared_pipeline_activator_.initialize_at_construction(
          [this]() { initialize_prepared_pipeline(); });
    } else {
      LOG(INFO) << "PreparedTaskPipeline speculative adapter will be "
                   "constructed after KV/embedding cache allocation";
    }
  }
}

Worker::~Worker() {
  prepared_pipeline_.reset();
  delete impl_;
}

bool Worker::init_model(const std::string& model_weights_path,
                        int32_t random_seed,
                        MasterStatus master_status) {
  return impl_->init_model(model_weights_path, random_seed, master_status);
}

bool Worker::allocate_kv_cache(const KVCacheShape& kv_cache_shape) {
  const bool success = impl_->allocate_kv_cache(kv_cache_shape);
  return finish_kv_cache_allocation(success);
}

bool Worker::set_speculative_validate_time_predictor(
    const SpeculativeProfileRegistry::ValidateTimePredictor& predictor) {
  SpeculativeProfileRegistry::get_instance().set_validate_time_predictor(
      predictor);
  return true;
}

void Worker::get_cache_info(uint64_t& cluster_id,
                            std::string& addr,
                            uint16_t& port) {
  impl_->get_cache_info(cluster_id, addr, port);
}

bool Worker::link_cluster(const std::vector<uint64_t>& cluster_ids,
                          const std::vector<std::string>& addrs,
                          const std::vector<uint16_t>& ports) {
  return impl_->link_cluster(cluster_ids, addrs, ports);
}

bool Worker::unlink_cluster(const std::vector<uint64_t>& cluster_ids,
                            const std::vector<std::string>& addrs,
                            const std::vector<uint16_t>& ports) {
  return impl_->unlink_cluster(cluster_ids, addrs, ports);
}

bool Worker::link_p2p(const std::string& remote_addr) {
  return impl_->link_p2p(remote_addr);
}

bool Worker::unlink_p2p(const std::string& remote_addr) {
  return impl_->unlink_p2p(remote_addr);
}

std::tuple<int64_t, int64_t> Worker::estimate_kv_cache_capacity() {
  return impl_->estimate_kv_cache_capacity();
}

ForwardInput Worker::prepare_inputs(Batch& batch) {
  return impl_->prepare_inputs(batch);
}

std::optional<ForwardOutput> Worker::step(const ForwardInput& inputs) {
  if (prepared_adapter_kind_ != PreparedAdapterKind::NONE) {
    CHECK(prepared_pipeline_ != nullptr)
        << "PreparedTaskPipeline is waiting for successful cache allocation";
    auto future = prepared_pipeline_->submit(inputs);
    return std::move(future).get();
  }
  return impl_->step(inputs);
}

const bool Worker::is_driver() { return impl_->is_driver(); }

folly::SemiFuture<std::tuple<int64_t, int64_t>>
Worker::estimate_kv_cache_capacity_async() {
  return impl_->estimate_kv_cache_capacity_async();
}

folly::SemiFuture<std::optional<ForwardOutput>> Worker::step_async(
    const ForwardInput& inputs) {
  if (prepared_adapter_kind_ != PreparedAdapterKind::NONE) {
    CHECK(prepared_pipeline_ != nullptr)
        << "PreparedTaskPipeline is waiting for successful cache allocation";
    return prepared_pipeline_->submit(inputs);
  }
  return impl_->step_async(inputs);
}

folly::SemiFuture<folly::Unit> Worker::process_group_test_async() {
  return impl_->process_group_test_async();
}

// initialize model, cache manager. async call
folly::SemiFuture<bool> Worker::init_model_async(
    const std::string& model_weights_path,
    int32_t random_seed,
    MasterStatus master_status) {
  return impl_->init_model_async(
      model_weights_path, random_seed, master_status);
}

folly::SemiFuture<bool> Worker::allocate_kv_cache_async(
    const KVCacheShape& kv_cache_shape) {
  return std::move(impl_->allocate_kv_cache_async(kv_cache_shape))
      .deferValue(
          [this](bool success) { return finish_kv_cache_allocation(success); });
}

folly::SemiFuture<bool> Worker::allocate_kv_cache_with_transfer_async(
    const KVCacheShape& kv_cache_shape) {
  return std::move(impl_->allocate_kv_cache_with_transfer_async(kv_cache_shape))
      .deferValue(
          [this](bool success) { return finish_kv_cache_allocation(success); });
}

folly::SemiFuture<bool> Worker::pull_kv_blocks_async(
    const uint64_t src_cluster_id,
    const std::string& src_addr,
    const std::vector<KVTransferMapping>& mappings) {
  return impl_->pull_kv_blocks_async(src_cluster_id, src_addr, mappings);
}

uint32_t Worker::transfer_kv_blocks(
    const uint64_t batch_id,
    const std::vector<BlockTransferInfo>& block_transfer_info) {
  return impl_->transfer_kv_blocks(batch_id, std::move(block_transfer_info));
}

uint32_t Worker::transfer_kv_blocks(
    const uint64_t batch_id,
    Slice<BlockTransferInfo>& block_transfer_info) {
  return impl_->transfer_kv_blocks(batch_id, block_transfer_info);
}

std::vector<uint8_t> Worker::prefetch_kv_blocks(
    Slice<BlockTransferInfo>& block_transfer_info) {
  return impl_->prefetch_kv_blocks(block_transfer_info);
}

const torch::Device& Worker::device() const { return impl_->device(); }

folly::SemiFuture<std::optional<ForwardOutput>>
Worker::get_last_step_result_async() {
  if (prepared_adapter_kind_ != PreparedAdapterKind::NONE) {
    CHECK(prepared_pipeline_ != nullptr)
        << "PreparedTaskPipeline is waiting for successful cache allocation";
    return prepared_pipeline_->get_last_step_result();
  }
  folly::Promise<std::optional<ForwardOutput>> promise;
  auto future = promise.getSemiFuture();
  threadpool_.schedule([this, promise = std::move(promise)]() mutable {
    promise.setValue(impl_->get_last_step_result());
  });
  return future;
}

int64_t Worker::get_active_activation_memory() {
  return impl_->get_active_activation_memory();
}

folly::SemiFuture<int64_t> Worker::get_active_activation_memory_async() {
  folly::Promise<int64_t> promise;
  auto future = promise.getSemiFuture();
  threadpool_.schedule([this, promise = std::move(promise)]() mutable {
    promise.setValue(impl_->get_active_activation_memory());
  });
  return future;
}

bool Worker::sleep(MasterStatus master_status) {
  if (prepared_pipeline_ != nullptr) {
    prepared_pipeline_->quiesce();
  }
  return impl_->sleep(master_status);
}

bool Worker::wakeup(const WakeupOptions& options) {
  const bool status = impl_->wakeup(options);
  if (status && prepared_pipeline_ != nullptr &&
      prepared_pipeline_->lifecycle() == PreparedPipelineLifecycle::QUIESCENT) {
    prepared_pipeline_->resume();
  }
  return status;
}

bool Worker::update_weights(const std::string& weights_path) {
  if (prepared_adapter_kind_ != PreparedAdapterKind::NONE) {
    LOG(ERROR) << "update_weights is not supported while "
                  "PreparedTaskPipeline is enabled";
    return false;
  }
  return impl_->update_weights(weights_path);
}

folly::SemiFuture<bool> Worker::wakeup_async(const WakeupOptions& options) {
  folly::Promise<bool> promise;
  auto future = promise.getSemiFuture();
  threadpool_.schedule([this, options, promise = std::move(promise)]() mutable {
    promise.setValue(this->wakeup(options));
  });
  return future;
}

bool Worker::start_profile() { return impl_->start_profile(); }

bool Worker::stop_profile() { return impl_->stop_profile(); }

folly::SemiFuture<bool> Worker::start_profile_async() {
  folly::Promise<bool> promise;
  auto future = promise.getSemiFuture();
  threadpool_.schedule([this, promise = std::move(promise)]() mutable {
    promise.setValue(this->start_profile());
  });
  return future;
}

folly::SemiFuture<bool> Worker::stop_profile_async() {
  folly::Promise<bool> promise;
  auto future = promise.getSemiFuture();
  threadpool_.schedule([this, promise = std::move(promise)]() mutable {
    promise.setValue(this->stop_profile());
  });
  return future;
}

void Worker::initialize_prepared_pipeline() {
  CHECK(prepared_adapter_kind_ != PreparedAdapterKind::NONE);
  CHECK(prepared_pipeline_ == nullptr)
      << "PreparedTaskPipeline may only be initialized once";
  CHECK_GT(prepared_input_arena_capacity_bytes_, 0);
  CHECK(prepared_slot_count_ == 1 || prepared_slot_count_ == 2);

  std::unique_ptr<PreparedTaskAdapter> adapter;
  const char* adapter_name = nullptr;
  switch (prepared_adapter_kind_) {
    case PreparedAdapterKind::LLM: {
      LLMWorkerImpl* llm_worker = dynamic_cast<LLMWorkerImpl*>(impl_);
      CHECK(llm_worker != nullptr);
      adapter = std::make_unique<LlmPreparedTaskAdapter>(
          llm_worker,
          prepared_input_arena_capacity_bytes_,
          prepared_slot_count_);
      adapter_name = "LLM";
      break;
    }
    case PreparedAdapterKind::MTP: {
      MTPWorkerImpl* mtp_worker = dynamic_cast<MTPWorkerImpl*>(impl_);
      CHECK(mtp_worker != nullptr);
      std::unique_ptr<MtpPreparedTaskBackend> backend =
          mtp_worker->create_prepared_task_backend(
              prepared_input_arena_capacity_bytes_, prepared_slot_count_);
      adapter = std::make_unique<MtpPreparedTaskAdapter>(
          std::move(backend),
          prepared_num_speculative_tokens_,
          prepared_slot_count_);
      adapter_name = "MTP";
      break;
    }
    case PreparedAdapterKind::DFLASH:
    case PreparedAdapterKind::DSPARK: {
      DFlashWorkerImpl* block_spec_worker =
          dynamic_cast<DFlashWorkerImpl*>(impl_);
      CHECK(block_spec_worker != nullptr);
      std::unique_ptr<BlockSpecPreparedTaskBackend> backend =
          block_spec_worker->create_prepared_task_backend(
              prepared_input_arena_capacity_bytes_, prepared_slot_count_);
      const bool is_dspark =
          prepared_adapter_kind_ == PreparedAdapterKind::DSPARK;
      adapter = std::make_unique<BlockSpecPreparedTaskAdapter>(
          std::move(backend),
          is_dspark ? BlockSpecAlgorithm::DSPARK : BlockSpecAlgorithm::DFLASH,
          prepared_num_speculative_tokens_,
          prepared_slot_count_);
      adapter_name = is_dspark ? "DSpark" : "DFlash";
      break;
    }
    case PreparedAdapterKind::NONE:
      LOG(FATAL) << "Cannot initialize a disabled PreparedTaskPipeline";
  }

  CHECK(adapter != nullptr);
  prepared_pipeline_ = std::make_unique<PreparedTaskPipeline>(
      std::move(adapter), prepared_slot_count_);
  LOG(INFO) << "PreparedTaskPipeline enabled: adapter=" << adapter_name
            << ", mode=eager, slot_count=" << prepared_slot_count_
            << ", input_arena_bytes=" << prepared_input_arena_capacity_bytes_;
}

bool Worker::finish_kv_cache_allocation(bool success) {
  if (!success || prepared_adapter_kind_ == PreparedAdapterKind::NONE ||
      prepared_adapter_kind_ == PreparedAdapterKind::LLM) {
    return success;
  }
  prepared_pipeline_activator_.finish_cache_allocation(
      success, [this]() { initialize_prepared_pipeline(); });
  return true;
}
}  // namespace xllm
