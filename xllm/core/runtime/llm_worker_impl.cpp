/* Copyright 2025-2026 The xLLM Authors.

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

#include "llm_worker_impl.h"

#include <c10/core/DeviceGuard.h>
#include <folly/Unit.h>
#include <folly/futures/Future.h>
#include <glog/logging.h>
#include <torch/torch.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <string_view>
#include <utility>

#include "common/device_monitor.h"
#include "common/metrics.h"
#include "common/types.h"
#include "core/framework/config/beam_search_config.h"
#include "core/framework/config/eplb_config.h"
#include "core/framework/config/kv_cache_config.h"
#include "core/framework/config/load_config.h"
#include "core/framework/config/model_config.h"
#include "framework/kv_cache/kv_cache.h"
#include "framework/kv_cache/linear_state_restore.h"
#include "framework/kv_cache_transfer/kv_transfer_completion.h"
#include "framework/model/model_args.h"
#include "framework/model/model_input_params.h"
#include "framework/state_dict/state_dict.h"
#if defined(USE_NPU)
#include "kernels/npu/xllm_ops/xllm_ops_api.h"
#endif
#if defined(USE_CUDA) || defined(USE_ILU) || defined(USE_MUSA)
#include "layers/cuda/flashinfer_workspace.h"
#endif
#include "models/model_registry.h"
#include "util/env_var.h"
#include "util/threadpool.h"
#include "util/timer.h"
#include "util/verbose_trace_logger.h"

namespace xllm {

namespace {

void wait_input_ready_events(const ForwardInput& input, const Stream& stream) {
  CHECK(stream.wait_event(input.metadata_ready_event))
      << "failed to wait ForwardInput metadata ready event";
}

StreamEventPtr record_current_stream_event(const Device& device) {
  std::unique_ptr<Stream> stream = device.current_stream();
  return stream->record_event_or_sync();
}

void trace_llm_decode_step_state(const ForwardInput& input,
                                 const torch::Tensor& next_tokens) {
  if (!VerboseTraceLogger::get_instance().enabled() ||
      input.input_params.meta.is_graph_warmup ||
      !input.input_params.meta.batch_forward_type.is_decode() ||
      !input.sampling_params.all_greedy_sample || !next_tokens.defined()) {
    return;
  }

  const std::vector<int32_t>& embedding_ids =
      input.input_params.embedding.embedding_ids;
  const std::vector<std::string>& request_ids =
      input.input_params.embedding.request_ids;
  const int64_t batch_size = static_cast<int64_t>(embedding_ids.size());
  CHECK_GT(batch_size, 0);
  CHECK(request_ids.empty() ||
        request_ids.size() == static_cast<size_t>(batch_size));
  CHECK_EQ(next_tokens.numel(), batch_size)
      << "greedy LLM decode must produce one token per sequence";
  CHECK(input.positions.defined());
  CHECK_EQ(input.positions.numel(), batch_size);
  const torch::Tensor& kv_seq_lens =
      input.input_params.attention.device.kv_seq_lens;
  CHECK(kv_seq_lens.defined());
  CHECK_EQ(kv_seq_lens.numel(), batch_size);

  const torch::Tensor next_tokens_cpu =
      next_tokens.to(torch::kCPU, torch::kInt64).contiguous().view({-1});
  const torch::Tensor positions_cpu =
      input.positions.to(torch::kCPU, torch::kInt64).contiguous().view({-1});
  const torch::Tensor kv_seq_lens_cpu =
      kv_seq_lens.to(torch::kCPU, torch::kInt64).contiguous().view({-1});
  const int64_t* token_data = next_tokens_cpu.const_data_ptr<int64_t>();
  const int64_t* position_data = positions_cpu.const_data_ptr<int64_t>();
  const int64_t* kv_seq_len_data = kv_seq_lens_cpu.const_data_ptr<int64_t>();
  for (int64_t row = 0; row < batch_size; ++row) {
    const std::string_view request_id =
        request_ids.empty()
            ? std::string_view()
            : std::string_view(request_ids[static_cast<size_t>(row)]);
    XLLM_VERBOSE_TRACE() << "event=llm_step_state request_id="
                         << (request_id.empty() ? std::string_view("-")
                                                : request_id)
                         << " embedding_id="
                         << embedding_ids[static_cast<size_t>(row)]
                         << " base_position=" << position_data[row]
                         << " base_kv_seq_len=" << kv_seq_len_data[row]
                         << " token=" << token_data[row];
  }
}

void check_fixed_prepared_output_tensor(const torch::Tensor& actual,
                                        const torch::Tensor& fixed_storage,
                                        const char* tensor_name) {
  CHECK(actual.defined()) << "Prepared model " << tensor_name
                          << " must be defined";
  CHECK(actual.sizes() == fixed_storage.sizes())
      << "Prepared model " << tensor_name << " shape does not match the "
      << "fixed output workspace";
  CHECK_EQ(actual.device(), fixed_storage.device())
      << "Prepared model " << tensor_name << " changed Device";
  CHECK_EQ(actual.scalar_type(), fixed_storage.scalar_type())
      << "Prepared model " << tensor_name << " changed dtype";
  CHECK(fixed_storage.is_contiguous())
      << "Prepared model fixed " << tensor_name << " must be contiguous";
  CHECK(actual.is_contiguous())
      << "Prepared model " << tensor_name << " must be contiguous";
  CHECK_EQ(actual.data_ptr(), fixed_storage.data_ptr())
      << "Prepared model " << tensor_name
      << " replaced the fixed output storage";
}

}  // namespace

void check_prepared_model_output_binding(
    const SampleOutput& sample_output,
    const PreparedModelOutputWorkspace& output_workspace) {
  if (output_workspace.next_tokens.defined()) {
    check_fixed_prepared_output_tensor(sample_output.next_tokens,
                                       output_workspace.next_tokens,
                                       "token output");
  }
  if (output_workspace.selected_embeddings.defined()) {
    const torch::Tensor& actual_embeddings =
        sample_output.selected_embeddings.defined()
            ? sample_output.selected_embeddings
            : sample_output.embeddings;
    check_fixed_prepared_output_tensor(actual_embeddings,
                                       output_workspace.selected_embeddings,
                                       "selected embedding output");
  }
}

torch::Tensor gather_prepared_selected_embeddings(
    const torch::Tensor& embeddings,
    const torch::Tensor& selected_token_idxes,
    const torch::Tensor& destination) {
  CHECK(embeddings.defined());
  CHECK(selected_token_idxes.defined());
  CHECK(destination.defined());
  CHECK_EQ(embeddings.dim(), 2);
  CHECK_EQ(selected_token_idxes.dim(), 1);
  CHECK_EQ(destination.dim(), 2);
  CHECK_EQ(destination.size(0), selected_token_idxes.numel());
  CHECK_EQ(destination.size(1), embeddings.size(1));
  CHECK(destination.device() == embeddings.device());
  CHECK(selected_token_idxes.device() == embeddings.device());
  CHECK_EQ(destination.scalar_type(), embeddings.scalar_type())
      << "Prepared selected embedding destination must match source dtype";
  CHECK(destination.is_contiguous())
      << "Prepared selected embedding destination must be contiguous";
  CHECK(selected_token_idxes.scalar_type() == torch::kInt ||
        selected_token_idxes.scalar_type() == torch::kLong)
      << "Prepared selected embedding indices must be int32 or int64";
  const void* destination_address = destination.data_ptr();
  torch::Tensor selected_embeddings = destination;
  torch::index_select_out(selected_embeddings,
                          embeddings,
                          /*dim=*/0,
                          selected_token_idxes);
  CHECK_EQ(selected_embeddings.data_ptr(), destination_address)
      << "Prepared selected embedding gather replaced fixed output storage";
  return selected_embeddings;
}

LLMWorkerImpl::LLMWorkerImpl(const ParallelArgs& parallel_args,
                             const torch::Device& device,
                             const runtime::Options& options)
    : WorkerImpl(parallel_args, device, options) {
  device_.set_device();
#if defined(USE_CUDA) || defined(USE_MUSA)
  const auto& model_config = ModelConfig::get_instance();
  if (!ModelConfig::is_python_model_impl(model_config.model_impl())) {
    threadpool_.schedule([this]() mutable {
      // initialize flashinfer workspace
      ::xllm::layer::flashinfer::FlashinferWorkspace::get_instance().initialize(
          device_);
    });
  }
#endif
}

bool LLMWorkerImpl::init_model(ModelContext& context) {
  CHECK(model_ == nullptr) << "Model is already initialized.";
  const auto& model_config = ModelConfig::get_instance();
#if defined(USE_MUSA)
  static const bool use_pool_compute_stream = util::get_bool_env(
      "XLLM_MUSA_POOL_COMPUTE_STREAM", /*default_value=*/true);
  const bool is_qwen3 = context.get_model_args().model_type() == "qwen3";
  if (use_pool_compute_stream && !is_qwen3) {
    compute_stream_ = device_.get_stream_from_pool();
  } else if (use_pool_compute_stream) {
    LOG(WARNING) << "MUSA pool compute streams are not validated for Qwen3 "
                    "attention; using the default compute stream.";
  }

  const auto& beam_search_config = BeamSearchConfig::get_instance();
  CHECK(!has_linear_attention_layers(context.get_model_args()) ||
        (!beam_search_config.enable_beam_search_kernel() &&
         beam_search_config.beam_width() <= 1))
      << "MUSA beam search is not supported for models with linear-attention "
         "layers.";
#endif

#if defined(USE_CUDA) || defined(USE_MUSA)
  // Ensure FlashinferWorkspace is initialized on the calling thread before
  // constructing model layers. When called synchronously from
  // SpeculativeWorkerImpl (e.g. MTP target/draft setup), init_model runs on
  // the MTP worker's thread (T_MTP) rather than on the LLMWorkerImpl's own
  // threadpool thread (T_worker) where the scheduled initialize() runs.
  // FlashinferWorkspace is thread_local, so T_MTP's instance must be
  // explicitly initialized here; otherwise FlashInferAttentionImpl captures
  // an undefined int_workspace_buffer_ and crashes at prefill time.
  //
  // Skip when model_impl=python: Python executor uses flashinfer's Python API
  // directly; initializing the C++ workspace would conflict with Python-side
  // TVM-FFI type registration.
  if (!ModelConfig::is_python_model_impl(model_config.model_impl())) {
    auto& ws = ::xllm::layer::flashinfer::FlashinferWorkspace::get_instance();
    if (!ws.get_int_workspace_buffer().defined()) {
      ws.initialize(device_);
    }
  }
#endif

  // Try to create a causal LM model
  context.set_model_impl(model_config.model_impl());
  model_ = create_llm_model(context);

  // Dont find model in causal models
  CHECK(model_ != nullptr) << "Failed to create model.";
  model_executor_ = std::make_unique<Executor>(
      model_.get(), context.get_model_args(), device_, options_);

  if (::xllm::EPLBConfig::get_instance().enable_eplb()) {
    eplb_executor_ = std::make_unique<EplbExecutor>(*model_, device_);
  }

  if (::xllm::BeamSearchConfig::get_instance().enable_beam_search_kernel()) {
    beam_searcher_ = std::make_unique<BeamSearcher>();
  }
  return true;
}

#if defined(USE_NPU)
bool LLMWorkerImpl::prepare_static_mtp_graph_tasks(
    const SpecVerifyGraphTaskSignal& signal,
    const Stream& signal_stream) {
  if (model_executor_ == nullptr) {
    return false;
  }
  return model_executor_->prepare_static_mtp_graph_tasks(signal, signal_stream);
}
#endif

std::optional<ForwardOutput> LLMWorkerImpl::step_no_sync(
    const ForwardInput& input) {
  ForwardInput input_on_device;
  prepare_work_before_execute(input, input_on_device);
  std::unique_ptr<Stream> current_stream = device_.current_stream();
  return execute_no_sync_on_stream(input_on_device, *current_stream);
}

std::optional<ForwardOutput> LLMWorkerImpl::execute_no_sync_on_stream(
    const ForwardInput& input,
    Stream& compute_stream,
    bool record_ready_event) {
  return execute_on_stream(
      input, compute_stream, record_ready_event, /*prepared_binding=*/nullptr);
}

std::optional<ForwardOutput> LLMWorkerImpl::execute_on_stream(
    const ForwardInput& input,
    Stream& compute_stream,
    bool record_ready_event,
    const PreparedSlotBinding* prepared_binding,
    const PreparedModelOutputWorkspace* output_workspace,
    bool retain_input_for_async_output) {
  const ForwardSyncPolicy sync_policy = ForwardSyncPolicy::NO_SYNC;
  c10::StreamGuard stream_guard = compute_stream.set_stream_guard();
  if (::xllm::LoadConfig::get_instance().enable_manual_loader()) {
#if defined(USE_NPU)
    if (!enable_schedule_overlap() && options_.backend() == "llm") {
      aclrtStream current_acl_stream =
          c10_npu::getCurrentNPUStream(device_.index()).stream();
      atb::Context* atb_context =
          const_cast<atb::Context*>(context_.get_atb_context());
      atb_context->SetExecuteStream(current_acl_stream);
      wait_input_ready_events(input, compute_stream);
      return step_internal(input,
                           sync_policy,
                           record_ready_event,
                           prepared_binding,
                           output_workspace,
                           retain_input_for_async_output);
    } else {
      SET_ATB_EXECUTE_STREAM((&compute_stream), device_, context_);
      wait_input_ready_events(input, compute_stream);
      return step_internal(input,
                           sync_policy,
                           record_ready_event,
                           prepared_binding,
                           output_workspace,
                           retain_input_for_async_output);
    }
#else
    wait_input_ready_events(input, compute_stream);
    return step_internal(input,
                         sync_policy,
                         record_ready_event,
                         prepared_binding,
                         output_workspace,
                         retain_input_for_async_output);
#endif
  }
  wait_input_ready_events(input, compute_stream);
  return step_internal(input,
                       sync_policy,
                       record_ready_event,
                       prepared_binding,
                       output_workspace,
                       retain_input_for_async_output);
}

std::optional<ForwardOutput> LLMWorkerImpl::execute_prepared_task(
    const ForwardInput& input,
    const std::optional<PreparedSlotBinding>& binding) {
  CHECK(compute_stream_ != nullptr);
  if (enable_schedule_overlap() &&
      has_linear_attention_layers(context_.get_model_args())) {
    c10::StreamGuard restore_guard = compute_stream_->set_stream_guard();
    ModelInputParams& mutable_params =
        const_cast<ModelInputParams&>(input.input_params);
    restore_linear_state_slots(kv_caches_,
                               mutable_params.linear_state_cache_ops,
                               mutable_params.linear_state_validity_mask);
  }
  return execute_prepared_on_stream(input, *compute_stream_, binding);
}

std::optional<ForwardOutput> LLMWorkerImpl::execute_prepared_on_stream(
    const ForwardInput& input,
    Stream& compute_stream,
    const std::optional<PreparedSlotBinding>& binding,
    const PreparedModelOutputWorkspace* output_workspace) {
  if (!binding.has_value()) {
    COUNTER_INC(prepared_task_execution_total_eager);
  }
  const PreparedSlotBinding* binding_ptr =
      binding.has_value() ? &binding.value() : nullptr;
  return execute_on_stream(input,
                           compute_stream,
                           /*record_ready_event=*/false,
                           binding_ptr,
                           output_workspace,
                           /*retain_input_for_async_output=*/false);
}

PreparedSlotBinding LLMWorkerImpl::bind_prepared_task(
    int32_t slot_id,
    const ForwardInput& input) {
  CHECK(prepared_graph_enabled());
  CHECK(model_executor_ != nullptr);
  return model_executor_->bind_prepared(slot_id, input, kv_caches_);
}

void LLMWorkerImpl::prepare_prepared_graph_input(int32_t slot_id,
                                                 ForwardInput& input) {
  CHECK(prepared_graph_enabled());
  CHECK(model_executor_ != nullptr);
  model_executor_->prepare_prepared_graph_input(slot_id, input, kv_caches_);
}

bool LLMWorkerImpl::prepared_graph_enabled() const {
  return ::xllm::ExecutionConfig::get_instance().enable_graph();
}

void LLMWorkerImpl::patch_prepared_task_input_for_schedule_overlap(
    ForwardInput& input) {
  CHECK(enable_schedule_overlap());
  CHECK(input.json_object_states.empty() &&
        input.json_object_state_snapshots.empty())
      << "JSON grammar is not supported by PreparedTaskPipeline phase 2";
  if (input.token_ids.numel() == 0 ||
      !input.input_params.meta.batch_forward_type.has_decode() ||
      !can_use_last_step_output_for_schedule_overlap(input)) {
    return;
  }
#if defined(USE_NPU)
  CHECK(compute_stream_ != nullptr);
  c10::StreamGuard stream_guard = compute_stream_->set_stream_guard();
  if (last_step_output_.ready_event != nullptr) {
    CHECK(compute_stream_->wait_event(last_step_output_.ready_event))
        << "Failed to wait for the preceding Prepared task output";
  }
  xllm::kernel::npu::replace_token(input.token_ids,
                                   last_step_output_.sample_output.next_tokens,
                                   /*synchronize_stream=*/false);
#else
  LOG(FATAL) << "Prepared schedule overlap only supports NPU";
#endif
}

void LLMWorkerImpl::publish_prepared_task_output(
    const ForwardInput& input,
    const std::optional<ForwardOutput>& output) {
  CHECK(enable_schedule_overlap());
  if (output.has_value()) {
    update_last_step_output(output,
                            input.input_params.embedding.request_ids,
                            input.sample_sequence_ids);
    return;
  }
  last_step_output_valid_ = false;
  last_step_output_ = ForwardOutput();
  last_step_request_ids_.clear();
  last_step_sample_sequence_ids_.clear();
}

std::optional<ForwardOutput> LLMWorkerImpl::step(const ForwardInput& input) {
#if defined(USE_NPU)
  if (::xllm::LoadConfig::get_instance().enable_manual_loader()) {
    if (!enable_schedule_overlap() && options_.backend() == "llm") {
      aclrtStream current_stream =
          c10_npu::getCurrentNPUStream(device_.index()).stream();
      atb::Context* atb_context =
          const_cast<atb::Context*>(context_.get_atb_context());
      atb_context->SetExecuteStream(current_stream);
    } else {
      SET_ATB_EXECUTE_STREAM(compute_stream_, device_, context_);
      wait_input_ready_events(input, *compute_stream_);
      return step_internal(input, ForwardSyncPolicy::LEGACY);
    }
  }
#endif

  std::unique_ptr<Stream> stream = device_.current_stream();
  wait_input_ready_events(input, *stream);
  return step_internal(input, ForwardSyncPolicy::LEGACY);
}

folly::SemiFuture<std::optional<ForwardOutput>>
LLMWorkerImpl::step_async_no_sync(const ForwardInput& input) {
  CHECK(!enable_schedule_overlap())
      << "step_async_no_sync is only supported for non-overlap workers";
  ForwardInput input_on_device;

  prepare_work_before_execute(input, input_on_device);

  folly::Promise<std::optional<ForwardOutput>> promise;
  auto future = promise.getSemiFuture();
  threadpool_.schedule([this,
                        input = std::move(input_on_device),
                        promise = std::move(promise)]() mutable {
    // hierarchy temporarily disabled during the block-manager refactor
    // if (hierarchy_kv_cache_transfer_ != nullptr) {
    //   hierarchy_kv_cache_transfer_->set_layer_synchronizer(input.input_params);
    // }

    const auto output = this->step_no_sync(input);
    promise.setValue(output);
  });
  return future;
}

std::optional<ForwardOutput> LLMWorkerImpl::step_for_schedule_overlap(
    const ForwardInput& input) {
  // Restore live recurrent-state slots from saved checkpoints here (worker
  // thread, on compute_stream_) instead of in prepare_work_before_execute on
  // prepare_stream_. The single-threaded worker pool guarantees the previous
  // chunk's forward kernels are already enqueued on compute_stream_ before
  // this task runs, so the restore copy is automatically stream-ordered
  // after those writes without needing a cross-stream barrier.
  if (has_linear_attention_layers(context_.get_model_args())) {
    c10::StreamGuard restore_guard = compute_stream_->set_stream_guard();
    ModelInputParams& mutable_params =
        const_cast<ModelInputParams&>(input.input_params);
    restore_linear_state_slots(kv_caches_,
                               mutable_params.linear_state_cache_ops,
                               mutable_params.linear_state_validity_mask);
  }
  return execute_no_sync_on_stream(input, *compute_stream_);
}

ForwardInput
LLMWorkerImpl::update_input_by_last_step_output_for_schedule_overlap(
    ForwardInput& input) {
  c10::StreamGuard stream_guard = compute_stream_->set_stream_guard();
  CHECK(compute_stream_->wait_event(last_step_output_.ready_event))
      << "failed to wait last step output ready event";
  return WorkerImpl::update_input_by_last_step_output_for_schedule_overlap(
      input);
}

std::optional<ForwardOutput> LLMWorkerImpl::step_internal(
    const ForwardInput& input,
    ForwardSyncPolicy sync_policy,
    bool record_ready_event,
    const PreparedSlotBinding* prepared_binding,
    const PreparedModelOutputWorkspace* output_workspace,
    bool retain_input_for_async_output) {
  MULTI_MODEL_STEP_LOCK(::xllm::KVCacheConfig::get_instance().enable_xtensor());

  Timer timer;
  auto& sampling_params = input.sampling_params;

  KVTransferCompletion kv_transfers;

  if (options_.kv_cache_transfer_mode() == "PUSH" &&
      !input.transfer_kv_infos.empty()) {
#if defined(USE_NPU)
    std::shared_ptr<NPULayerSynchronizerImpl> layer_synchronizer =
        std::make_shared<NPULayerSynchronizerImpl>(
            context_.get_model_args().n_layers());
#elif defined(USE_MLU)
    std::shared_ptr<MLULayerSynchronizerImpl> layer_synchronizer =
        std::make_shared<MLULayerSynchronizerImpl>(
            context_.get_model_args().n_layers());
#elif defined(USE_DCU)
    std::shared_ptr<DCULayerSynchronizerImpl> layer_synchronizer =
        std::make_shared<DCULayerSynchronizerImpl>(
            context_.get_model_args().n_layers());
#endif
#if defined(USE_NPU) || defined(USE_MLU) || defined(USE_DCU)
    const_cast<ModelInputParams*>(&(input.input_params))
        ->parallel.layer_synchronizer = layer_synchronizer;

    kv_transfers.add(
        kv_cache_transfer_->push_kv_blocks_async(input.transfer_kv_infos,
                                                 context_.get_parallel_args(),
                                                 layer_synchronizer,
                                                 is_spec_draft_));
#endif
  }
  auto wait_kv_push = [&kv_transfers]() {
    CHECK(kv_transfers.wait()) << "KV cache push failed";
  };
  if (::xllm::EPLBConfig::get_instance().enable_eplb()) {
    eplb_executor_->start_eplb_step(input.input_params.expert.eplb_info);
  }

  // call model executor forward to get hidden states
  ModelOutput model_output;
  if (prepared_binding != nullptr) {
    model_output =
        model_executor_->forward_prepared(*prepared_binding, input, kv_caches_);
  } else {
    model_output = model_executor_->forward(
        input.token_ids, input.positions, kv_caches_, input.input_params);
  }
  if (::xllm::EPLBConfig::get_instance().enable_eplb()) {
    eplb_executor_->finish_eplb_step();
  }
  if (!model_output.hidden_states.defined()) {
    wait_kv_push();
    return std::nullopt;
  }

  torch::Tensor logits;
  torch::Tensor selected_hidden;
  if (sampling_params.selected_token_idxes.defined()) {
    torch::Tensor selected_token_idxes = sampling_params.selected_token_idxes;
    if (model_output.hidden_states.defined() &&
        selected_token_idxes.device() != model_output.hidden_states.device()) {
      selected_token_idxes = selected_token_idxes
                                 .to(model_output.hidden_states.device(),
                                     /*non_blocking=*/false)
                                 .contiguous();
    }
    if (input.return_selected_hidden) {
      // Emit both selected hidden and logits from a single lm_head pass so the
      // ConfidenceHead can consume hidden without a second projection.
      logits = model_->logits(
          model_output.hidden_states, selected_token_idxes, selected_hidden);
      if (!selected_hidden.defined() && model_output.hidden_states.defined()) {
        // ATB lm_head backend does not expose a second output tensor
        // (LmHeadParam::outputHidden is false), so we surface the hidden here.
        // selected_hidden must align row-for-row with `logits`, which is
        // produced in selected_token_idxes order. index_select reproduces that
        // order for any selection. We deliberately do NOT alias the full
        // hidden_states on a numel match: equal row count does not imply the
        // idxes are the identity permutation, and a full-but-reordered
        // selection would silently misalign hidden against logits. The gather
        // is one [num_selected, hidden] copy per decode step (not per layer),
        // negligible next to the forward.
        selected_hidden = model_output.hidden_states.index_select(
            /*dim=*/0, selected_token_idxes.to(torch::kLong));
      }
    } else {
      logits = model_->logits(model_output.hidden_states, selected_token_idxes);
    }
  }

  ForwardOutput output;
  output.mtp_topk_state = std::move(model_output.mtp_topk_state);
  if (::xllm::EPLBConfig::get_instance().enable_eplb()) {
    output.expert_load_data = expert_load_data_;
    output.prepared_token = eplb_executor_->consume_ready_prepare_token();
  }

  if (!enable_schedule_overlap() && !driver_ && !dp_driver_ &&
      !options_.enable_speculative_decode()) {
    MULTI_MODEL_STEP_UNLOCK();
    if (sync_policy == ForwardSyncPolicy::NO_SYNC) {
      wait_kv_push();
      return std::nullopt;
    }
    int ret = device_.synchronize_default_stream();
    CHECK_EQ(ret, 0) << "synchronize_default_stream failed";
    wait_kv_push();
    if (::xllm::EPLBConfig::get_instance().enable_eplb()) {
      return output;
    }
    return std::nullopt;
  }

  // driver prepare model output
  if (sampling_params.selected_token_idxes.defined()) {
    output.logits = logits;
    output.selected_hidden = selected_hidden;
    output.do_sample = sampling_params.do_sample;
    output.logprobs = sampling_params.logprobs;
    output.max_top_logprobs = sampling_params.max_top_logprobs;
    if (!input.skip_sampling_for_logits_only) {
      torch::Tensor fixed_greedy_output;
      if (output_workspace != nullptr) {
        fixed_greedy_output = output_workspace->next_tokens;
      }
      auto sample_output = sampler_->forward(logits,
                                             sampling_params,
                                             /*filter_mask=*/torch::Tensor(),
                                             fixed_greedy_output);
      output.filter_bitmask_applied_to_logits =
          sampling_params.filter_bitmask.defined();

      // beam search kernel
      BeamSearchOutput beam_search_output;
      if (sampling_params.use_beam_search &&
          sampling_params.acc_logprob.defined() &&
          sampling_params.acc_logprob.numel() > 0) {
        beam_search_output =
            beam_searcher_->forward(sampling_params.acc_logprob,
                                    sample_output.top_tokens,
                                    sample_output.top_logprobs);
      }

      // set sample output to output
      output.sample_output = sample_output;
      if (!options_.enable_speculative_decode()) {
        trace_llm_decode_step_state(input, sample_output.next_tokens);
      }
      // set beam search output to output
      output.beam_search_output = beam_search_output;
    }
  }

  if (options_.enable_speculative_decode()) {
    torch::Tensor embeddings;
    if (model_output.aux_hidden_states.defined()) {
      embeddings = model_output.aux_hidden_states;
    } else {
      embeddings = model_output.hidden_states;
    }
    if (!input.input_params.meta.batch_forward_type.is_decode() &&
        !is_spec_draft_) {
      // Target prefill: keep full embeddings (global-real under model-side CP).
      output.sample_output.embeddings = embeddings;
      if (sampling_params.selected_token_idxes.defined() &&
          output_workspace != nullptr &&
          output_workspace->selected_embeddings.defined()) {
        output.sample_output.selected_embeddings =
            gather_prepared_selected_embeddings(
                embeddings,
                sampling_params.selected_token_idxes,
                output_workspace->selected_embeddings);
      }
    } else if (sampling_params.selected_token_idxes.defined()) {
      if (output_workspace != nullptr &&
          output_workspace->selected_embeddings.defined()) {
        output.sample_output.embeddings = gather_prepared_selected_embeddings(
            embeddings,
            sampling_params.selected_token_idxes,
            output_workspace->selected_embeddings);
      } else {
        output.sample_output.embeddings = embeddings.index_select(
            /*dim=*/0, sampling_params.selected_token_idxes);
      }
    }
  }

  if (output_workspace != nullptr) {
    check_prepared_model_output_binding(output.sample_output,
                                        *output_workspace);
  }

  MULTI_MODEL_STEP_UNLOCK();
  bool should_sync_default_stream = true;
#if defined(USE_NPU)
  should_sync_default_stream =
      !can_skip_npu_graph_decode_sync(input.input_params);
#endif
  if (sync_policy == ForwardSyncPolicy::NO_SYNC) {
    wait_kv_push();
    if (retain_input_for_async_output) {
      output.retained_inputs.emplace_back(
          std::make_shared<ForwardInput>(input));
    }
    if (enable_schedule_overlap() && record_ready_event) {
      output.ready_event = record_current_stream_event(device_);
    }
    return output;
  }
  if (should_sync_default_stream) {
    int ret = device_.synchronize_default_stream();
    CHECK_EQ(ret, 0) << "synchronize_default_stream failed";
  }

  wait_kv_push();

  COUNTER_ADD(execution_latency_seconds_model, timer.elapsed_seconds());
  if (should_sync_default_stream) {
    DeviceMonitor::get_instance().update_active_activation_memory(
        device_.index());
  }

  return output;
}

}  // namespace xllm
