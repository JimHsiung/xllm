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

#include "dsa_metadata_builder.h"

#include <algorithm>
#include <cstring>

#include "attention_metadata.h"
#include "attention_metadata_builder.h"
#include "dsa_metadata.h"
#include "framework/model/model_input_params.h"
#include "util/tensor_helper.h"

namespace xllm::layer {

namespace {

torch::Device infer_metadata_device(const ModelInputParams& params,
                                    const torch::Tensor& positions) {
  if (positions.defined()) {
    return positions.device();
  }
  if (params.attention.device.kv_seq_lens.defined()) {
    return params.attention.device.kv_seq_lens.device();
  }
  if (params.attention.device.q_seq_lens.defined()) {
    return params.attention.device.q_seq_lens.device();
  }
  if (params.attention.device.new_cache_slots.defined()) {
    return params.attention.device.new_cache_slots.device();
  }
  if (params.attention.device.block_tables.defined()) {
    return params.attention.device.block_tables.device();
  }
  return torch::Device(torch::kCPU);
}

int64_t vector_max_or_zero(const std::vector<int32_t>& values) {
  if (values.empty()) {
    return 0;
  }
  return *std::max_element(values.begin(), values.end());
}

torch::Tensor pad_block_table(const torch::Tensor& block_table,
                              int32_t target_rows,
                              int32_t target_cols,
                              int32_t pad_value) {
  if (!block_table.defined() || block_table.dim() != 2 ||
      (block_table.size(0) >= target_rows &&
       block_table.size(1) >= target_cols)) {
    return block_table;
  }

  const int64_t rows = std::max<int64_t>(target_rows, block_table.size(0));
  const int64_t cols = std::max<int64_t>(target_cols, block_table.size(1));
  auto padded = torch::full({rows, cols}, pad_value, block_table.options());
  padded.slice(/*dim=*/0, /*start=*/0, /*end=*/block_table.size(0))
      .slice(/*dim=*/1, /*start=*/0, /*end=*/block_table.size(1))
      .copy_(block_table);
  return padded;
}

torch::Tensor fixed_workspace_prefix(const torch::Tensor& storage,
                                     int64_t size,
                                     const torch::Device& device,
                                     torch::ScalarType scalar_type);

torch::Tensor normalize_group_table_rows(
    const torch::Tensor& block_table,
    int64_t batch_size,
    int64_t group_id,
    DSADeviceGeometryWorkspace* workspace) {
  CHECK(block_table.defined());
  CHECK_EQ(block_table.dim(), 2);
  CHECK_GT(block_table.size(0), 0);
  CHECK_GT(block_table.size(1), 0);
  if (block_table.size(0) == batch_size) {
    return block_table;
  }

  CHECK_EQ(batch_size % block_table.size(0), 0)
      << "Prepared DSA logical rows must be a uniform expansion of manager "
         "block-table rows";
  const int64_t repeat_factor = batch_size / block_table.size(0);
  if (workspace == nullptr) {
    torch::Tensor row_indices =
        torch::arange(batch_size, block_table.options().dtype(torch::kLong));
    torch::floor_divide_out(row_indices, row_indices, repeat_factor);
    return block_table.index_select(/*dim=*/0, row_indices);
  }

  CHECK_GE(group_id, 0);
  CHECK_LT(static_cast<size_t>(group_id),
           workspace->manager_expanded_block_tables.size());
  torch::Tensor row_indices =
      fixed_workspace_prefix(workspace->manager_row_indices,
                             batch_size,
                             block_table.device(),
                             torch::kLong);
  torch::arange_out(row_indices, batch_size);
  torch::floor_divide_out(row_indices, row_indices, repeat_factor);
  const torch::Tensor& expanded_storage =
      workspace->manager_expanded_block_tables[static_cast<size_t>(group_id)];
  torch::Tensor expanded_table =
      fixed_workspace_prefix(expanded_storage,
                             batch_size * block_table.size(1),
                             block_table.device(),
                             block_table.scalar_type())
          .view({batch_size, block_table.size(1)});
  torch::index_select_out(expanded_table, block_table, /*dim=*/0, row_indices);
  return expanded_table;
}

torch::Tensor fixed_workspace_prefix(const torch::Tensor& storage,
                                     int64_t size,
                                     const torch::Device& device,
                                     torch::ScalarType scalar_type) {
  CHECK(storage.defined());
  CHECK_EQ(storage.dim(), 1);
  CHECK_EQ(storage.device(), device);
  CHECK_EQ(storage.scalar_type(), scalar_type);
  CHECK(storage.is_contiguous());
  CHECK_GE(storage.numel(), size)
      << "Prepared DSA scratch exceeds fixed Device capacity";
  return storage.narrow(/*dim=*/0, /*start=*/0, /*length=*/size);
}

torch::Tensor compact_boundary_rows(
    const torch::Tensor& boundary_mask,
    const torch::Tensor& candidate_values,
    int64_t capacity,
    int64_t pad_value,
    const torch::Tensor& packed_storage = {},
    DSADeviceGeometryWorkspace* workspace = nullptr) {
  CHECK_EQ(boundary_mask.dim(), 1);
  CHECK_EQ(boundary_mask.scalar_type(), torch::kBool);
  CHECK_EQ(candidate_values.dim(), 1);
  CHECK_EQ(candidate_values.numel(), boundary_mask.numel());
  CHECK_GE(capacity, 0);
  if (capacity == 0) {
    if (packed_storage.defined()) {
      return packed_storage.narrow(/*dim=*/0, /*start=*/0, /*length=*/0);
    }
    return torch::empty({0}, candidate_values.options());
  }

  torch::Tensor boundary_ranks;
  torch::Tensor sentinel_indices;
  torch::Tensor destination_indices;
  torch::Tensor source_values;
  if (workspace == nullptr) {
    boundary_ranks =
        torch::cumsum(boundary_mask.to(torch::kLong), /*dim=*/0) - 1;
    // Give every non-boundary row a unique discarded destination. Some Device
    // scatter implementations do not define duplicate-index behavior even
    // when all duplicate writes carry the same padding value.
    sentinel_indices =
        torch::arange(boundary_mask.numel(), boundary_ranks.options()) +
        capacity;
    destination_indices =
        torch::where(boundary_mask, boundary_ranks, sentinel_indices);
    torch::Tensor padding = torch::full_like(candidate_values, pad_value);
    source_values = torch::where(boundary_mask, candidate_values, padding);
  } else {
    const int64_t token_count = boundary_mask.numel();
    boundary_ranks = fixed_workspace_prefix(workspace->boundary_ranks,
                                            token_count,
                                            boundary_mask.device(),
                                            torch::kLong);
    sentinel_indices = fixed_workspace_prefix(workspace->sentinel_indices,
                                              token_count,
                                              boundary_mask.device(),
                                              torch::kLong);
    destination_indices = fixed_workspace_prefix(workspace->destination_indices,
                                                 token_count,
                                                 boundary_mask.device(),
                                                 torch::kLong);
    torch::Tensor token_indices =
        fixed_workspace_prefix(workspace->token_indices,
                               token_count,
                               boundary_mask.device(),
                               torch::kLong);
    torch::cumsum_out(boundary_ranks,
                      boundary_mask,
                      /*dim=*/0,
                      /*dtype=*/torch::kLong);
    boundary_ranks.sub_(1);
    sentinel_indices.copy_(token_indices);
    sentinel_indices.add_(capacity);
    torch::where_out(
        destination_indices, boundary_mask, boundary_ranks, sentinel_indices);
    source_values = candidate_values;
    torch::Tensor non_boundary_mask = boundary_mask;
    non_boundary_mask.logical_not_();
    source_values.masked_fill_(non_boundary_mask, pad_value);
  }
  const int64_t packed_size = capacity + boundary_mask.numel();
  torch::Tensor packed;
  if (packed_storage.defined()) {
    CHECK_EQ(packed_storage.dim(), 1);
    CHECK_EQ(packed_storage.device(), candidate_values.device());
    CHECK_EQ(packed_storage.scalar_type(), candidate_values.scalar_type());
    CHECK(packed_storage.is_contiguous());
    CHECK_GE(packed_storage.numel(), packed_size)
        << "Prepared DSA compact output exceeds fixed Device capacity";
    packed = packed_storage.narrow(
        /*dim=*/0, /*start=*/0, /*length=*/packed_size);
    packed.fill_(pad_value);
  } else {
    packed = torch::full({packed_size}, pad_value, candidate_values.options());
  }
  packed.scatter_(/*dim=*/0, destination_indices, source_values);
  return packed.narrow(/*dim=*/0, /*start=*/0, capacity);
}

int64_t compressed_row_capacity(int64_t num_tokens,
                                int64_t batch_size,
                                int32_t ratio) {
  CHECK_GT(ratio, 0);
  return std::min<int64_t>(num_tokens, num_tokens / ratio + batch_size);
}

torch::Tensor build_device_compressed_positions(
    const torch::Tensor& positions,
    int64_t batch_size,
    int32_t ratio,
    const torch::Tensor& packed_storage = {},
    DSADeviceGeometryWorkspace* workspace = nullptr) {
  CHECK(positions.defined());
  CHECK_EQ(positions.dim(), 1);
  torch::Tensor next_positions;
  torch::Tensor boundary_mask;
  torch::Tensor compressed_positions;
  if (workspace == nullptr) {
    next_positions = positions + 1;
    boundary_mask = torch::remainder(next_positions, ratio).eq(0);
    compressed_positions = next_positions - ratio;
  } else {
    const int64_t token_count = positions.numel();
    next_positions = fixed_workspace_prefix(workspace->position_values,
                                            token_count,
                                            positions.device(),
                                            positions.scalar_type());
    torch::Tensor remainders =
        fixed_workspace_prefix(workspace->position_remainders,
                               token_count,
                               positions.device(),
                               positions.scalar_type());
    boundary_mask = fixed_workspace_prefix(workspace->boundary_mask,
                                           token_count,
                                           positions.device(),
                                           torch::kBool);
    next_positions.copy_(positions);
    next_positions.add_(1);
    torch::remainder_out(remainders, next_positions, ratio);
    torch::eq_out(boundary_mask, remainders, 0);
    next_positions.sub_(ratio);
    compressed_positions = next_positions;
  }
  return compact_boundary_rows(
      boundary_mask,
      compressed_positions,
      compressed_row_capacity(positions.numel(), batch_size, ratio),
      /*pad_value=*/0,
      packed_storage,
      workspace);
}

torch::Tensor build_device_token_slots(
    const torch::Tensor& positions,
    const torch::Tensor& block_table,
    int32_t ratio,
    int32_t block_size,
    const torch::Tensor& packed_storage = {},
    DSADeviceGeometryWorkspace* workspace = nullptr) {
  CHECK(positions.defined());
  CHECK_EQ(positions.dim(), 1);
  CHECK_EQ(block_table.dim(), 2);
  CHECK_GT(block_table.size(0), 0);
  CHECK_GT(block_table.size(1), 0);
  CHECK_GT(ratio, 0);
  CHECK_GT(block_size, 0);
  CHECK_EQ(positions.numel() % block_table.size(0), 0)
      << "Prepared DSA token rows must be uniform per logical sequence";

  const int64_t batch_size = block_table.size(0);
  const int64_t query_width = positions.numel() / batch_size;
  if (workspace != nullptr) {
    const int64_t token_count = positions.numel();
    torch::Tensor compressed_indices =
        fixed_workspace_prefix(workspace->position_values,
                               token_count,
                               positions.device(),
                               positions.scalar_type());
    torch::Tensor remainders =
        fixed_workspace_prefix(workspace->position_remainders,
                               token_count,
                               positions.device(),
                               positions.scalar_type());
    torch::Tensor boundary_mask =
        fixed_workspace_prefix(workspace->boundary_mask,
                               token_count,
                               positions.device(),
                               torch::kBool);
    torch::Tensor block_indices =
        fixed_workspace_prefix(workspace->boundary_ranks,
                               token_count,
                               positions.device(),
                               torch::kLong);
    torch::Tensor safe_block_indices =
        fixed_workspace_prefix(workspace->sentinel_indices,
                               token_count,
                               positions.device(),
                               torch::kLong);
    torch::Tensor flat_block_indices =
        fixed_workspace_prefix(workspace->destination_indices,
                               token_count,
                               positions.device(),
                               torch::kLong);
    torch::Tensor token_indices =
        fixed_workspace_prefix(workspace->token_indices,
                               token_count,
                               positions.device(),
                               torch::kLong);
    torch::Tensor block_offsets =
        fixed_workspace_prefix(workspace->token_offsets,
                               token_count,
                               block_table.device(),
                               block_table.scalar_type());
    torch::Tensor candidate_slots =
        fixed_workspace_prefix(workspace->token_candidates,
                               token_count,
                               block_table.device(),
                               block_table.scalar_type());
    torch::Tensor mapping_valid =
        fixed_workspace_prefix(workspace->mapping_valid,
                               token_count,
                               positions.device(),
                               torch::kBool);
    torch::Tensor mapping_valid_aux =
        fixed_workspace_prefix(workspace->mapping_valid_aux,
                               token_count,
                               positions.device(),
                               torch::kBool);

    compressed_indices.copy_(positions);
    compressed_indices.add_(1);
    torch::remainder_out(remainders, compressed_indices, ratio);
    torch::eq_out(boundary_mask, remainders, 0);
    torch::floor_divide_out(compressed_indices, compressed_indices, ratio);
    compressed_indices.sub_(1);

    block_indices.copy_(compressed_indices);
    torch::floor_divide_out(block_indices, block_indices, block_size);
    safe_block_indices.copy_(block_indices);
    safe_block_indices.clamp_(/*min=*/0,
                              /*max=*/block_table.size(1) - 1);

    CHECK_EQ(block_table.stride(1), 1)
        << "Prepared DSA token manager requires contiguous table columns";
    const int64_t row_stride = block_table.stride(0);
    const int64_t physical_span =
        (block_table.size(0) - 1) * row_stride + block_table.size(1);
    torch::Tensor flat_block_table =
        block_table.as_strided({physical_span}, {1});
    flat_block_indices.copy_(token_indices);
    torch::floor_divide_out(
        flat_block_indices, flat_block_indices, query_width);
    flat_block_indices.mul_(row_stride);
    flat_block_indices.add_(safe_block_indices);
    torch::index_select_out(candidate_slots,
                            flat_block_table,
                            /*dim=*/0,
                            flat_block_indices);

    block_offsets.copy_(compressed_indices);
    block_offsets.remainder_(block_size);
    torch::ge_out(mapping_valid, block_indices, 0);
    torch::lt_out(mapping_valid_aux, block_indices, block_table.size(1));
    mapping_valid.logical_and_(mapping_valid_aux);
    torch::ge_out(mapping_valid_aux, candidate_slots, 0);
    mapping_valid.logical_and_(mapping_valid_aux);
    candidate_slots.mul_(block_size);
    candidate_slots.add_(block_offsets);
    mapping_valid.logical_not_();
    candidate_slots.masked_fill_(mapping_valid, -1);

    return compact_boundary_rows(
        boundary_mask,
        candidate_slots,
        compressed_row_capacity(positions.numel(), batch_size, ratio),
        /*pad_value=*/-1,
        packed_storage,
        workspace);
  }

  torch::Tensor next_positions = positions + 1;
  torch::Tensor boundary_mask = torch::remainder(next_positions, ratio).eq(0);
  torch::Tensor compressed_indices =
      torch::floor_divide(next_positions, ratio) - 1;
  torch::Tensor block_indices =
      torch::floor_divide(compressed_indices, block_size).to(torch::kLong);
  torch::Tensor safe_block_indices =
      block_indices.clamp(/*min=*/0, /*max=*/block_table.size(1) - 1);

  torch::Tensor sequence_indices = torch::arange(
      positions.numel(), block_table.options().dtype(torch::kLong));
  torch::floor_divide_out(sequence_indices, sequence_indices, query_width);
  torch::Tensor per_token_tables =
      block_table.index_select(/*dim=*/0, sequence_indices);
  torch::Tensor block_ids =
      per_token_tables
          .gather(/*dim=*/1, safe_block_indices.unsqueeze(/*dim=*/1))
          .squeeze(/*dim=*/1);
  torch::Tensor block_offsets = torch::remainder(compressed_indices, block_size)
                                    .to(block_table.scalar_type());
  torch::Tensor candidate_slots = block_ids * block_size + block_offsets;
  torch::Tensor mapping_valid =
      block_indices.ge(0).logical_and(block_indices.lt(block_table.size(1)));
  mapping_valid.logical_and_(block_ids.ge(0));
  candidate_slots = torch::where(
      mapping_valid, candidate_slots, torch::full_like(candidate_slots, -1));

  return compact_boundary_rows(
      boundary_mask,
      candidate_slots,
      compressed_row_capacity(positions.numel(), batch_size, ratio),
      /*pad_value=*/-1,
      packed_storage,
      workspace);
}

torch::Tensor build_device_swa_block_table(
    const torch::Tensor& raw_block_table,
    const torch::Tensor& kv_seq_lens,
    int32_t block_size,
    int64_t max_seq_len,
    const torch::Tensor& output_storage = {},
    DSADeviceGeometryWorkspace* workspace = nullptr) {
  CHECK_EQ(raw_block_table.dim(), 2);
  CHECK_EQ(kv_seq_lens.dim(), 1);
  CHECK_EQ(raw_block_table.size(0), kv_seq_lens.numel());
  CHECK_GT(raw_block_table.size(1), 0);
  CHECK_GT(block_size, 0);
  CHECK_GE(max_seq_len, 0);

  const int64_t max_logical_cols =
      std::max<int64_t>((max_seq_len + block_size - 1) / block_size, 1);
  torch::Tensor gathered;
  torch::Tensor valid;
  if (workspace == nullptr) {
    torch::Tensor logical_blocks =
        torch::floor_divide(kv_seq_lens + block_size - 1, block_size)
            .clamp(/*min=*/0, /*max=*/max_logical_cols)
            .to(torch::kLong);
    torch::Tensor retained_blocks =
        logical_blocks.clamp_max(raw_block_table.size(1));
    torch::Tensor first_retained = logical_blocks - retained_blocks;
    torch::Tensor logical_columns =
        torch::arange(max_logical_cols,
                      raw_block_table.options().dtype(torch::kLong))
            .view({1, max_logical_cols})
            .expand({kv_seq_lens.numel(), max_logical_cols});
    torch::Tensor physical_columns =
        torch::remainder(logical_columns, raw_block_table.size(1));
    gathered = raw_block_table.gather(/*dim=*/1, physical_columns);
    valid = logical_columns.ge(first_retained.unsqueeze(/*dim=*/1));
    valid.logical_and_(logical_columns.lt(logical_blocks.unsqueeze(/*dim=*/1)));
  } else {
    const int64_t batch_size = kv_seq_lens.numel();
    const int64_t matrix_size = batch_size * max_logical_cols;
    torch::Tensor logical_blocks =
        fixed_workspace_prefix(workspace->boundary_ranks,
                               batch_size,
                               kv_seq_lens.device(),
                               torch::kLong);
    torch::Tensor retained_blocks =
        fixed_workspace_prefix(workspace->sentinel_indices,
                               batch_size,
                               kv_seq_lens.device(),
                               torch::kLong);
    torch::Tensor first_retained =
        fixed_workspace_prefix(workspace->destination_indices,
                               batch_size,
                               kv_seq_lens.device(),
                               torch::kLong);
    torch::Tensor logical_column_indices =
        fixed_workspace_prefix(workspace->swa_logical_column_indices,
                               max_logical_cols,
                               kv_seq_lens.device(),
                               torch::kLong);
    torch::arange_out(logical_column_indices, max_logical_cols);
    torch::Tensor logical_columns =
        logical_column_indices.view({1, max_logical_cols})
            .expand({batch_size, max_logical_cols});
    torch::Tensor physical_columns =
        fixed_workspace_prefix(workspace->swa_physical_columns,
                               matrix_size,
                               kv_seq_lens.device(),
                               torch::kLong)
            .view({batch_size, max_logical_cols});
    gathered = fixed_workspace_prefix(workspace->swa_gathered,
                                      matrix_size,
                                      raw_block_table.device(),
                                      raw_block_table.scalar_type())
                   .view({batch_size, max_logical_cols});
    valid = fixed_workspace_prefix(workspace->swa_valid,
                                   matrix_size,
                                   raw_block_table.device(),
                                   torch::kBool)
                .view({batch_size, max_logical_cols});
    torch::Tensor valid_aux = fixed_workspace_prefix(workspace->swa_valid_aux,
                                                     matrix_size,
                                                     raw_block_table.device(),
                                                     torch::kBool)
                                  .view({batch_size, max_logical_cols});

    logical_blocks.copy_(kv_seq_lens);
    logical_blocks.add_(block_size - 1);
    torch::floor_divide_out(logical_blocks, logical_blocks, block_size);
    logical_blocks.clamp_(/*min=*/0, /*max=*/max_logical_cols);
    retained_blocks.copy_(logical_blocks);
    retained_blocks.clamp_max_(raw_block_table.size(1));
    first_retained.copy_(logical_blocks);
    first_retained.sub_(retained_blocks);
    physical_columns.copy_(logical_columns);
    physical_columns.remainder_(raw_block_table.size(1));
    torch::gather_out(gathered,
                      raw_block_table,
                      /*dim=*/1,
                      physical_columns);
    torch::ge_out(valid, logical_columns, first_retained.unsqueeze(/*dim=*/1));
    torch::lt_out(
        valid_aux, logical_columns, logical_blocks.unsqueeze(/*dim=*/1));
    valid.logical_and_(valid_aux);
  }
  if (!output_storage.defined()) {
    return torch::where(valid, gathered, torch::full_like(gathered, -1));
  }

  CHECK_EQ(output_storage.dim(), 1);
  CHECK_EQ(output_storage.device(), raw_block_table.device());
  CHECK_EQ(output_storage.scalar_type(), raw_block_table.scalar_type());
  CHECK(output_storage.is_contiguous());
  const int64_t output_size = raw_block_table.size(0) * max_logical_cols;
  CHECK_GE(output_storage.numel(), output_size)
      << "Prepared DSA SWA logical table exceeds fixed Device capacity";
  torch::Tensor output =
      output_storage.narrow(/*dim=*/0, /*start=*/0, output_size)
          .view({raw_block_table.size(0), max_logical_cols});
  output.copy_(gathered);
  valid.logical_not_();
  output.masked_fill_(valid, -1);
  return output;
}

}  // namespace

void validate_dsa_group_order(const std::vector<DSAGroupInfo>& group_infos) {
  if (group_infos.empty()) {
    return;
  }

  const DSAGroupInfo& swa_group = group_infos.front();
  CHECK(swa_group.type == DSACacheType::SLIDING_WINDOW)
      << "DSA manager 0 must be the SWA group";
  CHECK_EQ(swa_group.ratio, 1)
      << "DSA SWA manager must use compression ratio 1";
  CHECK_GT(swa_group.block_size, 0);

  int32_t previous_ratio = 1;
  for (size_t group_id = 1; group_id < group_infos.size(); ++group_id) {
    const DSAGroupInfo& group_info = group_infos[group_id];
    CHECK(group_info.type == DSACacheType::TOKEN)
        << "DSA manager " << group_id << " must be a token-compression group";
    CHECK(group_info.ratio == 4 || group_info.ratio == 128)
        << "DSA token manager " << group_id
        << " must use compression ratio 4 or 128";
    CHECK_GT(group_info.ratio, previous_ratio)
        << "DSA token managers must follow canonical C4/C128 order";
    CHECK_GT(group_info.block_size, 0);
    previous_ratio = group_info.ratio;
  }
}

AttentionMetadata DSAMetadataBuilder::build(
    const ModelInputParams& params,
    const torch::Tensor& positions,
    const torch::Tensor& dsa_cos_sin,
    const std::vector<std::vector<DSACacheInfo>>& caches_info,
    const std::vector<DSAGroupInfo>& group_infos,
    const torch::Tensor& dsa_c4_cos_sin,
    const torch::Tensor& dsa_c128_cos_sin) {
  validate_dsa_group_order(group_infos);

  // 1. Build base AttentionMetadata (q_cu_seq_lens, block_table, etc.)
  AttentionMetadata attn_metadata =
      AttentionMetadataBuilder::build(params,
                                      /*enable_mla=*/false,
                                      /*attn_mask=*/{},
                                      infer_metadata_device(params, positions));

  // 2. Build DSA-specific fields
  auto dsa_metadata = std::make_shared<DSAMetadata>();
  build_dsa_fields(params,
                   positions,
                   dsa_cos_sin,
                   dsa_c4_cos_sin,
                   dsa_c128_cos_sin,
                   caches_info,
                   group_infos,
                   *dsa_metadata);

  // 3. Keep DSA metadata independent while syncing base attention tensors.
  if (attn_metadata.attn_mask.defined()) {
    dsa_metadata->attn_mask = attn_metadata.attn_mask.clone();
  }

  if (attn_metadata.mrope_cos.defined() && !dsa_metadata->cos_table.defined()) {
    dsa_metadata->cos_table = attn_metadata.mrope_cos;
  }
  if (attn_metadata.mrope_sin.defined() && !dsa_metadata->sin_table.defined()) {
    dsa_metadata->sin_table = attn_metadata.mrope_sin;
  }

  // 4. Attach to AttentionMetadata
  attn_metadata.dsa_metadata = std::move(dsa_metadata);

  return attn_metadata;
}

void DSAMetadataBuilder::build_dsa_fields(
    const ModelInputParams& params,
    const torch::Tensor& positions,
    const torch::Tensor& dsa_cos_sin,
    const torch::Tensor& dsa_c4_cos_sin,
    const torch::Tensor& dsa_c128_cos_sin,
    const std::vector<std::vector<DSACacheInfo>>& caches_info,
    const std::vector<DSAGroupInfo>& group_infos,
    DSAMetadata& dsa) {
  dsa.input_positions = positions;
  const bool is_acl_graph = params.enable_graph;
  dsa.is_acl_graph = is_acl_graph;
  dsa.device_geometry_authoritative = params.dsa_device_geometry_authoritative;

  // Keep base RoPE tables in metadata. Per-forward cos/sin slices are
  // calculated in DeepseekV4ModelImpl::forward to align with MindIE timing.
  if (dsa_cos_sin.defined()) {
    auto cos_sin_chunks = dsa_cos_sin.chunk(/*chunks=*/2, /*dim=*/-1);
    dsa.cos_table = cos_sin_chunks[0].contiguous();
    dsa.sin_table = cos_sin_chunks[1].contiguous();
  }

  (void)dsa_c4_cos_sin;
  (void)dsa_c128_cos_sin;

  if (params.dsa_device_geometry_authoritative) {
    CHECK(params.dsa_device_geometry_workspace != nullptr)
        << "Prepared authoritative DSA geometry requires fixed workspace";
    CHECK(!params.device_multi_block_tables.empty())
        << "Prepared authoritative DSA geometry requires Device manager "
           "tables";
    // Continuation has already advanced the fixed Device q/KV lengths and
    // manager tables. Avoid constructing stale Host geometry that is replaced
    // by patch_device_geometry() in model forward. ACL graph callers use the
    // same fixed workspace: capture records the out/in-place geometry patch,
    // and replay updates the captured addresses from current Slot inputs.
    dsa.caches_info = &caches_info;
    return;
  }

  const int32_t batch_size =
      static_cast<int32_t>(params.attention.host.kv_seq_lens.size());
  std::vector<int32_t> q_lens_vec;
  q_lens_vec.reserve(batch_size);
  const torch::Device metadata_device(torch::kCPU);

  // Build per-batch sequence length metadata.
  build_seq_lengths(params, metadata_device, batch_size, dsa);
  if (static_cast<int32_t>(params.attention.host.q_seq_lens.size()) ==
      batch_size) {
    q_lens_vec.assign(params.attention.host.q_seq_lens.begin(),
                      params.attention.host.q_seq_lens.end());
  } else if (params.meta.batch_forward_type.no_decode()) {
    q_lens_vec.assign(params.attention.host.kv_seq_lens.begin(),
                      params.attention.host.kv_seq_lens.end());
  } else {
    q_lens_vec.assign(batch_size, 1);
  }

  if (positions.defined()) {
    build_positions(params, batch_size, dsa);
  }

  // --- Block tables / slots expansion ---
  if (!params.multi_block_tables.empty() && !caches_info.empty()) {
    std::vector<torch::Tensor> active_multi_block_tables =
        params.multi_block_tables;
    int32_t manager_num =
        static_cast<int32_t>(active_multi_block_tables.size());
    if (manager_num == 1 && batch_size == 1 &&
        active_multi_block_tables[0].defined() &&
        active_multi_block_tables[0].dim() == 2 &&
        active_multi_block_tables[0].size(0) > 1 &&
        static_cast<size_t>(active_multi_block_tables[0].size(0)) <=
            group_infos.size()) {
      const auto packed = active_multi_block_tables[0].contiguous();
      std::vector<torch::Tensor> unpacked_tables;
      unpacked_tables.reserve(packed.size(0));
      for (int64_t m = 0; m < packed.size(0); ++m) {
        unpacked_tables.push_back(packed[m].unsqueeze(0).contiguous());
      }
      active_multi_block_tables = std::move(unpacked_tables);
      manager_num = static_cast<int32_t>(active_multi_block_tables.size());
      LOG(WARNING)
          << "DSAMetadataBuilder detected packed multi_block_tables layout "
             "([manager, blocks]) while batch_size==1; auto-unpacked to "
             "[manager][batch, blocks]. manager_num="
          << manager_num;
    }

    CHECK_EQ(batch_size,
             static_cast<int32_t>(params.attention.host.kv_seq_lens.size()))
        << "DSAMetadataBuilder: batch_size mismatch with kv_seq_lens_vec size.";
    CHECK_LE(manager_num, static_cast<int32_t>(group_infos.size()))
        << "DSAMetadataBuilder: manager_num(" << manager_num
        << ") exceeds group_infos size(" << group_infos.size()
        << "), cannot align manager/group mapping.";
    int32_t graph_block_table_capacity_cols = 0;
    if (is_acl_graph && params.attention.device.block_tables.defined() &&
        params.attention.device.block_tables.dim() == 2) {
      graph_block_table_capacity_cols =
          static_cast<int32_t>(params.attention.device.block_tables.size(1));
    }
    if (is_acl_graph && graph_block_table_capacity_cols > 0) {
      for (int32_t m = 0; m < manager_num; ++m) {
        const auto& block_table = active_multi_block_tables[m];
        CHECK(block_table.defined() && block_table.dim() == 2)
            << "DSAMetadataBuilder: ACL graph multi_block_tables manager " << m
            << " must be a 2-D tensor.";
        CHECK_LE(block_table.size(1), graph_block_table_capacity_cols)
            << "DSAMetadataBuilder: ACL graph multi_block_tables exceeds "
            << "bucket column capacity: manager_id=" << m
            << ", required_cols=" << block_table.size(1)
            << ", capacity_cols=" << graph_block_table_capacity_cols;
      }
    }
    const int32_t n_layers = static_cast<int32_t>(caches_info.size());
    const auto& ctx_lens = params.attention.host.kv_seq_lens;
    const int64_t graph_slot_capacity =
        is_acl_graph && positions.defined() ? positions.numel() : 0;
    int64_t total_tokens = 0;
    for (int32_t len : ctx_lens) {
      total_tokens += len;
    }

    std::vector<torch::Tensor> proc_slots(manager_num);
    std::vector<torch::Tensor> proc_bt(manager_num);
    for (int32_t m = 0; m < manager_num; ++m) {
      const torch::Tensor& device_cache_slots =
          params.attention.device.new_cache_slots;
      if (params.dsa_device_geometry_authoritative) {
        CHECK_LT(static_cast<size_t>(m),
                 params.device_multi_block_tables.size())
            << "Prepared DSA manager table is unavailable on Device";
        proc_bt[m] = params.device_multi_block_tables[m];
        CHECK(proc_bt[m].defined());
        CHECK_EQ(proc_bt[m].dim(), 2);
        CHECK_EQ(proc_bt[m].scalar_type(), torch::kInt32);
        if (group_infos[m].type == DSACacheType::SLIDING_WINDOW) {
          CHECK(device_cache_slots.defined());
          proc_slots[m] = device_cache_slots;
        }
        continue;
      }

      process_group(active_multi_block_tables[m],
                    group_infos[m],
                    ctx_lens,
                    q_lens_vec,
                    params.attention.host.new_cache_slots,
                    batch_size,
                    total_tokens,
                    graph_slot_capacity,
                    graph_block_table_capacity_cols,
                    proc_bt[m],
                    proc_slots[m]);
      if (group_infos[m].type == DSACacheType::SLIDING_WINDOW &&
          device_cache_slots.defined() && device_cache_slots.dim() == 1 &&
          device_cache_slots.is_contiguous() &&
          device_cache_slots.numel() == proc_slots[m].numel()) {
        CHECK_EQ(device_cache_slots.scalar_type(), proc_slots[m].scalar_type())
            << "DSA SWA cache-slot dtype mismatch";
        // Prepared speculative geometry is patched in fixed Device storage
        // after Host preparation. Reuse that storage directly so the DSA
        // model consumes the patched ring slots instead of repacking stale
        // Host values. Legacy inputs take the same path when their already
        // uploaded slot tensor has the exact semantic width.
        proc_slots[m] = device_cache_slots;
      }
    }

    // Keep expanded metadata on host. DeepSeek V4 packs these small tensors
    // into one contiguous transfer for both graph and non-graph NPU forwards.

    // Step 3: expand by layer using group_id
    dsa.block_tables.resize(n_layers);
    dsa.slot_mappings.resize(n_layers);
    for (int32_t lid = 0; lid < n_layers; ++lid) {
      const auto& lci = caches_info[lid];
      dsa.block_tables[lid].resize(lci.size());
      dsa.slot_mappings[lid].resize(lci.size());
      for (size_t ci = 0; ci < lci.size(); ++ci) {
        int32_t gid = lci[ci].group_id;
        if (gid < manager_num) {
          dsa.block_tables[lid][ci] = proc_bt[gid];
          dsa.slot_mappings[lid][ci] = proc_slots[gid];
        }
      }
    }
  }

  // Attach cache spec pointer
  dsa.caches_info = &caches_info;
}

void DSAMetadataBuilder::patch_device_geometry(
    const ModelInputParams& params,
    const std::vector<DSAGroupInfo>& group_infos,
    DSAMetadata& dsa) {
  if (!params.dsa_device_geometry_authoritative) {
    return;
  }
  validate_dsa_group_order(group_infos);

  const torch::Tensor& kv_seq_lens = params.attention.device.kv_seq_lens;
  const torch::Tensor& q_seq_lens = params.attention.device.q_seq_lens;
  CHECK(kv_seq_lens.defined());
  CHECK(q_seq_lens.defined());
  CHECK(dsa.input_positions.defined());
  CHECK_EQ(kv_seq_lens.dim(), 1);
  CHECK_EQ(q_seq_lens.dim(), 1);
  CHECK_EQ(kv_seq_lens.scalar_type(), torch::kInt32);
  CHECK_EQ(q_seq_lens.scalar_type(), torch::kInt32);
  CHECK_EQ(kv_seq_lens.numel(), q_seq_lens.numel());
  CHECK_GT(kv_seq_lens.numel(), 0);
  CHECK_EQ(dsa.input_positions.dim(), 1);
  CHECK_GT(dsa.input_positions.numel(), 0);
  CHECK_EQ(dsa.input_positions.numel() % kv_seq_lens.numel(), 0)
      << "Prepared DSA requires a uniform fixed query width";
  CHECK_EQ(kv_seq_lens.device(), q_seq_lens.device());
  CHECK_EQ(kv_seq_lens.device(), dsa.input_positions.device());
  CHECK(dsa.caches_info != nullptr);

  dsa.seq_lens = kv_seq_lens;
  dsa.actual_seq_lengths_kv = kv_seq_lens;
  dsa.seq_lens_q = q_seq_lens;
  DSADeviceGeometryWorkspace* geometry_workspace =
      params.dsa_device_geometry_workspace.get();
  if (geometry_workspace != nullptr) {
    dsa.start_pos = fixed_workspace_prefix(geometry_workspace->start_pos,
                                           kv_seq_lens.numel(),
                                           kv_seq_lens.device(),
                                           torch::kInt32);
    torch::sub_out(dsa.start_pos, kv_seq_lens, q_seq_lens);
  }
  if (geometry_workspace == nullptr) {
    torch::Tensor int_zero = torch::zeros({1}, q_seq_lens.options());
    dsa.actual_seq_lengths_query = torch::cat(
        {int_zero,
         torch::cumsum(q_seq_lens, /*dim=*/0, /*dtype=*/torch::kInt32)});
    dsa.kv_cu_seq_lens = torch::cat(
        {torch::zeros({1}, kv_seq_lens.options()),
         torch::cumsum(kv_seq_lens, /*dim=*/0, /*dtype=*/torch::kInt32)});
    dsa.max_seqlen_q = torch::max(q_seq_lens).to(torch::kInt32).reshape({1});
    dsa.max_seqlen_kv = torch::max(kv_seq_lens).to(torch::kInt32).reshape({1});
  } else {
    DSADeviceGeometryWorkspace& workspace = *geometry_workspace;
    const int64_t cumulative_size = kv_seq_lens.numel() + 1;
    CHECK(workspace.actual_seq_lengths_query.defined());
    CHECK(workspace.kv_cu_seq_lens.defined());
    CHECK(workspace.max_seqlen_q.defined());
    CHECK(workspace.max_seqlen_kv.defined());
    CHECK_EQ(workspace.actual_seq_lengths_query.device(), q_seq_lens.device());
    CHECK_EQ(workspace.kv_cu_seq_lens.device(), kv_seq_lens.device());
    CHECK_EQ(workspace.max_seqlen_q.device(), q_seq_lens.device());
    CHECK_EQ(workspace.max_seqlen_kv.device(), kv_seq_lens.device());
    CHECK_EQ(workspace.actual_seq_lengths_query.scalar_type(), torch::kInt32);
    CHECK_EQ(workspace.kv_cu_seq_lens.scalar_type(), torch::kInt32);
    CHECK_EQ(workspace.max_seqlen_q.scalar_type(), torch::kInt32);
    CHECK_EQ(workspace.max_seqlen_kv.scalar_type(), torch::kInt32);
    CHECK(workspace.actual_seq_lengths_query.is_contiguous());
    CHECK(workspace.kv_cu_seq_lens.is_contiguous());
    CHECK(workspace.max_seqlen_q.is_contiguous());
    CHECK(workspace.max_seqlen_kv.is_contiguous());
    CHECK_GE(workspace.actual_seq_lengths_query.numel(), cumulative_size);
    CHECK_GE(workspace.kv_cu_seq_lens.numel(), cumulative_size);
    CHECK_GE(workspace.max_seqlen_q.numel(), 1);
    CHECK_GE(workspace.max_seqlen_kv.numel(), 1);

    dsa.actual_seq_lengths_query = workspace.actual_seq_lengths_query.narrow(
        /*dim=*/0, /*start=*/0, cumulative_size);
    dsa.kv_cu_seq_lens = workspace.kv_cu_seq_lens.narrow(
        /*dim=*/0, /*start=*/0, cumulative_size);
    dsa.max_seqlen_q = workspace.max_seqlen_q.narrow(
        /*dim=*/0, /*start=*/0, /*length=*/1);
    dsa.max_seqlen_kv = workspace.max_seqlen_kv.narrow(
        /*dim=*/0, /*start=*/0, /*length=*/1);
    dsa.actual_seq_lengths_query.select(/*dim=*/0, /*index=*/0).zero_();
    dsa.kv_cu_seq_lens.select(/*dim=*/0, /*index=*/0).zero_();
    torch::Tensor query_cumsum_output = dsa.actual_seq_lengths_query.narrow(
        /*dim=*/0, /*start=*/1, q_seq_lens.numel());
    torch::Tensor kv_cumsum_output = dsa.kv_cu_seq_lens.narrow(
        /*dim=*/0, /*start=*/1, kv_seq_lens.numel());
    torch::cumsum_out(query_cumsum_output,
                      q_seq_lens,
                      /*dim=*/0,
                      /*dtype=*/torch::kInt32);
    torch::cumsum_out(kv_cumsum_output,
                      kv_seq_lens,
                      /*dim=*/0,
                      /*dtype=*/torch::kInt32);
    torch::amax_out(dsa.max_seqlen_q,
                    q_seq_lens,
                    /*dim=*/{0},
                    /*keepdim=*/true);
    torch::amax_out(dsa.max_seqlen_kv,
                    kv_seq_lens,
                    /*dim=*/{0},
                    /*keepdim=*/true);
  }

  const int64_t query_width = dsa.input_positions.numel() / kv_seq_lens.numel();
  dsa.max_query_len = std::max<int64_t>(
      {dsa.max_query_len, params.meta.q_max_seq_len, query_width});
  const int64_t host_kv_max =
      std::max<int64_t>(params.meta.kv_max_seq_len,
                        vector_max_or_zero(params.attention.host.kv_seq_lens));
  CHECK_GE(params.dsa_device_geometry_kv_headroom, 0);
  dsa.max_seq_len = std::max<int64_t>(
      dsa.max_seq_len, host_kv_max + params.dsa_device_geometry_kv_headroom);

  const int64_t batch_size = kv_seq_lens.numel();
  if (geometry_workspace != nullptr) {
    torch::Tensor token_indices =
        fixed_workspace_prefix(geometry_workspace->token_indices,
                               dsa.input_positions.numel(),
                               dsa.input_positions.device(),
                               torch::kLong);
    torch::arange_out(token_indices, dsa.input_positions.numel());
  }
  const torch::Tensor c4_position_storage =
      geometry_workspace == nullptr ? torch::Tensor()
                                    : geometry_workspace->c4_compact_positions;
  const torch::Tensor c128_position_storage =
      geometry_workspace == nullptr
          ? torch::Tensor()
          : geometry_workspace->c128_compact_positions;
  dsa.c4_pad_positions = build_device_compressed_positions(dsa.input_positions,
                                                           batch_size,
                                                           /*ratio=*/4,
                                                           c4_position_storage,
                                                           geometry_workspace);
  dsa.c128_pad_positions =
      build_device_compressed_positions(dsa.input_positions,
                                        batch_size,
                                        /*ratio=*/128,
                                        c128_position_storage,
                                        geometry_workspace);

  CHECK_LE(params.device_multi_block_tables.size(), group_infos.size());
  std::vector<torch::Tensor> group_block_tables;
  std::vector<torch::Tensor> group_slot_mappings;
  group_block_tables.reserve(params.device_multi_block_tables.size());
  group_slot_mappings.reserve(params.device_multi_block_tables.size());
  for (size_t group_id = 0; group_id < params.device_multi_block_tables.size();
       ++group_id) {
    const DSAGroupInfo& group_info = group_infos[group_id];
    torch::Tensor raw_block_table =
        normalize_group_table_rows(params.device_multi_block_tables[group_id],
                                   batch_size,
                                   static_cast<int64_t>(group_id),
                                   geometry_workspace);
    if (group_info.type == DSACacheType::SLIDING_WINDOW) {
      const torch::Tensor output_storage =
          geometry_workspace == nullptr ? torch::Tensor()
                                        : geometry_workspace->swa_block_table;
      group_block_tables.emplace_back(
          build_device_swa_block_table(raw_block_table,
                                       kv_seq_lens,
                                       group_info.block_size,
                                       dsa.max_seq_len,
                                       output_storage,
                                       geometry_workspace));
      group_slot_mappings.emplace_back(params.attention.device.new_cache_slots);
      continue;
    }
    if (group_info.type == DSACacheType::TOKEN) {
      torch::Tensor output_storage;
      const torch::Tensor* compressed_positions = nullptr;
      if (geometry_workspace != nullptr) {
        CHECK(group_info.ratio == 4 || group_info.ratio == 128)
            << "Prepared DeepSeek-V4 fixed DSA workspace supports only C4 "
               "and C128 token managers";
        output_storage = group_info.ratio == 4
                             ? geometry_workspace->c4_compact_slots
                             : geometry_workspace->c128_compact_slots;
        compressed_positions = group_info.ratio == 4 ? &dsa.c4_pad_positions
                                                     : &dsa.c128_pad_positions;
      }
      group_block_tables.emplace_back(raw_block_table);
      torch::Tensor token_slots =
          build_device_token_slots(dsa.input_positions,
                                   raw_block_table,
                                   group_info.ratio,
                                   group_info.block_size,
                                   output_storage,
                                   geometry_workspace);
      if (compressed_positions != nullptr) {
        CHECK(compressed_positions->defined());
        CHECK_EQ(token_slots.numel(), compressed_positions->numel())
            << "Prepared DSA compressed slot rows must match compressed "
               "position rows for ratio "
            << group_info.ratio;
      }
      group_slot_mappings.emplace_back(std::move(token_slots));
      continue;
    }
    group_block_tables.emplace_back(raw_block_table);
    group_slot_mappings.emplace_back(torch::Tensor());
  }

  const std::vector<std::vector<DSACacheInfo>>& caches_info = *dsa.caches_info;
  dsa.block_tables.resize(caches_info.size());
  dsa.slot_mappings.resize(caches_info.size());
  for (size_t layer_id = 0; layer_id < caches_info.size(); ++layer_id) {
    const std::vector<DSACacheInfo>& layer_caches = caches_info[layer_id];
    dsa.block_tables[layer_id].resize(layer_caches.size());
    dsa.slot_mappings[layer_id].resize(layer_caches.size());
    for (size_t cache_id = 0; cache_id < layer_caches.size(); ++cache_id) {
      const int32_t group_id = layer_caches[cache_id].group_id;
      CHECK_GE(group_id, 0);
      CHECK_LT(static_cast<size_t>(group_id), group_block_tables.size());
      dsa.block_tables[layer_id][cache_id] = group_block_tables[group_id];
      dsa.slot_mappings[layer_id][cache_id] = group_slot_mappings[group_id];
    }
  }
}

torch::Tensor DSAMetadataBuilder::expand_blocks_to_slots(
    const torch::Tensor& block_table,
    const DSAGroupInfo& gi,
    const std::vector<int32_t>& ctx_lens,
    int32_t batch_size,
    int64_t total_tokens) {
  const int32_t bs = gi.block_size;
  auto slots = torch::full({total_tokens}, -1, torch::kInt32);
  auto slots_acc = slots.accessor<int32_t, 1>();
  auto bt_acc = block_table.accessor<int32_t, 2>();
  const int32_t max_blocks = static_cast<int32_t>(block_table.size(1));

  int64_t start_idx = 0;
  for (int32_t seq = 0; seq < batch_size; ++seq) {
    int64_t token_len = ctx_lens[seq];
    int64_t slot_num = compute_slot_num(gi, token_len);
    if (seq >= block_table.size(0)) {
      // ACL graph capture pads sequence-len vectors to bucket size while the
      // request-shaped CPU multi_block_tables only contains real batch rows.
      // Treat missing padded rows as dummy rows filled with -1.
      start_idx += token_len;
      continue;
    }

    int64_t filled = 0;
    for (int32_t blk = 0; blk < max_blocks && filled < slot_num; ++blk) {
      int32_t block_id = bt_acc[seq][blk];
      if (block_id < 0) break;
      for (int32_t off = 0; off < bs && filled < slot_num; ++off) {
        slots_acc[start_idx + filled] =
            static_cast<int32_t>(static_cast<int64_t>(block_id) * bs + off);
        ++filled;
      }
    }
    start_idx += token_len;
  }
  return slots;
}

int64_t DSAMetadataBuilder::compute_slot_num(const DSAGroupInfo& gi,
                                             int64_t token_len) {
  if (gi.type == DSACacheType::TOKEN) {
    return token_len / gi.ratio;
  }
  // SLIDING_WINDOW
  const int32_t bs = gi.block_size;
  if (token_len > bs) {
    return token_len % bs + bs;
  }
  int64_t n = token_len % bs;
  return (n == 0 && token_len > 0) ? bs : n;
}

void DSAMetadataBuilder::process_group(
    const torch::Tensor& raw_bt,
    const DSAGroupInfo& gi,
    const std::vector<int32_t>& ctx_lens,
    const std::vector<int32_t>& q_lens,
    const std::vector<int32_t>& new_cache_slots,
    int32_t batch_size,
    int64_t total_tokens,
    int64_t graph_slot_capacity,
    int32_t block_table_capacity_cols,
    torch::Tensor& out_bt,
    torch::Tensor& out_slots) {
  if (gi.type == DSACacheType::TOKEN) {
    process_token_group(raw_bt,
                        gi.ratio,
                        gi.block_size,
                        ctx_lens,
                        q_lens,
                        batch_size,
                        total_tokens,
                        graph_slot_capacity,
                        block_table_capacity_cols,
                        out_bt,
                        out_slots);
  } else if (gi.type == DSACacheType::SLIDING_WINDOW) {
    process_swa_group(raw_bt,
                      gi.block_size,
                      ctx_lens,
                      q_lens,
                      new_cache_slots,
                      batch_size,
                      graph_slot_capacity,
                      block_table_capacity_cols,
                      out_bt,
                      out_slots);
  } else {
    auto raw_slots =
        expand_blocks_to_slots(raw_bt, gi, ctx_lens, batch_size, total_tokens);
    out_slots =
        torch::where(raw_slots.eq(-1), torch::zeros_like(raw_slots), raw_slots);
    out_bt = raw_bt;
  }
}

void DSAMetadataBuilder::process_token_group(
    const torch::Tensor& raw_bt,
    int32_t ratio,
    int32_t block_size,
    const std::vector<int32_t>& ctx_lens,
    const std::vector<int32_t>& q_lens,
    int32_t batch_size,
    int64_t total_tokens,
    int64_t graph_slot_capacity,
    int32_t block_table_capacity_cols,
    torch::Tensor& out_bt,
    torch::Tensor& out_slots) {
  CHECK_EQ(static_cast<int32_t>(ctx_lens.size()), batch_size)
      << "process_token_group requires ctx_lens.size == batch_size, got "
      << ctx_lens.size() << " vs " << batch_size;
  CHECK_EQ(static_cast<int32_t>(q_lens.size()), batch_size)
      << "process_token_group requires q_lens.size == batch_size, got "
      << q_lens.size() << " vs " << batch_size;
  CHECK_GT(ratio, 0) << "process_token_group requires ratio > 0, got " << ratio;
  CHECK_GT(block_size, 0) << "process_token_group requires block_size > 0, got "
                          << block_size;
  CHECK_EQ(raw_bt.dim(), 2)
      << "process_token_group requires raw_bt dim == 2, got " << raw_bt.dim();

  int64_t query_total_tokens = 0;
  for (const int q_len : q_lens) {
    query_total_tokens += static_cast<int64_t>(q_len);
  }

  // Token caches commit one row only when a sequence crosses a compression
  // boundary. Count rows per sequence so multi-batch padding/dummy rows do not
  // become cache writes.
  int64_t committed_rows = 0;
  for (int32_t seq = 0; seq < batch_size; ++seq) {
    const int64_t ctx_len = static_cast<int64_t>(ctx_lens[seq]);
    const int64_t q_len =
        std::clamp<int64_t>(static_cast<int64_t>(q_lens[seq]), 0, ctx_len);
    const int64_t prev_ctx_len = ctx_len - q_len;
    committed_rows += ctx_len / ratio - prev_ctx_len / ratio;
  }

  // Token caches write only the compressed rows produced by the current
  // forward step. Padded RoPE/compressor rows must not become cache writes.
  const int64_t out_slot_rows =
      graph_slot_capacity > 0
          ? std::max<int64_t>(graph_slot_capacity, committed_rows)
          : committed_rows;
  auto out_slots_tensor = torch::full({out_slot_rows}, -1, raw_bt.options());
  auto out_slots_acc = out_slots_tensor.accessor<int32_t, 1>();
  auto raw_bt_acc = raw_bt.accessor<int32_t, 2>();
  const int64_t semantic_cols = raw_bt.size(1);
  const int64_t block_size_i64 = static_cast<int64_t>(block_size);

  auto slot_for_compressed_index = [&](int32_t seq,
                                       int64_t compressed_idx) -> int32_t {
    if (seq >= raw_bt.size(0) || semantic_cols <= 0) {
      return -1;
    }
    const int64_t block_idx = compressed_idx / block_size_i64;
    if (block_idx >= semantic_cols) {
      return -1;
    }
    const int32_t block_id = raw_bt_acc[seq][block_idx];
    if (block_id < 0) {
      return -1;
    }
    const int64_t block_offset = compressed_idx % block_size_i64;
    return static_cast<int32_t>(
        static_cast<int64_t>(block_id) * block_size_i64 + block_offset);
  };

  int64_t write_idx = 0;
  for (int32_t seq = 0; seq < batch_size; ++seq) {
    const int64_t ctx_len = static_cast<int64_t>(ctx_lens[seq]);
    const int64_t q_len =
        std::clamp<int64_t>(static_cast<int64_t>(q_lens[seq]), 0, ctx_len);
    const int64_t prev_ctx_len = ctx_len - q_len;
    const int64_t prev_committed = prev_ctx_len / ratio;
    const int64_t committed = ctx_len / ratio;
    const int64_t new_committed = committed - prev_committed;
    for (int64_t i = 0; i < new_committed; ++i) {
      out_slots_acc[write_idx++] =
          slot_for_compressed_index(seq, prev_committed + i);
    }
  }

  CHECK_EQ(write_idx, committed_rows)
      << "process_token_group committed slot count mismatch: write_idx="
      << write_idx << ", committed_rows=" << committed_rows
      << ", query_total_tokens=" << query_total_tokens << ", ratio=" << ratio
      << ", batch_size=" << batch_size << ", total_tokens=" << total_tokens;
  out_slots = out_slots_tensor;
  out_bt = graph_slot_capacity > 0
               ? pad_block_table(
                     raw_bt,
                     batch_size,
                     std::max<int32_t>(block_table_capacity_cols,
                                       static_cast<int32_t>(raw_bt.size(1))),
                     /*pad_value=*/-1)
               : raw_bt;
}

void DSAMetadataBuilder::process_swa_group(
    const torch::Tensor& raw_bt,
    int32_t block_size,
    const std::vector<int32_t>& ctx_lens,
    const std::vector<int32_t>& q_lens,
    const std::vector<int32_t>& new_cache_slots,
    int32_t batch_size,
    int64_t graph_slot_capacity,
    int32_t block_table_capacity_cols,
    torch::Tensor& out_bt,
    torch::Tensor& out_slots) {
  CHECK_EQ(static_cast<int32_t>(ctx_lens.size()), batch_size)
      << "process_swa_group requires ctx_lens.size == batch_size, got "
      << ctx_lens.size() << " vs " << batch_size;
  CHECK_EQ(static_cast<int32_t>(q_lens.size()), batch_size)
      << "process_swa_group requires q_lens.size == batch_size, got "
      << q_lens.size() << " vs " << batch_size;
  CHECK_GT(block_size, 0) << "process_swa_group requires block_size > 0, got "
                          << block_size;
  CHECK_EQ(raw_bt.dim(), 2)
      << "process_swa_group requires raw_bt dim == 2, got " << raw_bt.dim();

  int64_t query_total_tokens = 0;
  for (int32_t seq = 0; seq < batch_size; ++seq) {
    query_total_tokens += std::clamp<int64_t>(
        static_cast<int64_t>(q_lens[seq]), 0, ctx_lens[seq]);
  }

  // SWA cache writes only the tokens produced by the current forward step.
  // Do not reuse the full-context raw_slots prefix here: decode has one kv row,
  // and prefix slot 0 would incorrectly overwrite cache row 0.
  const int64_t out_slot_rows =
      graph_slot_capacity > 0
          ? std::max<int64_t>(graph_slot_capacity, query_total_tokens)
          : query_total_tokens;
  auto out_slots_tensor = torch::full({out_slot_rows}, -1, raw_bt.options());
  auto out_slots_acc = out_slots_tensor.accessor<int32_t, 1>();
  auto raw_bt_acc = raw_bt.accessor<int32_t, 2>();
  const int64_t semantic_cols = raw_bt.size(1);
  const int64_t storage_cols =
      graph_slot_capacity > 0 && block_table_capacity_cols > 0
          ? std::max<int64_t>(block_table_capacity_cols, raw_bt.size(1))
          : raw_bt.size(1);
  const int64_t block_size_i64 = static_cast<int64_t>(block_size);

  auto slot_for_position = [&](int32_t seq, int64_t pos) -> int32_t {
    if (semantic_cols <= 0) {
      return -1;
    }
    const int64_t block_idx = (pos / block_size_i64) % semantic_cols;
    const int32_t block_id = raw_bt_acc[seq][block_idx];
    if (block_id < 0) {
      return -1;
    }
    const int64_t block_offset = pos % block_size_i64;
    return static_cast<int32_t>(
        static_cast<int64_t>(block_id) * block_size_i64 + block_offset);
  };

  int64_t write_idx = 0;
  if (static_cast<int64_t>(new_cache_slots.size()) == query_total_tokens) {
    // Expanded block-parallel rows all report q_len=1 and the same kv_len, so
    // q_start cannot recover their distinct positions. The row builder already
    // resolved those positions through the SWA ring; preserve those slots.
    std::memcpy(out_slots_acc.data(),
                new_cache_slots.data(),
                query_total_tokens * sizeof(int32_t));
    write_idx = query_total_tokens;
  } else {
    for (int32_t seq = 0; seq < batch_size; ++seq) {
      const int64_t ctx_len = static_cast<int64_t>(ctx_lens[seq]);
      const int64_t q_len =
          std::clamp<int64_t>(static_cast<int64_t>(q_lens[seq]), 0, ctx_len);
      if (seq >= raw_bt.size(0)) {
        write_idx += q_len;
        continue;
      }
      const int64_t q_start = ctx_len - q_len;
      for (int64_t i = 0; i < q_len; ++i) {
        out_slots_acc[write_idx++] = slot_for_position(seq, q_start + i);
      }
    }
  }
  CHECK_EQ(write_idx, query_total_tokens)
      << "process_swa_group slot count mismatch";

  out_slots = out_slots_tensor;

  const int32_t current_cols = static_cast<int32_t>(semantic_cols);
  int32_t max_dst_len = 0;
  std::vector<int32_t> dst_lens(batch_size);
  for (int32_t s = 0; s < batch_size; ++s) {
    const int64_t ctx_len = std::max<int64_t>(ctx_lens[s], 0);
    dst_lens[s] = static_cast<int32_t>((ctx_len + block_size - 1) / block_size);
    max_dst_len = std::max(max_dst_len, dst_lens[s]);
  }
  max_dst_len = std::max(max_dst_len, current_cols);
  if (graph_slot_capacity > 0) {
    max_dst_len =
        std::max<int32_t>(max_dst_len, static_cast<int32_t>(storage_cols));
  }

  auto new_bt = torch::full({batch_size, max_dst_len}, -1, raw_bt.options());
  auto new_acc = new_bt.accessor<int32_t, 2>();
  auto old_acc = raw_bt.accessor<int32_t, 2>();

  for (int32_t s = 0; s < batch_size; ++s) {
    if (s >= raw_bt.size(0)) {
      continue;
    }
    const int32_t retained_cols = std::min(current_cols, dst_lens[s]);
    const int32_t start_col = dst_lens[s] - retained_cols;
    for (int32_t j = 0; j < retained_cols; ++j) {
      const int32_t logical_col = start_col + j;
      // Keep the read-side block table aligned with slot_for_position().
      const int32_t physical_col = logical_col % current_cols;
      new_acc[s][logical_col] = old_acc[s][physical_col];
    }
  }
  out_bt = new_bt;
}

void DSAMetadataBuilder::build_seq_lengths(const ModelInputParams& params,
                                           const torch::Device& target_device,
                                           int32_t batch_size,
                                           DSAMetadata& dsa_metadata) {
  auto int_options =
      torch::TensorOptions().dtype(torch::kInt32).device(target_device);
  torch::Tensor kv_lens = params.attention.device.kv_seq_lens;
  if (target_device.is_cpu() &&
      static_cast<int32_t>(params.attention.host.kv_seq_lens.size()) ==
          batch_size) {
    kv_lens = torch::tensor(
        std::vector<int32_t>(params.attention.host.kv_seq_lens.begin(),
                             params.attention.host.kv_seq_lens.end()),
        int_options);
  } else if (!kv_lens.defined() || kv_lens.numel() == 0) {
    kv_lens = torch::tensor(
        std::vector<int32_t>(params.attention.host.kv_seq_lens.begin(),
                             params.attention.host.kv_seq_lens.end()),
        int_options);
  } else if (kv_lens.device() != target_device) {
    kv_lens = safe_to(kv_lens, int_options, true);
  }
  dsa_metadata.seq_lens = kv_lens;
  dsa_metadata.actual_seq_lengths_kv = kv_lens;

  torch::Tensor q_lens;
  q_lens = params.attention.device.q_seq_lens;
  if (static_cast<int32_t>(params.attention.host.q_seq_lens.size()) ==
      batch_size) {
    // Prefer explicit per-sequence query lengths from ModelInputParams.
    // This is accurate for prefill/decode/chunked/mixed batches.
    if (target_device.is_cpu()) {
      q_lens = torch::tensor(
          std::vector<int32_t>(params.attention.host.q_seq_lens.begin(),
                               params.attention.host.q_seq_lens.end()),
          int_options);
    } else if (!q_lens.defined() || q_lens.numel() == 0) {
      q_lens = torch::tensor(
          std::vector<int32_t>(params.attention.host.q_seq_lens.begin(),
                               params.attention.host.q_seq_lens.end()),
          int_options);
    } else if (q_lens.device() != target_device) {
      q_lens = safe_to(q_lens, int_options, true);
    }
  } else if (params.meta.batch_forward_type.no_decode()) {
    // Pure prefill path fallback: query lengths follow KV context lengths.
    q_lens = kv_lens;
  } else {
    // Decode fallback: each sequence contributes one query token.
    q_lens = torch::ones({batch_size}, int_options);
  }
  // cumsum with leading 0: shape (batch_size+1,)
  auto cumsum = torch::cumsum(q_lens, /*dim=*/0, /*dtype=*/torch::kInt32);
  dsa_metadata.actual_seq_lengths_query =
      torch::cat({torch::zeros({1}, int_options), cumsum});
  dsa_metadata.seq_lens_q = q_lens;

  // Precompute the kv cumulative sequence lengths once per forward so the
  // per-layer indexer metadata builder can reuse it instead of running a
  // host-side cumsum on every DSA layer (kv_lens is identical across layers
  // within one forward).
  if (kv_lens.numel() > 0) {
    torch::Tensor kv_lens_i32 = kv_lens.to(torch::kInt32);
    torch::Tensor kv_cumsum =
        torch::cumsum(kv_lens_i32, /*dim=*/0, /*dtype=*/torch::kInt32);
    dsa_metadata.kv_cu_seq_lens =
        torch::cat({torch::zeros({1}, int_options), kv_cumsum});
  }

  if (kv_lens.numel() > 0) {
    dsa_metadata.max_seqlen_kv = torch::max(kv_lens).to(torch::kInt32);
  } else {
    dsa_metadata.max_seqlen_kv = torch::zeros({1}, int_options);
  }

  if (q_lens.numel() > 0) {
    dsa_metadata.max_seqlen_q = torch::max(q_lens).to(torch::kInt32);
  } else {
    dsa_metadata.max_seqlen_q = torch::zeros({1}, int_options);
  }

  dsa_metadata.max_query_len =
      std::max<int64_t>(params.meta.q_max_seq_len,
                        vector_max_or_zero(params.attention.host.q_seq_lens));
  dsa_metadata.max_seq_len =
      std::max<int64_t>(params.meta.kv_max_seq_len,
                        vector_max_or_zero(params.attention.host.kv_seq_lens));
}

void DSAMetadataBuilder::build_positions(const ModelInputParams& params,
                                         int32_t batch_size,
                                         DSAMetadata& dsa_metadata) {
  if (!dsa_metadata.input_positions.defined()) return;

  auto input_positions = dsa_metadata.input_positions;
  int64_t num_tokens = input_positions.size(0);
  const bool is_acl_graph = params.enable_graph;
  const auto target_device = input_positions.device();
  const auto pos_dtype = input_positions.scalar_type();
  auto cpu_options =
      torch::TensorOptions().dtype(pos_dtype).device(torch::kCPU);

  const bool has_host_lens =
      static_cast<int32_t>(params.attention.host.kv_seq_lens.size()) ==
      batch_size;
  if (has_host_lens) {
    std::vector<int64_t> c4_positions;
    std::vector<int64_t> c128_positions;
    c4_positions.reserve(
        static_cast<size_t>(std::max<int64_t>(num_tokens / 4 + batch_size, 0)));
    c128_positions.reserve(static_cast<size_t>(
        std::max<int64_t>(num_tokens / 128 + batch_size, 0)));
    for (int32_t seq = 0; seq < batch_size; ++seq) {
      const int64_t kv_len = params.attention.host.kv_seq_lens[seq];
      int64_t q_len = 1;
      if (static_cast<int32_t>(params.attention.host.q_seq_lens.size()) ==
          batch_size) {
        q_len = params.attention.host.q_seq_lens[seq];
      } else if (params.meta.batch_forward_type.no_decode()) {
        q_len = kv_len;
      }
      q_len = std::clamp<int64_t>(q_len, 0, kv_len);
      const int64_t start_pos = kv_len - q_len;
      for (int64_t i = 0; i < q_len; ++i) {
        const int64_t pos = start_pos + i;
        const int64_t next_pos = pos + 1;
        if (next_pos % 4 == 0) {
          c4_positions.push_back(next_pos - 4);
        }
        if (next_pos % 128 == 0) {
          c128_positions.push_back(next_pos - 128);
        }
      }
    }

    const int64_t c4_target =
        is_acl_graph
            ? num_tokens
            : std::min<int64_t>(num_tokens, num_tokens / 4 + batch_size);
    const int64_t c128_target =
        is_acl_graph
            ? num_tokens
            : std::min<int64_t>(num_tokens, num_tokens / 128 + batch_size);
    c4_positions.resize(static_cast<size_t>(std::max<int64_t>(c4_target, 0)),
                        0);
    c128_positions.resize(
        static_cast<size_t>(std::max<int64_t>(c128_target, 0)), 0);

    dsa_metadata.c4_pad_positions = torch::tensor(c4_positions, cpu_options);
    dsa_metadata.c128_pad_positions =
        torch::tensor(c128_positions, cpu_options);
    return;
  }

  std::vector<int64_t> host_positions;
  host_positions.reserve(static_cast<size_t>(std::max<int64_t>(num_tokens, 0)));
  if (static_cast<int64_t>(host_positions.size()) < num_tokens &&
      input_positions.device().is_cpu()) {
    auto positions_cpu = input_positions.contiguous();
    if (positions_cpu.scalar_type() == torch::kInt64) {
      auto acc = positions_cpu.accessor<int64_t, 1>();
      for (int64_t i = static_cast<int64_t>(host_positions.size());
           i < num_tokens;
           ++i) {
        host_positions.push_back(acc[i]);
      }
    } else {
      auto positions_i64 = positions_cpu.to(torch::kInt64);
      auto acc = positions_i64.accessor<int64_t, 1>();
      for (int64_t i = static_cast<int64_t>(host_positions.size());
           i < num_tokens;
           ++i) {
        host_positions.push_back(acc[i]);
      }
    }
  }

  if (static_cast<int64_t>(host_positions.size()) > num_tokens) {
    host_positions.resize(static_cast<size_t>(num_tokens));
  }
  if (static_cast<int64_t>(host_positions.size()) < num_tokens) {
    host_positions.resize(static_cast<size_t>(num_tokens), 0);
  }

  auto build_compressed_positions = [&](int64_t ratio) {
    std::vector<int64_t> compressed;
    compressed.reserve(host_positions.size() / ratio + batch_size);
    for (const int64_t pos : host_positions) {
      if ((pos + 1) % ratio == 0) {
        compressed.push_back((pos + 1) - ratio);
      }
    }
    const int64_t target =
        is_acl_graph
            ? num_tokens
            : std::min<int64_t>(num_tokens, num_tokens / ratio + batch_size);
    compressed.resize(static_cast<size_t>(std::max<int64_t>(target, 0)), 0);
    auto tensor = torch::tensor(compressed, cpu_options);
    return tensor;
  };

  dsa_metadata.c4_pad_positions = build_compressed_positions(4);
  dsa_metadata.c128_pad_positions = build_compressed_positions(128);
}

}  // namespace xllm::layer
