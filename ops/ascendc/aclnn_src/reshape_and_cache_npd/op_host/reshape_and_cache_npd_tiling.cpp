/**
 * Copyright 2025 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "reshape_and_cache_npd/op_host/reshape_and_cache_npd_tiling.h"
#include <string>
#include <vector>
#include "register/op_def_registry.h"
#include "reshape_and_cache_npd/op_host/reshape_and_cache_npd_comm.h"
#include "graph/utils/type_utils.h"
#include "tiling/platform/platform_ascendc.h"
#include "utils/log/asc_cpu_log.h"

namespace optiling {
int32_t ReshapeAndCacheNpdTilingId(ge::DataType dtype, int32_t cacheConfig, int32_t kvConfig) {
  int32_t tiling_id = 0;
  if (cacheConfig) tiling_id |= 1;
  if (kvConfig) tiling_id |= 1 << 1;
  if (dtype == ge::DataType::DT_INT8) tiling_id |= 1 << 2;
  return tiling_id;
}

static ge::graphStatus ReshapeAndCacheNpdTilingGetAttr(gert::TilingContext *context, int *kv_cache_mode, int *kv_mode) {
  *kv_cache_mode = std::string(context->GetAttrs()->GetAttrPointer<char>(0)) == std::string(BNSD_LAYOUT);
  *kv_mode = std::string(context->GetAttrs()->GetAttrPointer<char>(1)) == std::string(NPD_LAYOUT);

  return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetSeqLen(gert::TilingContext *context, int batch, size_t input_idx,
                                 std::vector<int64_t> *seq_len) {
  auto t = context->GetInputTensor(input_idx);
  if (t == nullptr) {
    ASC_CPU_LOG_ERROR("seq length is null");
    return ge::GRAPH_FAILED;
  }
  if (t->GetShapeSize() != batch) {
    ASC_CPU_LOG_ERROR("seq length size is illegal, size = %d batch size = %d", t->GetShapeSize(), batch);
    return ge::GRAPH_FAILED;
  }
  auto p = t->GetData<int64_t>();
  seq_len->assign(p, p + batch);
  return ge::GRAPH_SUCCESS;
}

static ge::graphStatus ReshapeAndCacheNpdValidateInput(gert::TilingContext *context) {
  size_t k_idx = static_cast<size_t>(ReshapeAndCacheNpdInputIndex::kInputKeyIndex);
  size_t k_cache_idx = static_cast<size_t>(ReshapeAndCacheNpdInputIndex::kInputKeyCacheIndex);
  size_t v_idx = static_cast<size_t>(ReshapeAndCacheNpdInputIndex::kInputValueIndex);
  size_t v_cache_idx = static_cast<size_t>(ReshapeAndCacheNpdInputIndex::kInputValueCacheIndex);
  size_t sm_idx = static_cast<size_t>(ReshapeAndCacheNpdInputIndex::kInputSlotMappingIndex);

  auto k_data_data_type = context->GetInputDesc(k_idx)->GetDataType();
  auto k_cache_data_type = context->GetInputDesc(k_cache_idx)->GetDataType();
  auto sm_data_type = context->GetInputDesc(sm_idx)->GetDataType();
  auto value_cache_shape = context->GetInputShape(v_cache_idx)->GetOriginShape();
  auto slot_mapping_shape = context->GetInputShape(sm_idx)->GetOriginShape();

  if (k_data_data_type == ge::DataType::DT_UNDEFINED || k_cache_data_type == ge::DataType::DT_UNDEFINED ||
      sm_data_type == ge::DataType::DT_UNDEFINED) {
    ASC_CPU_LOG_ERROR(
      "Input 0, 2, 4(key, key_cache, slot_mapping) are required. Their dtypes cannot be None, but got %d %d %d",
      k_data_data_type, k_cache_data_type, sm_data_type);
    return ge::GRAPH_FAILED;
  }
  auto v_data_data_type = context->GetInputDesc(v_idx)->GetDataType();
  auto v_cache_data_type = context->GetInputDesc(v_cache_idx)->GetDataType();
  if ((v_data_data_type == ge::DataType::DT_UNDEFINED) ^ (v_cache_data_type == ge::DataType::DT_UNDEFINED)) {
    ASC_CPU_LOG_ERROR(
      "Input 1, 3 (value, value_cache) should either both be None or have the same dtype, but got  %d %d",
      v_data_data_type, v_cache_data_type);
    return ge::GRAPH_FAILED;
  }
  auto key_shape = context->GetInputShape(k_idx)->GetOriginShape();
  auto key_cache_shape = context->GetInputShape(k_cache_idx)->GetOriginShape();
  if (key_shape.GetDimNum() != kDim3 && key_shape.GetDimNum() != kDim2) {
    ASC_CPU_LOG_ERROR("The dim of key should be = %d or %d, but got %d", kDim2, kDim3, key_shape.GetDimNum());
    return ge::GRAPH_FAILED;
  }

  if (slot_mapping_shape.GetDimNum() != kDim1) {
    ASC_CPU_LOG_ERROR("The dim of slot mapping should be = %d, but got %d", kDim1, slot_mapping_shape.GetDimNum());
    return ge::GRAPH_FAILED;
  }

  if (key_cache_shape.GetDimNum() != kDim4) {
    ASC_CPU_LOG_ERROR("The dim of key cache should be = %d, but got %d", kDim4, key_cache_shape.GetDimNum());
    return ge::GRAPH_FAILED;
  }

  if (value_cache_shape.GetDimNum() != kDim4) {
    ASC_CPU_LOG_ERROR("The dim of value cache should be = %d, but got %d", kDim4, value_cache_shape.GetDimNum());
    return ge::GRAPH_FAILED;
  }
  return ge::GRAPH_SUCCESS;
}

static ge::graphStatus ReshapeAndCacheNpdTiling(gert::TilingContext *context) {
  size_t k_idx = static_cast<size_t>(ReshapeAndCacheNpdInputIndex::kInputKeyIndex);
  size_t k_cache_idx = static_cast<size_t>(ReshapeAndCacheNpdInputIndex::kInputKeyCacheIndex);
  size_t block_tbl_idx = static_cast<size_t>(ReshapeAndCacheNpdInputIndex::kInputBlockTblIndex);
  size_t q_seq_len_idx = static_cast<size_t>(ReshapeAndCacheNpdInputIndex::kInputQSeqIndex);
  size_t kv_seq_len_idx = static_cast<size_t>(ReshapeAndCacheNpdInputIndex::kInputKVSeqIndex);

  int32_t cache_npd, kv_npd;
  ReshapeAndCacheNpdTilingGetAttr(context, &cache_npd, &kv_npd);
  if (ReshapeAndCacheNpdValidateInput(context) != ge::GRAPH_SUCCESS) {
    ASC_CPU_LOG_ERROR("failed to validate reshape and cache npd input");
    return ge::GRAPH_FAILED;
  }

  auto key_shape = context->GetInputShape(k_idx)->GetOriginShape();
  auto key_cache_shape = context->GetInputShape(k_cache_idx)->GetOriginShape();
  auto block_tbl_shape = context->GetInputShape(block_tbl_idx)->GetOriginShape();

  // TH is the default layout
  uint32_t num_tokens = key_shape.GetDim(0);
  uint32_t hidden_size = key_shape.GetDim(kDim1);
  if (key_shape.GetDimNum() == kRank3) {  // BSH
    hidden_size = key_shape.GetDim(kDim2);
    num_tokens *= key_shape.GetDim(kDim1);
  }

  auto num_tokens_max = key_cache_shape.GetDim(0) * key_cache_shape.GetDim(kDim1);
  if (cache_npd) {
    // [BlockNum, N, BlockSize, D]
    num_tokens_max = key_cache_shape.GetDim(0) * key_cache_shape.GetDim(kDim2);
  }
  if (num_tokens_max != 0 && num_tokens_max < num_tokens) {
    ASC_CPU_LOG_ERROR("Number of tokens(%d) exceed max cacpcity(%d)", num_tokens, num_tokens_max);
    return ge::GRAPH_FAILED;
  }
  uint32_t key_head_num = key_cache_shape.GetDim(kDim1);
  uint32_t page_size = key_cache_shape.GetDim(kDim2);
  if (!cache_npd) {
    key_head_num = key_cache_shape.GetDim(kDim2);
    page_size = key_cache_shape.GetDim(kDim1);
  }
  uint32_t value_head_num = key_head_num;
  uint32_t batch_size = block_tbl_shape.GetDim(0);
  uint32_t batch_tbl_dim1 = block_tbl_shape.GetDim(kDim1);
  if (batch_size > KMaxBatch) {
    ASC_CPU_LOG_ERROR("Batch size too big");
    return ge::GRAPH_FAILED;
  }
  std::vector<int64_t> q_seq_len, kv_seq_len;
  if (GetSeqLen(context, batch_size, q_seq_len_idx, &q_seq_len) != ge::GRAPH_SUCCESS ||
      GetSeqLen(context, batch_size, kv_seq_len_idx, &kv_seq_len) != ge::GRAPH_SUCCESS) {
    ASC_CPU_LOG_ERROR("Failed to get sequence lengths");
    return ge::GRAPH_FAILED;
  }

  auto ascendc_platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
  auto core_num = ascendc_platform.GetCoreNumAiv();
  size_t ub_size = 0;
  ascendc_platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);
  auto p1_core = core_num;
  if (num_tokens < core_num) {
    p1_core = num_tokens;
  }
  size_t ws_size = batch_size * kDim2;
  auto tiling_key = ReshapeAndCacheNpdTilingId(context->GetInputDesc(0)->GetDataType(), cache_npd, kv_npd);
  ReshapeAndCacheNpdTilingData tiling;
  auto tiling_size = (KLastIdx + KSeqLenArrayNum * batch_size) * sizeof(uint32_t);
  auto tiling_buf = tiling.get_buf();
  tiling_buf[KTilingId] = tiling_key;
  tiling_buf[KP1UseCoreNumIdx] = p1_core;
  tiling_buf[KNumTokensIdx] = num_tokens;
  tiling_buf[KHiddenSizeIdx] = hidden_size;
  tiling_buf[KKeyHeadNumIdx] = key_head_num;
  tiling_buf[KVHeadNumIdx] = value_head_num;
  tiling_buf[KPageSizeIdx] = page_size;
  tiling_buf[KBatchSizeIdx] = batch_size;
  tiling_buf[kUbSizeIdx] = ub_size;
  tiling_buf[kWorkspaceSizeIdx] = ws_size;
  tiling_buf[kBlockTblDim1Idx] = batch_tbl_dim1;
  for (uint32_t i = 0; i < batch_size; i++) {
    tiling_buf[KLastIdx + i] = static_cast<uint32_t>(kv_seq_len.at(i));
    tiling_buf[KLastIdx + i + batch_size] = static_cast<uint32_t>(q_seq_len.at(i));
  }
  tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
  context->GetRawTilingData()->SetDataSize(tiling_size);
  size_t *currentWorkspace = context->GetWorkspaceSizes(kDim1);
  currentWorkspace[0] = ws_size * sizeof(int) * core_num;
  context->SetTilingKey(tiling_key);
  context->SetBlockDim(core_num);
  return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(ReshapeAndCacheNpd).Tiling(ReshapeAndCacheNpdTiling);
}  // namespace optiling
