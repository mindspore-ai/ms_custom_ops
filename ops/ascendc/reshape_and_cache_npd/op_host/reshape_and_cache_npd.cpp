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
#include <string>
#include <vector>
#include "reshape_and_cache_npd.h"
#include "reshape_and_cache_npd_tiling.h"
#include "register/op_def_registry.h"
#include "graph/utils/type_utils.h"
#include "tiling/platform/platform_ascendc.h"
#include "utils/log/asc_cpu_log.h"

namespace optiling {
static constexpr auto kRank2 = 2;
static constexpr auto kRank3 = 3;
static constexpr auto kDim1 = 1;
static constexpr auto kDim2 = 2;
static constexpr auto kDim3 = 3;
static constexpr auto kDim4 = 4;
static constexpr auto kDim7 = 7;

int32_t ReshapeAndCacheNpdTilingId(ge::DataType dtype, int32_t cacheConfig, int32_t KvConfig) {
  int32_t tiling_id = 0;
  if (cacheConfig) tiling_id |= 1;
  if (KvConfig) tiling_id |= 1 << 1;
  return tiling_id;
}

static ge::graphStatus ReshapeAndCacheNpdTiling(gert::TilingContext *context, int *kv_cache_mode, int *kv_mode) {
  *kv_cache_mode = std::string(context->GetAttrs()->GetAttrPointer<char>(0)) == std::string(NPD_LAYOUT);
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

// Helper: validate inputs and extract shapes/parameters
static ge::graphStatus ValidateAndExtractNpdParams(gert::TilingContext *context, int32_t &cache_npd, int32_t &kv_npd,
                                                   uint32_t &num_tokens, uint32_t &hidden_size, uint32_t &key_head_num,
                                                   uint32_t &value_head_num, uint32_t &page_size, uint32_t &batch_size,
                                                   uint32_t &core_num, std::vector<int64_t> &q_seq_len,
                                                   std::vector<int64_t> &kv_seq_len) {
  size_t k_idx = static_cast<size_t>(ms_custom_ops::ReshapeAndCacheNpdInputIndex::kInputKeyIndex);
  size_t k_cache_idx = static_cast<size_t>(ms_custom_ops::ReshapeAndCacheNpdInputIndex::kInputKeyCacheIndex);
  size_t sm_idx = static_cast<size_t>(ms_custom_ops::ReshapeAndCacheNpdInputIndex::kInputSlotMappingIndex);
  size_t v_idx = static_cast<size_t>(ms_custom_ops::ReshapeAndCacheNpdInputIndex::kInputValueIndex);
  size_t v_cache_idx = static_cast<size_t>(ms_custom_ops::ReshapeAndCacheNpdInputIndex::kInputValueCacheIndex);
  size_t block_tbl_idx = static_cast<size_t>(ms_custom_ops::ReshapeAndCacheNpdInputIndex::kInputBlockTblIndex);
  size_t q_seq_len_idx = static_cast<size_t>(ms_custom_ops::ReshapeAndCacheNpdInputIndex::kInputQSeqIndex);
  size_t kv_seq_len_idx = static_cast<size_t>(ms_custom_ops::ReshapeAndCacheNpdInputIndex::kInputKVSeqIndex);

  ReshapeAndCacheNpdTiling(context, &cache_npd, &kv_npd);

  // dtype validation
  auto k_data_data_type = context->GetInputDesc(k_idx)->GetDataType();
  auto k_cache_data_type = context->GetInputDesc(k_cache_idx)->GetDataType();
  auto sm_data_type = context->GetInputDesc(sm_idx)->GetDataType();
  if (k_data_data_type == ge::DataType::DT_UNDEFINED || k_cache_data_type == ge::DataType::DT_UNDEFINED ||
      sm_data_type == ge::DataType::DT_UNDEFINED) {
    ASC_CPU_LOG_ERROR("Input key/key_cache/slot_mapping dtypes cannot be None");
    return ge::GRAPH_FAILED;
  }

  auto v_data_data_type = context->GetInputDesc(v_idx)->GetDataType();
  auto v_cache_data_type = context->GetInputDesc(v_cache_idx)->GetDataType();
  if ((v_data_data_type == ge::DataType::DT_UNDEFINED) ^ (v_cache_data_type == ge::DataType::DT_UNDEFINED)) {
    ASC_CPU_LOG_ERROR("value/value_cache must both be None or same dtype");
    return ge::GRAPH_FAILED;
  }

  // platform info
  auto ascendc_platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
  uint64_t ub_size;
  ascendc_platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);
  core_num = ascendc_platform.GetCoreNumAiv();

  // shape validation
  auto key_shape = context->GetInputShape(k_idx)->GetOriginShape();
  auto key_cache_shape = context->GetInputShape(k_cache_idx)->GetOriginShape();
  auto value_cache_shape = context->GetInputShape(v_cache_idx)->GetOriginShape();
  auto slot_mapping_shape = context->GetInputShape(sm_idx)->GetOriginShape();
  auto block_tbl_shape = context->GetInputShape(block_tbl_idx)->GetOriginShape();

  if (key_shape.GetDimNum() != kDim3 && key_shape.GetDimNum() != kDim2) {
    ASC_CPU_LOG_ERROR("Key dim must be 2 or 3");
    return ge::GRAPH_FAILED;
  }
  if (slot_mapping_shape.GetDimNum() != kDim1) {
    ASC_CPU_LOG_ERROR("Slot mapping dim must be 1");
    return ge::GRAPH_FAILED;
  }
  if (key_cache_shape.GetDimNum() != kDim4 || value_cache_shape.GetDimNum() != kDim4) {
    ASC_CPU_LOG_ERROR("Key/value cache dim must be 4");
    return ge::GRAPH_FAILED;
  }

  // token/hidden size
  num_tokens = key_shape.GetDim(0);
  hidden_size = key_shape.GetDim(kDim1);
  if (key_shape.GetDimNum() == kRank3) {  // BSH
    hidden_size = key_shape.GetDim(kDim2);
    num_tokens *= key_shape.GetDim(kDim1);
  }

  auto num_tokens_max = key_cache_shape.GetDim(0) * key_cache_shape.GetDim(kDim1);
  if (cache_npd) {
    num_tokens_max = key_cache_shape.GetDim(0) * key_cache_shape.GetDim(kDim2);
  }
  if (num_tokens_max != 0 && num_tokens_max < num_tokens) {
    ASC_CPU_LOG_ERROR("num_tokens exceeds max capacity");
    return ge::GRAPH_FAILED;
  }
  if (num_tokens < core_num) {
    core_num = num_tokens;
  }

  key_head_num = key_cache_shape.GetDim(kDim1);
  page_size = key_cache_shape.GetDim(kDim2);
  if (!cache_npd) {
    key_head_num = key_cache_shape.GetDim(kDim2);
    page_size = key_cache_shape.GetDim(kDim1);
  }
  value_head_num = key_head_num;
  batch_size = block_tbl_shape.GetDim(0);

  if (GetSeqLen(context, batch_size, q_seq_len_idx, &q_seq_len) != ge::GRAPH_SUCCESS ||
      GetSeqLen(context, batch_size, kv_seq_len_idx, &kv_seq_len) != ge::GRAPH_SUCCESS) {
    ASC_CPU_LOG_ERROR("Failed to get sequence lengths");
    return ge::GRAPH_FAILED;
  }

  if (batch_size > KMaxBatch) {
    ASC_CPU_LOG_ERROR("Batch size too big");
    return ge::GRAPH_FAILED;
  }

  return ge::GRAPH_SUCCESS;
}

// Main function: uses helper above, then fills tiling buffer
static ge::graphStatus ReshapeAndCacheNpdTiling(gert::TilingContext *context) {
  int32_t cache_npd, kv_npd;
  uint32_t num_tokens, hidden_size, key_head_num, value_head_num, page_size, batch_size, core_num;
  std::vector<int64_t> q_seq_len, kv_seq_len;

  auto status = ValidateAndExtractNpdParams(context, cache_npd, kv_npd, num_tokens, hidden_size, key_head_num,
                                            value_head_num, page_size, batch_size, core_num, q_seq_len, kv_seq_len);
  if (status != ge::GRAPH_SUCCESS) {
    return status;
  }

  ReshapeAndCacheNpdTilingData tiling;
  auto tiling_size = (KLastIdx + KSeqLenArrayNum * batch_size) * sizeof(uint32_t);
  auto tiling_buf = tiling.get_buf();
  auto tiling_key = optiling::ReshapeAndCacheNpdTilingId(context->GetInputDesc(0)->GetDataType(), cache_npd, kv_npd);

  tiling_buf[KTilingId] = tiling_key;
  tiling_buf[KUseCoreNumIdx] = core_num;
  tiling_buf[KNumTokensIdx] = num_tokens;
  tiling_buf[KHiddenSizeIdx] = hidden_size;
  tiling_buf[KKeyHeadNumIdx] = key_head_num;
  tiling_buf[KVHeadNumIdx] = value_head_num;
  tiling_buf[KPageSizeIdx] = page_size;
  tiling_buf[KBatchSizeIdx] = batch_size;

  for (uint32_t i = 0; i < batch_size; i++) {
    tiling_buf[KLastIdx + i] = static_cast<uint32_t>(kv_seq_len.at(i));
    tiling_buf[KLastIdx + i + batch_size] = static_cast<uint32_t>(q_seq_len.at(i));
  }

  context->SetBlockDim(core_num);
  tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
  context->GetRawTilingData()->SetDataSize(tiling_size);
  size_t *currentWorkspace = context->GetWorkspaceSizes(kDim1);
  currentWorkspace[0] = 0;
  context->SetTilingKey(tiling_key);

  ASC_CPU_LOG_INFO("Reshape and cache npd tiling %d %d %d %d %d %d %d %d %d", num_tokens, hidden_size, key_head_num,
                   value_head_num, page_size, batch_size, tiling_key, core_num);

  return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge {
static ge::graphStatus ReshapeAndCacheNpdInferShape(gert::InferShapeContext *context) {
  int32_t cache_kv_layout = std::string(context->GetAttrs()->GetAttrPointer<char>(0)) == std::string(NPD_LAYOUT);
  int32_t kv_layout = std::string(context->GetAttrs()->GetAttrPointer<char>(1)) == std::string(NPD_LAYOUT);
  auto k_idx = static_cast<size_t>(ms_custom_ops::ReshapeAndCacheNpdInputIndex::kInputKeyIndex);
  auto k_shape = context->GetInputShape(k_idx);
  auto t = context->GetInputShape(0)->GetDim(0);
  auto ps = context->GetInputShape(optiling::kDim2)->GetDim(cache_kv_layout ? optiling::kDim2 : optiling::kDim1);
  auto bs = context->GetInputShape(optiling::kDim7)->GetDim(0);
  int max_seq = t + (ps - optiling::kDim1) * bs;
  gert::Shape *out_key_shape = context->GetOutputShape(0);
  gert::Shape *out_value_shape = context->GetOutputShape(optiling::kDim1);

  *out_key_shape = *(context->GetInputShape(0));
  *out_value_shape = *(context->GetInputShape(0));
  if ((kv_layout == 1) && (k_shape->GetDim(0) >= 0)) {
    out_key_shape->SetDim(0, max_seq);
    out_value_shape->SetDim(0, max_seq);
  }
  return GRAPH_SUCCESS;
}

static graphStatus ReshapeAndCacheNpdInferDataType(gert::InferDataTypeContext *context) {
  const auto inputDataType = context->GetInputDataType(0);
  context->SetOutputDataType(0, inputDataType);
  context->SetOutputDataType(optiling::kDim1, inputDataType);
  return ge::GRAPH_SUCCESS;
}
}  // namespace ge

namespace ops {
class ReshapeAndCacheNpd : public OpDef {
 public:
  explicit ReshapeAndCacheNpd(const char *name) : OpDef(name) {
    this->Input("key")
      .ParamType(REQUIRED)
      .DataType({ge::DT_FLOAT16, ge::DT_BF16})
      .Format({ge::FORMAT_ND, ge::FORMAT_ND})
      .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND})
      .AutoContiguous();
    this->Input("value")
      .ParamType(REQUIRED)
      .DataType({ge::DT_FLOAT16, ge::DT_BF16})
      .Format({ge::FORMAT_ND, ge::FORMAT_ND})
      .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND})
      .AutoContiguous();
    this->Input("key_cache")
      .ParamType(REQUIRED)
      .DataType({ge::DT_FLOAT16, ge::DT_BF16})
      .Format({ge::FORMAT_ND, ge::FORMAT_ND})
      .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND})
      .AutoContiguous();
    this->Input("value_cache")
      .ParamType(REQUIRED)
      .DataType({ge::DT_FLOAT16, ge::DT_BF16})
      .Format({ge::FORMAT_ND, ge::FORMAT_ND})
      .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND})
      .AutoContiguous();
    this->Input("slot_mapping")
      .ParamType(REQUIRED)
      .DataType({ge::DT_INT32, ge::DT_INT32})
      .Format({ge::FORMAT_ND, ge::FORMAT_ND})
      .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND})
      .AutoContiguous();
    this->Input("actual_seq_qlen")
      .ParamType(REQUIRED)
      .DataType({ge::DT_INT64, ge::DT_INT64})
      .Format({ge::FORMAT_ND, ge::FORMAT_ND})
      .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND})
      .ValueDepend(Option::OPTIONAL, DependScope::TILING)
      .AutoContiguous();
    this->Input("actual_seq_kvlen")
      .ParamType(REQUIRED)
      .DataType({ge::DT_INT64, ge::DT_INT64})
      .Format({ge::FORMAT_ND, ge::FORMAT_ND})
      .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND})
      .ValueDepend(Option::OPTIONAL, DependScope::TILING)
      .AutoContiguous();
    this->Input("block_tbl")
      .ParamType(REQUIRED)
      .DataType({ge::DT_INT32, ge::DT_INT32})
      .Format({ge::FORMAT_ND, ge::FORMAT_ND})
      .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND})
      .AutoContiguous();
    this->Attr("kv_cache_layout").AttrType(OPTIONAL).String(NPD_LAYOUT);
    this->Attr("key_value_layout").AttrType(OPTIONAL).String(NPD_LAYOUT);
    this->Output("out_key")
      .ParamType(REQUIRED)
      .DataType({ge::DT_FLOAT16, ge::DT_BF16})
      .Format({ge::FORMAT_ND, ge::FORMAT_ND})
      .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
    this->Output("out_value")
      .ParamType(REQUIRED)
      .DataType({ge::DT_FLOAT16, ge::DT_BF16})
      .Format({ge::FORMAT_ND, ge::FORMAT_ND})
      .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});

    this->SetInferShape(ge::ReshapeAndCacheNpdInferShape).SetInferDataType(ge::ReshapeAndCacheNpdInferDataType);
    this->AICore().SetTiling(optiling::ReshapeAndCacheNpdTiling).AddConfig("ascend910b");
  }
};
OP_ADD(ReshapeAndCacheNpd);
}  // namespace ops
