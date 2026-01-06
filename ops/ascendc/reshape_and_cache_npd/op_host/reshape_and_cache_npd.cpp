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
#include "../op_host/reshape_and_cache_npd_tiling.h"
#include "register/op_def_registry.h"
#include "graph/utils/type_utils.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
constexpr int32_t KCACHE_VCACHE = 11;
constexpr int32_t KCACHE = 10;
constexpr int32_t NPD_TILING = 1000;
static constexpr auto kRank2 = 2;
static constexpr auto kRank3 = 3;
static constexpr auto kDim1 = 1;
static constexpr auto kDim2 = 2;
static constexpr auto kDim3 = 3;
static constexpr auto kDim4 = 4;
static constexpr auto kDim7 = 7;

#define GE_LOGE if (false) std::cout
int32_t ReshapeAndCacheNpdTilingId(ge::DataType dtype, int32_t cacheConfig, uint32_t npd) {
  int32_t tilingId = 0;
  return tilingId;
}

static ge::graphStatus ReshapeAndCacheNpdTiling(gert::TilingContext *context) {
  if (context->GetInputDesc(0)->GetDataType() == ge::DataType::DT_UNDEFINED ||
      context->GetInputDesc(kDim2)->GetDataType() == ge::DataType::DT_UNDEFINED ||
      context->GetInputDesc(kDim4)->GetDataType() == ge::DataType::DT_UNDEFINED) {
      GE_LOGE << "Input 0, 2, 4(key, key_cache, slot_mapping) are required. Their dtypes cannot be None, but got "
              << context->GetInputDesc(0)->GetDataType() << ", " << context->GetInputDesc(kDim2)->GetDataType() << ", "
              << context->GetInputDesc(kDim4)->GetDataType();
    return ge::GRAPH_FAILED;
  }
  if ((context->GetInputDesc(kDim1)->GetDataType() == ge::DataType::DT_UNDEFINED) ^
      (context->GetInputDesc(kDim3)->GetDataType() == ge::DataType::DT_UNDEFINED)) {
      GE_LOGE << "Input 1, 3 (value, value_cache) should either both be None or have the same dtype, but got "
              << (context->GetInputDesc(kDim1)->GetDataType() == ge::DataType::DT_UNDEFINED) << ", "
              << (context->GetInputDesc(0)->GetDataType() == ge::DataType::DT_UNDEFINED);
    return ge::GRAPH_FAILED;
  }
  ReshapeAndCacheNpdTilingData tiling;
  uint64_t ub_size;
  auto ascendc_platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
  ascendc_platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);
  auto core_num = ascendc_platform.GetCoreNum();
  auto key_shape = context->GetInputShape(0)->GetOriginShape();
  auto key_cache_shape = context->GetInputShape(kDim2)->GetOriginShape();
  auto value_cache_shape = context->GetInputShape(kDim3)->GetOriginShape();
  auto slot_mapping_shape = context->GetInputShape(kDim4)->GetOriginShape();
  auto block_tbl_shape = context->GetInputShape(kDim7)->GetOriginShape();
  if (key_shape.GetDimNum() != kDim3 && key_shape.GetDimNum() != kDim2 || slot_mapping_shape.GetDimNum() != kDim1 ||
      key_cache_shape.GetDimNum() < kDim3) {
      GE_LOGE << "The dim of input should be key=" << kDim3 << " or " << kDim2 << ", slot_mapping=" << kDim1
              << ", key_cache>=" << kDim3 << ", but got " << key_shape.GetDimNum() << ", "
              << slot_mapping_shape.GetDimNum() << ", " << key_cache_shape.GetDimNum();
    return ge::GRAPH_FAILED;
  }
  uint32_t num_tokens = key_shape.GetDim(0);
  uint32_t hidden_size = key_shape.GetDim(kDim1);
  if (key_shape.GetDimNum() == kRank3) {
    hidden_size = key_shape.GetDim(kDim2);
    num_tokens *= key_shape.GetDim(kDim1);
  }
  auto num_tokens_max = key_cache_shape.GetDim(0) * key_cache_shape.GetDim(kDim1);
  const uint32_t *npd = context->GetAttrs()->GetAttrPointer<uint32_t>(0);
  if (*npd) {
    // [BlockNum, N, BlockSize, D]
    num_tokens_max = key_cache_shape.GetDim(0) * key_cache_shape.GetDim(kDim2);
  }
  if (num_tokens_max != 0 && num_tokens_max < num_tokens) {
    GE_LOGE << "The number of tokens should be less than or equal to block_num * block_size = " << num_tokens_max
              << ", but got " << num_tokens;
    return ge::GRAPH_FAILED;
  }
  if (num_tokens < core_num) {
    core_num = num_tokens;
  }

  uint32_t key_head_num = key_cache_shape.GetDim(kDim1);
  uint32_t page_size = key_cache_shape.GetDim(kDim2);
  if (!*npd) {
    key_head_num = key_cache_shape.GetDim(kDim2);
    page_size = key_cache_shape.GetDim(kDim1);
  }
  uint32_t value_head_num = key_head_num;
  uint32_t batch_size = block_tbl_shape.GetDim(0);

  int32_t cache_config = (context->GetInputDesc(kDim1)->GetDataType() == ge::DataType::DT_UNDEFINED &&
                          context->GetInputTensor(kDim3)->GetDataType() == ge::DataType::DT_UNDEFINED)
                           ? kDim1
                           : 0;

  if (cache_config == 0) {
    value_head_num = *npd ? value_cache_shape.GetDim(kDim1) : value_cache_shape.GetDim(kDim2);
  }
  auto tiling_key = optiling::ReshapeAndCacheNpdTilingId(context->GetInputDesc(0)->GetDataType(), cache_config, *npd);

  tiling.set_numTokens(num_tokens);
  tiling.set_hiddenSize(hidden_size);
  tiling.set_kHeadNum(key_head_num);
  tiling.set_vHeadNum(value_head_num);
  tiling.set_pageSize(page_size);
  tiling.set_batchSize(batch_size);
  tiling.set_tilingId(tiling_key);

  context->SetBlockDim(core_num);
  tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
  context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
  size_t *currentWorkspace = context->GetWorkspaceSizes(kDim1);
  currentWorkspace[0] = 0;
  return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge {
static ge::graphStatus ReshapeAndCacheNpdInferShape(gert::InferShapeContext *context) {
  auto t = context->GetInputShape(0)->GetDim(0);
  auto ps = context->GetInputShape(optiling::kDim2)->GetDim(optiling::kDim2);
  auto bs = context->GetInputShape(optiling::kDim7)->GetDim(0);
  int max_seq = t + (ps - optiling::kDim1) * bs;
  gert::Shape *out_key_shape = context->GetOutputShape(0);
  gert::Shape *out_value_shape = context->GetOutputShape(optiling::kDim1);
  *out_key_shape = *(context->GetInputShape(0));
  *out_value_shape = *(context->GetInputShape(0));
  out_key_shape->SetDim(0, max_seq);
  out_value_shape->SetDim(0, max_seq);
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
    this->Input("q_seq")
      .ParamType(REQUIRED)
      .DataType({ge::DT_INT32, ge::DT_INT32})
      .Format({ge::FORMAT_ND, ge::FORMAT_ND})
      .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND})
      .AutoContiguous();
    this->Input("kv_seq")
      .ParamType(REQUIRED)
      .DataType({ge::DT_INT32, ge::DT_INT32})
      .Format({ge::FORMAT_ND, ge::FORMAT_ND})
      .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND})
      .AutoContiguous();
    this->Input("block_tbl")
      .ParamType(REQUIRED)
      .DataType({ge::DT_INT32, ge::DT_INT32})
      .Format({ge::FORMAT_ND, ge::FORMAT_ND})
      .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND})
      .AutoContiguous();
    this->Attr("cache_mode").AttrType(OPTIONAL).Int(1);
    this->Output("k_out")
      .ParamType(REQUIRED)
      .DataType({ge::DT_FLOAT16, ge::DT_BF16})
      .Format({ge::FORMAT_ND, ge::FORMAT_ND})
      .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
    this->Output("v_out")
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
