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
#include <iostream>
#include "apply_rotary_pos_emb_v3_tiling.h"  // NOLINT(build/include_subdir)
#include "register/op_def_registry.h"
#include "graph/utils/type_utils.h"
#include "utils/log/asc_cpu_log.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
constexpr uint32_t ROPE_V3_USE_TBUF_COUNT = 6;
constexpr uint32_t ROPE_V3_USE_TBUF_COSSIN_COUNT = 2;
constexpr uint32_t ROPE_V3_TILINGKEY_FP16 = 1;
constexpr uint32_t ROPE_V3_TILINGKEY_FP32 = 2;
constexpr uint32_t ROPE_V3_TILINGKEY_FACTOR = 10;
constexpr uint32_t ROPE_V3_ROTARY_DIM_FACTOR = 2;
constexpr uint32_t kIndex0 = 0;
constexpr uint32_t kIndex1 = 1;
constexpr uint32_t kIndex2 = 2;
constexpr uint32_t kDim0 = 0;
constexpr uint32_t kDim1 = 1;
constexpr uint32_t kDim2 = 2;

static ge::graphStatus ApplyRotaryPosEmbV3Tiling(gert::TilingContext *context) {
  ApplyRotaryPosEmbV3TilingData tiling;
  uint32_t tiling_key{0};
  uint64_t ub_total_size;
  auto ascendc_platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
  ascendc_platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_total_size);
  auto coreNum = ascendc_platform.GetCoreNum();

  auto query_shape = context->GetInputShape(kIndex0)->GetOriginShape();
  auto key_shape = context->GetInputShape(kIndex1)->GetOriginShape();
  auto cos_shape = context->GetInputShape(kIndex2)->GetOriginShape();
  ge::DataType query_type = context->GetInputDesc(kIndex0)->GetDataType();
  ge::DataType cos_type = context->GetInputDesc(kIndex2)->GetDataType();

  uint32_t tokens = query_shape.GetDim(kDim0);
  uint32_t query_head_num = query_shape.GetDim(kDim1);
  uint32_t key_head_num = key_shape.GetDim(kDim1);

  uint32_t query_head_dim = query_shape.GetDim(kDim2);
  uint32_t cos_head_dim = cos_shape.GetDim(kDim1);
  uint32_t rotary_dim = cos_head_dim * ROPE_V3_ROTARY_DIM_FACTOR;

  uint32_t is_split = (rotary_dim == query_head_dim ? 0 : 1);
  tiling.set_queryHeadDim(query_head_dim);
  tiling.set_qHeadNum(query_head_num);
  tiling.set_kHeadNum(key_head_num);
  tiling.set_rotaryDim(rotary_dim);
  tiling.set_qHiddenSize(query_head_num * rotary_dim);
  tiling.set_kHiddenSize(key_head_num * rotary_dim);
  tiling.set_cosHeadDim(cos_head_dim);

  if (tokens < coreNum) {
    coreNum = tokens;
  }
  tiling.set_tokensPerCore(static_cast<uint32_t>(tokens / coreNum));
  tiling.set_tokensTail(tokens % coreNum);
  const uint32_t *layout = context->GetAttrs()->GetAttrPointer<uint32_t>(kIndex0);
  const uint32_t *rotaryMode = context->GetAttrs()->GetAttrPointer<uint32_t>(kIndex1);
  tiling.set_layout(*layout);
  tiling.set_rotaryMode(*rotaryMode);
  uint32_t ub_use =
    query_head_dim * (key_head_num + query_head_num) * ge::GetSizeByDataType(query_type) +
    ROPE_V3_USE_TBUF_COSSIN_COUNT * cos_head_dim * ge::GetSizeByDataType(cos_type) +
    (tiling.get_qHiddenSize() + tiling.get_kHiddenSize()) * ROPE_V3_USE_TBUF_COUNT * ge::GetSizeByDataType(query_type);

  if (ub_use > ub_total_size) {
    ASC_CPU_LOG_ERROR(
      "Not support, (cos.head_dim * 2 + query.hidden_size + key.hidden_size + (query.head_num + key.head_num) * 6) * "
      "sizeof(type)  should be less than or equal to UB size, but got %lld > %lld",
      ub_use, ub_total_size);
    return ge::GRAPH_FAILED;
  }
  if (query_type == ge::DataType::DT_FLOAT16) {
    tiling_key = ROPE_V3_TILINGKEY_FP16;
  } else if (query_type == ge::DataType::DT_FLOAT) {
    tiling_key = ROPE_V3_TILINGKEY_FP32;
  }
  tiling_key = tiling_key * ROPE_V3_TILINGKEY_FACTOR + is_split;

  context->SetBlockDim(coreNum);
  context->SetTilingKey(tiling_key);
  tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
  context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
  size_t *currentWorkspace = context->GetWorkspaceSizes(1);
  currentWorkspace[0] = 0;
  return ge::GRAPH_SUCCESS;
}
}  // namespace optiling
namespace ge {
static ge::graphStatus ApplyRotaryPosEmbV3InferShape(gert::InferShapeContext *context) {
  const gert::Shape *query_shape = context->GetInputShape(0);
  const gert::Shape *key_shape = context->GetInputShape(1);
  gert::Shape *out_query_shape = context->GetOutputShape(0);
  gert::Shape *out_key_shape = context->GetOutputShape(1);
  *out_query_shape = *query_shape;
  *out_key_shape = *key_shape;
  return GRAPH_SUCCESS;
}
static graphStatus ApplyRotaryPosEmbV3InferDataType(gert::InferDataTypeContext *context) {
  const auto inputDataType = context->GetInputDataType(0);
  context->SetOutputDataType(0, inputDataType);
  context->SetOutputDataType(1, inputDataType);
  return ge::GRAPH_SUCCESS;
}
}  // namespace ge

namespace ops {
class ApplyRotaryPosEmbV3 : public OpDef {
 public:
  explicit ApplyRotaryPosEmbV3(const char *name) : OpDef(name) {
    this->Input("query")
      .ParamType(REQUIRED)
      .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
      .Format({ge::FORMAT_ND, ge::FORMAT_ND})
      .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND})
      .AutoContiguous();
    this->Input("key")
      .ParamType(REQUIRED)
      .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
      .Format({ge::FORMAT_ND, ge::FORMAT_ND})
      .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND})
      .AutoContiguous();
    this->Input("cos")
      .ParamType(REQUIRED)
      .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
      .Format({ge::FORMAT_ND, ge::FORMAT_ND})
      .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND})
      .AutoContiguous();
    this->Input("sin")
      .ParamType(REQUIRED)
      .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
      .Format({ge::FORMAT_ND, ge::FORMAT_ND})
      .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND})
      .AutoContiguous();
    this->Output("query")
      .ParamType(REQUIRED)
      .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
      .Format({ge::FORMAT_ND, ge::FORMAT_ND})
      .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
    this->Output("key")
      .ParamType(REQUIRED)
      .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
      .Format({ge::FORMAT_ND, ge::FORMAT_ND})
      .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
    this->Attr("layout").AttrType(OPTIONAL).Int(1);
    this->Attr("rotary_mode").AttrType(OPTIONAL).String("interleave");

    this->SetInferShape(ge::ApplyRotaryPosEmbV3InferShape).SetInferDataType(ge::ApplyRotaryPosEmbV3InferDataType);
    this->AICore().SetTiling(optiling::ApplyRotaryPosEmbV3Tiling).AddConfig("ascend310p");
  }
};
OP_ADD(ApplyRotaryPosEmbV3);
}  // namespace ops
