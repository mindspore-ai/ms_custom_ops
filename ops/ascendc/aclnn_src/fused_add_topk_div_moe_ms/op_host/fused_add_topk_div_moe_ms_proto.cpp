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

#include "register/op_def_registry.h"
#include "fused_add_topk_div_moe_ms/op_host/fused_add_topk_div_moe_ms_common.h"

namespace ge {
using fused_add_topk_div_moe_ms::kIndex0;
using fused_add_topk_div_moe_ms::kIndex1;
using fused_add_topk_div_moe_ms::kDim1;
using fused_add_topk_div_moe_ms::kTopK;

static graphStatus FusedAddTopkDivMoeMSInferDataType(gert::InferDataTypeContext *context) {
  auto input_data_type = context->GetInputDataType(kIndex0);
  context->SetOutputDataType(kIndex1, ge::DataType::DT_INT32);
  context->SetOutputDataType(kIndex0, input_data_type);
  return ge::GRAPH_SUCCESS;
}

static ge::graphStatus FusedAddTopkDivMoeMSInferShape(gert::InferShapeContext *context) {
  gert::Shape *expert_weight_shape = context->GetOutputShape(kIndex0);
  gert::Shape *expert_index_shape = context->GetOutputShape(kIndex1);
  const gert::Shape *logits_shape = context->GetInputShape(kIndex0);
  *expert_weight_shape = *logits_shape;
  (*expert_weight_shape)[kDim1] = kTopK;
  *expert_index_shape = *expert_weight_shape;
  return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(FusedAddTopkDivMoeMS)
    .InferShape(FusedAddTopkDivMoeMSInferShape)
    .InferDataType(FusedAddTopkDivMoeMSInferDataType);
}  // namespace ge
