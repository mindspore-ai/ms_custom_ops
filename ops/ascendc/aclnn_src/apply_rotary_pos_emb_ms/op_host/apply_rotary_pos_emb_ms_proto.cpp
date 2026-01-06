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

namespace ge {
static ge::graphStatus ApplyRotaryPosEmbMSInferShape(gert::InferShapeContext *context) {
  const gert::Shape *query_shape = context->GetInputShape(0);
  const gert::Shape *key_shape = context->GetInputShape(1);
  gert::Shape *out_query_shape = context->GetOutputShape(0);
  gert::Shape *out_key_shape = context->GetOutputShape(1);
  *out_query_shape = *query_shape;
  *out_key_shape = *key_shape;
  return GRAPH_SUCCESS;
}

static graphStatus ApplyRotaryPosEmbMSInferDataType(gert::InferDataTypeContext *context) {
  const auto inputDataType = context->GetInputDataType(0);
  context->SetOutputDataType(0, inputDataType);
  context->SetOutputDataType(1, inputDataType);
  return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(ApplyRotaryPosEmbMS)
  .InferShape(ApplyRotaryPosEmbMSInferShape)
  .InferDataType(ApplyRotaryPosEmbMSInferDataType);
}  // namespace ge
