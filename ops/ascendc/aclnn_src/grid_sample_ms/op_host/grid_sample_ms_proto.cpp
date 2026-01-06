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
#include "grid_sample_ms/op_host/grid_sample_ms_comm.h"

namespace ge {
static ge::graphStatus GridSampleMSInferShape(gert::InferShapeContext *context) {
  const gert::Shape *input_shape = context->GetInputShape(kIndex0);
  const gert::Shape *grid_shape = context->GetInputShape(kIndex1);
  gert::Shape *out_shape = context->GetOutputShape(kIndex0);
  *out_shape = *grid_shape;
  (*out_shape)[kDim3] = (*input_shape)[kDim3];
  return GRAPH_SUCCESS;
}
static graphStatus GridSampleMSInferDataType(gert::InferDataTypeContext *context) {
  const auto inputDataType = context->GetInputDataType(kIndex0);
  context->SetOutputDataType(kIndex0, inputDataType);
  return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(GridSampleMS).InferShape(GridSampleMSInferShape).InferDataType(GridSampleMSInferDataType);
}  // namespace ge
