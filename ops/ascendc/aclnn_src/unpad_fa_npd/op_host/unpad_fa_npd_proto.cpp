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
static graphStatus UnpadFaNpdInferShape(gert::InferShapeContext *context) {
  auto q_shape = context->GetInputShape(0);
  gert::Shape *attn_out_shape = context->GetOutputShape(0);
  *attn_out_shape = *q_shape;
  return ge::GRAPH_SUCCESS;
}

static graphStatus UnpadFaNpdInferDataType(gert::InferDataTypeContext *context) {
  const auto inputDataType = context->GetInputDataType(0);
  context->SetOutputDataType(0, inputDataType);
  return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(UnpadFaNpd).InferShape(UnpadFaNpdInferShape).InferDataType(UnpadFaNpdInferDataType);
}  // namespace ge
