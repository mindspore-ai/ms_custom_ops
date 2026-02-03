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

namespace ops {
class FusedAddTopkDivMoeMS : public OpDef {
 public:
  explicit FusedAddTopkDivMoeMS(const char *name) : OpDef(name) {
    this->Input("bias")
      .ParamType(REQUIRED)
      .Format({ge::FORMAT_ND})
      .DataType({ge::DT_FLOAT})
      .UnknownShapeFormat({ge::FORMAT_ND})
      .AutoContiguous();
    this->Input("logits")
      .ParamType(REQUIRED)
      .Format({ge::FORMAT_ND})
      .DataType({ge::DT_FLOAT})
      .UnknownShapeFormat({ge::FORMAT_ND})
      .AutoContiguous();
    this->Output("expert_weight")
      .ParamType(REQUIRED)
      .Format({ge::FORMAT_ND})
      .DataType({ge::DT_FLOAT})
      .UnknownShapeFormat({ge::FORMAT_ND});
    this->Output("expert_index")
      .ParamType(REQUIRED)
      .Format({ge::FORMAT_ND})
      .DataType({ge::DT_INT32})
      .UnknownShapeFormat({ge::FORMAT_ND});


    this->AICore().AddConfig("ascend310p");
  }
};
OP_ADD(FusedAddTopkDivMoeMS);
}  // namespace ops
