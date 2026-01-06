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
class ApplyRotaryPosEmbMS : public OpDef {
 public:
  explicit ApplyRotaryPosEmbMS(const char *name) : OpDef(name) {
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

    this->AICore().AddConfig("ascend310p");
  }
};
OP_ADD(ApplyRotaryPosEmbMS);
}  // namespace ops
