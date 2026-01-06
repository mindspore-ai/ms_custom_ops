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
#include "unpad_fa_npd/op_host/unpad_fa_npd_comm.h"

namespace ops {
class UnpadFaNpd : public OpDef {
 public:
  explicit UnpadFaNpd(const char *name) : OpDef(name) {
    this->Input("query")
      .ParamType(REQUIRED)
      .DataType({ge::DT_FLOAT16, ge::DT_BF16})
      .Format({ge::FORMAT_ND, ge::FORMAT_ND})
      .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND})
      .AutoContiguous();
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
    this->Input("attn_mask")
      .ParamType(OPTIONAL)
      .DataType({ge::DT_FLOAT16, ge::DT_BF16})
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

    this->Attr("head_num").AttrType(REQUIRED).Int(1);
    this->Attr("scale_value").AttrType(OPTIONAL).Float(1.0f);
    this->Attr("q_input_layout").AttrType(OPTIONAL).String(LAYOUT_TH);
    this->Attr("kv_input_layout").AttrType(OPTIONAL).String(LAYOUT_NPD);
    this->Attr("block_size").AttrType(OPTIONAL).Int(16);
    this->Output("attention_out")
      .ParamType(REQUIRED)
      .DataType({ge::DT_FLOAT16, ge::DT_BF16})
      .Format({ge::FORMAT_ND, ge::FORMAT_ND})
      .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
    this->AICore().AddConfig("ascend910b");
  }
};
OP_ADD(UnpadFaNpd);
}  // namespace ops
