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
#include "reshape_and_cache_npd/op_host/reshape_and_cache_npd_comm.h"

namespace ops {
class ReshapeAndCacheNpd : public OpDef {
 public:
  explicit ReshapeAndCacheNpd(const char *name) : OpDef(name) {
    this->Input("key")
      .ParamType(REQUIRED)
      .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_INT8})
      .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
      .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
      .AutoContiguous();
    this->Input("value")
      .ParamType(REQUIRED)
      .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_INT8})
      .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
      .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
      .AutoContiguous();
    this->Input("key_cache")
      .ParamType(REQUIRED)
      .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_INT8})
      .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
      .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
      .AutoContiguous();
    this->Input("value_cache")
      .ParamType(REQUIRED)
      .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_INT8})
      .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
      .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
      .AutoContiguous();
    this->Input("slot_mapping")
      .ParamType(REQUIRED)
      .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32})
      .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
      .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
      .AutoContiguous();
    this->Input("actual_seq_qlen")
      .ParamType(REQUIRED)
      .DataType({ge::DT_INT64, ge::DT_INT64, ge::DT_INT64})
      .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
      .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
      .ValueDepend(Option::OPTIONAL, DependScope::TILING)
      .AutoContiguous();
    this->Input("actual_seq_kvlen")
      .ParamType(REQUIRED)
      .DataType({ge::DT_INT64, ge::DT_INT64, ge::DT_INT64})
      .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
      .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
      .ValueDepend(Option::OPTIONAL, DependScope::TILING)
      .AutoContiguous();
    this->Input("block_tbl")
      .ParamType(REQUIRED)
      .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32})
      .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
      .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
      .AutoContiguous();
    this->Attr("kv_cache_layout").AttrType(OPTIONAL).String(BNSD_LAYOUT);
    this->Attr("key_value_layout").AttrType(OPTIONAL).String(NPD_LAYOUT);
    this->Output("out_key")
      .ParamType(REQUIRED)
      .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_INT8})
      .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
      .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
    this->Output("out_value")
      .ParamType(REQUIRED)
      .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_INT8})
      .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
      .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

    this->AICore().AddConfig("ascend910b");
  }
};
OP_ADD(ReshapeAndCacheNpd);
}  // namespace ops
