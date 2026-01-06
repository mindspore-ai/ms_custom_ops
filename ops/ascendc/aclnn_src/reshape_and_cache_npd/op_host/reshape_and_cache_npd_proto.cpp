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

#include <string>
#include "register/op_def_registry.h"
#include "reshape_and_cache_npd/op_host/reshape_and_cache_npd_comm.h"

namespace ge {
static ge::graphStatus ReshapeAndCacheNpdInferShape(gert::InferShapeContext *context) {
  int32_t cache_kv_layout = std::string(context->GetAttrs()->GetAttrPointer<char>(0)) == std::string(BNSD_LAYOUT);
  auto k_idx = static_cast<size_t>(ReshapeAndCacheNpdInputIndex::kInputKeyIndex);
  auto k_cache_idx = static_cast<size_t>(ReshapeAndCacheNpdInputIndex::kInputKeyCacheIndex);
  auto k_out_idx = static_cast<size_t>(ReshapeAndCacheNpdOutputIndex::kOutputKeyIndex);
  auto v_out_idx = static_cast<size_t>(ReshapeAndCacheNpdOutputIndex::kOutputValueIndex);
  auto k_shape = context->GetInputShape(k_idx);
  auto blk_size = context->GetInputShape(k_cache_idx)->GetDim(cache_kv_layout ? kDim2 : kDim1);
  auto blk_num = context->GetInputShape(k_cache_idx)->GetDim(0);

  gert::Shape *out_key_shape = context->GetOutputShape(k_out_idx);
  gert::Shape *out_value_shape = context->GetOutputShape(v_out_idx);

  *out_key_shape = *k_shape;
  *out_value_shape = *k_shape;
  if (k_shape->GetDim(0) >= 0) {
    out_key_shape->SetDim(0, blk_num * blk_size);
    out_value_shape->SetDim(0, blk_num * blk_size);
  }
  return GRAPH_SUCCESS;
}

static graphStatus ReshapeAndCacheNpdInferDataType(gert::InferDataTypeContext *context) {
  auto k_idx = static_cast<size_t>(ReshapeAndCacheNpdInputIndex::kInputKeyIndex);
  auto k_out_idx = static_cast<size_t>(ReshapeAndCacheNpdOutputIndex::kOutputKeyIndex);
  auto v_out_idx = static_cast<size_t>(ReshapeAndCacheNpdOutputIndex::kOutputValueIndex);

  const auto inputDataType = context->GetInputDataType(k_idx);
  context->SetOutputDataType(k_out_idx, inputDataType);
  context->SetOutputDataType(v_out_idx, inputDataType);
  return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(ReshapeAndCacheNpd)
  .InferShape(ReshapeAndCacheNpdInferShape)
  .InferDataType(ReshapeAndCacheNpdInferDataType);
}  // namespace ge
