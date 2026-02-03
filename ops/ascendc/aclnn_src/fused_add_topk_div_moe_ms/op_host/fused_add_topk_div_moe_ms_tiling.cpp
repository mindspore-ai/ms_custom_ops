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
#include "fused_add_topk_div_moe_ms/op_host/fused_add_topk_div_moe_ms_tiling.h"
#include "fused_add_topk_div_moe_ms/op_host/fused_add_topk_div_moe_ms_common.h"
#include "register/op_def_registry.h"
#include "graph/utils/type_utils.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
using fused_add_topk_div_moe_ms::kDim0;
using fused_add_topk_div_moe_ms::kDim1;
using fused_add_topk_div_moe_ms::kIndex0;
constexpr uint32_t kCoreNum = 8;

static ge::graphStatus FusedAddTopkDivMoeMSTiling(gert::TilingContext *context) {
  FusedAddTopkDivMoeMSTilingData tiling;
  uint32_t tiling_key{0};

  auto logits_shape = context->GetInputShape(0)->GetOriginShape();
  int32_t num_tokens = logits_shape.GetDim(kDim0);
  tiling.set_num_tokens(num_tokens);

  context->SetBlockDim(kCoreNum);
  context->SetTilingKey(tiling_key);
  tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
  context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
  size_t *current_workspace = context->GetWorkspaceSizes(kDim1);
  current_workspace[kIndex0] = 0;
  return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(FusedAddTopkDivMoeMS).Tiling(FusedAddTopkDivMoeMSTiling);
}  // namespace optiling
