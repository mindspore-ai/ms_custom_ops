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
#include "grid_sample_ms/op_host/grid_sample_ms_tiling.h"
#include "grid_sample_ms/op_host/grid_sample_ms_comm.h"
#include "register/op_def_registry.h"
#include "graph/utils/type_utils.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
constexpr uint32_t kCoreNum = 8;
static ge::graphStatus GridSampleMSTiling(gert::TilingContext *context) {
  GridSampleMSTilingData tiling;
  uint32_t tiling_key{0};

  auto input_shape = context->GetInputShape(0)->GetOriginShape();
  auto grid_shape = context->GetInputShape(1)->GetOriginShape();

  int32_t n_in = input_shape.GetDim(kDim0);
  float h_in = static_cast<float>(input_shape.GetDim(kDim1));
  float w_in = static_cast<float>(input_shape.GetDim(kDim2));
  int32_t c_in = input_shape.GetDim(kDim3);

  int32_t h_out = grid_shape.GetDim(kDim1);
  int32_t w_out = grid_shape.GetDim(kDim2);

  tiling.set_h_in(h_in);
  tiling.set_w_in(w_in);
  tiling.set_h_out(h_out);
  tiling.set_w_out(w_out);
  tiling.set_n_in(n_in);
  tiling.set_c_in(c_in);

  context->SetBlockDim(kCoreNum);
  context->SetTilingKey(tiling_key);
  tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
  context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
  size_t *currentWorkspace = context->GetWorkspaceSizes(kDim1);
  currentWorkspace[kIndex0] = 0;
  return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(GridSampleMS).Tiling(GridSampleMSTiling);
}  // namespace optiling
