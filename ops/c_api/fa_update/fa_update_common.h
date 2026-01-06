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

#ifndef __MS_CUSTOM_OPS_CCSRC_OPS_C_API_FA_UPDATE_H__
#define __MS_CUSTOM_OPS_CCSRC_OPS_C_API_FA_UPDATE_H__

#include <cstdint>

namespace ms_custom_ops {
enum FaUpdateInputIndex : size_t {
  kFaUpdateInputLseIndex = 0,
  kFaUpdateInputLocalOutIndex,
  kFaUpdateParamFaUpdateTypeIndex,
  kFaUpdateParamSpIndex,
  kFaUpdateInputsNum
};

enum PagedCacheLoadOutputIndex : size_t {
  kFaUpdateOutputIndex = 0,
  kFaUpdateOutputsNum
};

#define ALIGN_8(v) (((v) & (8 - 1)) == 0)
static constexpr auto kFaUpdateLocalOutShapeRank = 3;
}  // namespace ms_custom_ops
#endif
