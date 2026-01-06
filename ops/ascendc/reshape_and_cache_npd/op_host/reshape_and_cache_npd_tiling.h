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
#ifndef MS_CUSTOM_OPS_OPS_ASCEND_C_RESHAPE_AND_CACHE_NPD_OP_HOST_RESHAPE_AND_CACHE_NPD_TILING_H
#define MS_CUSTOM_OPS_OPS_ASCEND_C_RESHAPE_AND_CACHE_NPD_OP_HOST_RESHAPE_AND_CACHE_NPD_TILING_H
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(ReshapeAndCacheNpdTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, tilingId);
  TILING_DATA_FIELD_DEF(uint32_t, useCoreNum);
  TILING_DATA_FIELD_DEF(uint32_t, numTokens);
  TILING_DATA_FIELD_DEF(uint32_t, hiddenSize);
  TILING_DATA_FIELD_DEF(uint32_t, kHeadNum);
  TILING_DATA_FIELD_DEF(uint32_t, vHeadNum);
  TILING_DATA_FIELD_DEF(uint32_t, pageSize);
  TILING_DATA_FIELD_DEF(uint32_t, batchSize);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(ReshapeAndCacheNpd, ReshapeAndCacheNpdTilingData)
}  // namespace optiling
#endif  // MS_CUSTOM_OPS_OPS_ASCEND_C_RESHAPE_AND_CACHE_NPD_OP_HOST_RESHAPE_AND_CACHE_NPD_TILING_H
