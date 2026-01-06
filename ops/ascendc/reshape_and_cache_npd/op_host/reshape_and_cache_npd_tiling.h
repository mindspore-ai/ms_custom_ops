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

#define KTilingId 0
#define KUseCoreNumIdx 1
#define KNumTokensIdx 2
#define KHiddenSizeIdx 3
#define KKeyHeadNumIdx 4
#define KVHeadNumIdx 5
#define KPageSizeIdx 6
#define KBatchSizeIdx 7
#define KLastIdx 32
#define KSeqLenArrayNum 2
#define KMaxBatch   512

BEGIN_TILING_DATA_DEF(ReshapeAndCacheNpdTilingData)
TILING_DATA_FIELD_DEF_ARR(uint32_t, KLastIdx + KSeqLenArrayNum * KMaxBatch, buf);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(ReshapeAndCacheNpd, ReshapeAndCacheNpdTilingData);
}  // namespace optiling
#endif  // MS_CUSTOM_OPS_OPS_ASCEND_C_RESHAPE_AND_CACHE_NPD_OP_HOST_RESHAPE_AND_CACHE_NPD_TILING_H
