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
#ifndef ADD_CUSTOM_TILING_H
#define ADD_CUSTOM_TILING_H
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(ApplyRotaryPosEmbV3TilingData)
  TILING_DATA_FIELD_DEF(uint32_t, tilingId);
  TILING_DATA_FIELD_DEF(uint32_t, useCoreNum);
  TILING_DATA_FIELD_DEF(uint32_t, tokensPerCore);
  TILING_DATA_FIELD_DEF(uint32_t, tokensTail);
  TILING_DATA_FIELD_DEF(uint32_t, qHeadNum);
  TILING_DATA_FIELD_DEF(uint32_t, kHeadNum);
  TILING_DATA_FIELD_DEF(uint32_t, qHiddenSize);
  TILING_DATA_FIELD_DEF(uint32_t, kHiddenSize);
  TILING_DATA_FIELD_DEF(uint32_t, queryHeadDim);
  TILING_DATA_FIELD_DEF(uint32_t, cosHeadDim);
  TILING_DATA_FIELD_DEF(uint32_t, rotaryDim);
  TILING_DATA_FIELD_DEF(uint32_t, layout);
  TILING_DATA_FIELD_DEF(uint32_t, rotaryMode);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(ApplyRotaryPosEmbV3, ApplyRotaryPosEmbV3TilingData)
}
#endif // ADD_CUSTOM_TILING_H
