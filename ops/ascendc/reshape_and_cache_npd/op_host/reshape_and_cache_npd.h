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

#ifndef MS_CUSTOM_OPS_ASCENDC_OP_HOST_RESHAPE_AND_CACHE_NPD_H
#define MS_CUSTOM_OPS_ASCENDC_OP_HOST_RESHAPE_AND_CACHE_NPD_H

namespace ms_custom_ops {
#define TH_LAYOUT "TH"
#define ND_LAYOUT "ND"
#define NPD_LAYOUT "NPD"

enum class ReshapeAndCacheNpdInputIndex {
  kInputKeyIndex = 0,
  kInputValueIndex = 1,
  kInputKeyCacheIndex = 2,
  kInputValueCacheIndex = 3,
  kInputSlotMappingIndex = 4,
  kInputQSeqIndex = 5,
  kInputKVSeqIndex = 6,
  kInputBlockTblIndex = 7,
  kInputKVCacheModeIndex = 8,
  kInputKeyValueModeIndex = 9
};

enum class ReshapeAndCacheNpdOutputIndex { kOutputKeyIndex = 0, kOutputValueIndex = 1 };

}  // namespace ms_custom_ops
#endif  // MS_CUSTOM_OPS_ASCENDC_OP_HOST_RESHAPE_AND_CACHE_NPD_H
