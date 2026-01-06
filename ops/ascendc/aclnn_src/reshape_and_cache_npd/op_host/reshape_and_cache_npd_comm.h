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
#ifndef MS_CUSTOM_OPS_OPS_ASCEND_C_RESHAPE_AND_CACHE_NPD_OP_HOST_RESHAPE_AND_CACHE_NPD_COMM_H
#define MS_CUSTOM_OPS_OPS_ASCEND_C_RESHAPE_AND_CACHE_NPD_OP_HOST_RESHAPE_AND_CACHE_NPD_COMM_H

namespace {
static constexpr auto kRank2 = 2;
static constexpr auto kRank3 = 3;
static constexpr auto kDim1 = 1;
static constexpr auto kDim2 = 2;
static constexpr auto kDim3 = 3;
static constexpr auto kDim4 = 4;
static constexpr auto kDim7 = 7;

#define TH_LAYOUT "TH"
#define NPD_LAYOUT "NPD"
#define BSND_LAYOUT "BSND"
#define BNSD_LAYOUT "BNSD"

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
}  // namespace
#endif  // MS_CUSTOM_OPS_OPS_ASCEND_C_RESHAPE_AND_CACHE_NPD_OP_HOST_RESHAPE_AND_CACHE_NPD_COMM_H
