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

#ifndef __MS_CUSTOM_OPS_OPS_C_API_SPARSE_FLASH_ATTENTION_SPARSE_FLASH_ATTENTION_COMMON_H__
#define __MS_CUSTOM_OPS_OPS_C_API_SPARSE_FLASH_ATTENTION_SPARSE_FLASH_ATTENTION_COMMON_H__

#include <cstdint>
#include "mindspore/include/custom_op_api.h"

namespace ms_custom_ops {
static constexpr auto kQueryIndex = 0;
static constexpr auto kKeyIndex = 1;
static constexpr auto kValueIndex = 2;
static constexpr auto kSparseIndicesIndex = 3;
static constexpr auto kBlockTable = 4;
static constexpr auto kQSeqLenIndex = 5;
static constexpr auto kKSeqLenIndex = 6;
static constexpr auto kQRopeIndex = 7;
static constexpr auto kKRopeIndex = 8;
static constexpr auto kScaleValueIndex = 9;
static constexpr auto kSparseBlockSizeIndex = 10;
static constexpr auto kLayoutQueryIndex = 11;
static constexpr auto kLayoutKVIndex = 12;
static constexpr auto kSparseModeIndex = 13;

static constexpr auto kLayoutQueryMaxInt = 1;
static constexpr auto kLayoutKVMaxInt = 1;

static const std::vector<std::string> kQueryLayouts = {"BSND", "TND"};
static const std::vector<std::string> kKVLayouts = {"BSND", "PA_BSND"};

inline void CheckLayout(int64_t layout_query, int64_t layout_kv) {
  if (layout_query > kLayoutQueryMaxInt) {
    MS_LOG(EXCEPTION) << "The layout_query of sparse_flash_attention_ascendc must be less than " << kLayoutQueryMaxInt
                      << ", but got: " << layout_query;
  }

  if (layout_kv > kLayoutKVMaxInt) {
    MS_LOG(EXCEPTION) << "The layout_kv of sparse_flash_attention_ascendc must be less than " << kLayoutKVMaxInt
                      << ", but got: " << layout_kv;
  }
}
}  // namespace ms_custom_ops
#endif  //  __MS_CUSTOM_OPS_OPS_C_API_SPARSE_FLASH_ATTENTION_SPARSE_FLASH_ATTENTION_COMMON_H__
