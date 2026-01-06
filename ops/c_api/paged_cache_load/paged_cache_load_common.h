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

#ifndef __MS_CUSTOM_OPS_CCSRC_OPS_MS_KERNELS_INTERNAL_PAGED_CACHE_LOAD_H__
#define __MS_CUSTOM_OPS_CCSRC_OPS_MS_KERNELS_INTERNAL_PAGED_CACHE_LOAD_H__

#include <cstdint>
#include "ops/framework/ms_kernels_internal/graphmode/internal_kernel_mod.h"
#include "ops/framework/ms_kernels_internal/pyboost/internal_pyboost_runner.h"
#include "ops/framework/utils.h"

namespace ms_custom_ops {
enum PagedCacheLoadInputIndex : size_t {
  kPCLInputKeyCacheIndex = 0,
  kPCLInputValueCacheIndex,
  kPCLInputBlockTableIndex,
  kPCLInputSeqLensIndex,
  kPCLInputSeqStartsIndex,
  kPCLInputParamKvCacheCfgIndex,
  kPCLInputParamIsSeqLensCumsumTypeIndex,
  kPCLInputParamHasSeqStartsIndex,
  kPCLInputsNum
};

enum PagedCacheLoadOutputIndex : size_t { kPCLOutputKeyOutIndex = 0, kPCLOutputValueOutIndex, kPCLOutputsNum };

inline constexpr int64_t kNumHeadsIndex = 2;
inline constexpr int64_t kHeadSizeIndex = 3;
inline constexpr int64_t kNdFormatType = 0;
inline constexpr int64_t kNzFormatType = 1;
inline constexpr int64_t kNumHeadsMulHeadSizeIndex = 2;

inline internal_v2::InternalOpPtr CreatePagedCacheLoadOpWithFormat(const internal_v2::InputsImmutableInfoList &inputs,
                                                                   const internal_v2::OutputsImmutableInfoList &outputs,
                                                                   const internal_v2::PagedCacheLoadParam &param) {
  if (param.kv_cache_cfg_type == 1) {
    auto inputs_clone = inputs;
    inputs_clone[kPCLInputKeyCacheIndex].SetFormat(internal_v2::kFormatFRACTAL_NZ);
    inputs_clone[kPCLInputValueCacheIndex].SetFormat(internal_v2::kFormatFRACTAL_NZ);
    return internal_v2::CreatePagedCacheLoadOp(inputs_clone, outputs, param,
                                               internal_v2::kInternalPagedCacheLoadOpName);
  }
  return internal_v2::CreatePagedCacheLoadOp(inputs, outputs, param, internal_v2::kInternalPagedCacheLoadOpName);
}
}  // namespace ms_custom_ops
#endif
