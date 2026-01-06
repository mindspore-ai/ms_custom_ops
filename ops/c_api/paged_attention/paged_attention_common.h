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

#ifndef MS_CUSTOM_OPS_OPS_C_API_PAGED_ATTENTION_PAGED_ATTENTION_COMMON_H_
#define MS_CUSTOM_OPS_OPS_C_API_PAGED_ATTENTION_PAGED_ATTENTION_COMMON_H_

#include <cstdint>

namespace ms_custom_ops {

enum PagedAttentionInputIndex : int32_t {
    kPagedAttentionInputQueryIndex = 0,              // 0
    kPagedAttentionInputKeyCacheIndex,               // 1
    kPagedAttentionInputValueCacheIndex,             // 2
    kPagedAttentionInputBlockTablesIndex,            // 3
    kPagedAttentionInputContextLensIndex,            // 4
    kPagedAttentionInputAttnMaskIndex,               // 5
    kPagedAttentionInputBatchRunStatusIndex,         // 6
    kPagedAttentionInputKDescalekIndex,              // 7
    kPagedAttentionInputKOffsetIndex,                // 8
    kPagedAttentionInputVDescaleIndex,               // 9
    kPagedAttentionInputVOffsetIndex,                // 10
    kPagedAttentionInputRazorOffsetIndex,            // 11
    kPagedAttentionInputPScaleIndex,                 // 12
    kPagedAttentionInputLogNIndex,                   // 13
    kPagedAttentionInputQSeqLenIndex,                // 14
    kPagedAttentionInputQHeadNumIndex,               // 15
    kPagedAttentionInputQKScaleIndex,                // 16
    kPagedAttentionInputKVHeadNumIndex,              // 17
    kPagedAttentionInputMaskTypeIndex,               // 18
    kPagedAttentionInputBatchRunStatusEnableIndex,   // 19
    kPagedAttentionInputQuantTypeIndex,              // 20
    kPagedAttentionInputOutDataTypeIndex,            // 21
    kPagedAttentionInputHasQuantOffsetIndex,         // 22
    kPagedAttentionInputCompressTypeIndex,           // 23
    kPagedAttentionInputCalcTypeIndex,               // 24
    kPagedAttentionInputScaleTypeIndex,              // 25
    kPagedAttentionInputInputLayoutIndex,            // 26
    kPagedAttentionInputMlaVDimHeadSizeIndex,        // 27
    kPagedAttentionInputInputFormatIndex,            // 28
    kPagedAttentionInputsNum                         // 29
};

enum PAOutputIndex : int32_t {
    kPagedAttentionOutputIndex = 0,
    kPagedAttentionOutputNum,
};

enum PAMaskType : int32_t {
    kPA_MASK_UNDEFINED = 0,
    kPA_MASK_TYPE_NORM,
    kPA_MASK_TYPE_ALIBI,
    kPA_MASK_TYPE_SPEC,
    kPA_MASK_TYPE_MASK_FREE,
};

enum PAQuantType : int32_t {
    kPA_TYPE_QUANT_UNDEFINED = 0,
    kPA_TYPE_QUANT_UNQUANT = 0,
    kPA_TYPE_DEQUANT_FUSION,
    kPA_TYPE_QUANT_QKV_OFFLINE,
    kPA_TYPE_QUANT_QKV_ONLINE,
};

enum PACompressType : int32_t {
    kPA_COMPRESS_TYPE_UNDEFINED = 0,
    kPA_COMPRESS_TYPE_KVHEAD,
    kPA_COMPRESS_TYPE_KVHEAD_ROPE,
    kPA_COMPRESS_TYPE_MAX,
};

enum PACalcType : int32_t {
    kPA_CALC_TYPE_UNDEFINED = 0,
    kPA_CALC_TYPE_SPEC,
};

enum PAOutDataType : int32_t {
    kPA_ACL_DT_UNDEFINED = -1,
    kPA_ACL_FLOAT16 = 1,
    kPA_ACL_BF16 = 27,
};

enum PAInputLayout : int32_t {
    kPA_INPUT_LAYOUT_BSND = 0,
    kPA_INPUT_LAYOUT_BNSD = 1,
};

enum PAScaleType : int32_t {
    kPA_SCALE_TYPE_TOR = 0,
    kPA_SCALE_TYPE_LOGN,
    kPA_SCALE_TYPE_MAX
};

enum PAInputFormat : int8_t {
    kKVFormatND = 0,
    kKVFormatNZ
};

static constexpr auto kPAQShapeRank = 3;
static constexpr auto kPAKVCacheRank = 4;
static constexpr auto kPAKVCacheRankAltas = 3;
static constexpr auto kPABlockTableRank = 2;
static constexpr auto kPAContextLenRank = 1;

}  // namespace ms_custom_ops

#endif  // MS_CUSTOM_OPS_OPS_C_API_PAGED_ATTENTION_PAGED_ATTENTION_COMMON_H_
