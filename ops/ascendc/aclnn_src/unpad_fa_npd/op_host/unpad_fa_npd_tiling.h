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

#ifndef MS_CUSTOM_OPS_OPS_ASCENDC_UNPAD_FA_NPD_OP_HOST_UNPAD_FA_NPD_TILING_H
#define MS_CUSTOM_OPS_OPS_ASCENDC_UNPAD_FA_NPD_OP_HOST_UNPAD_FA_NPD_TILING_H
#include "register/tilingdata_base.h"

namespace optiling {
#define KBatchIdx 0
#define KMaxSeqIdx 1
#define KInnerBatchSizeIdx 2
#define KEmbeddingSizeIdx 3
#define KBvHeadNumIdx 4
#define KTorIdx 5
#define KHeadStrideIdx 6
#define KMaskStrideIdx 7
#define KIsTriuMaskIdx 8
#define KTotalQBlkNumIdx 9
#define KIsClampIdx 10
#define KClampMinUptrIdx 11
#define KClampMaxUptrIdx 12
#define KNoneIdx 13
#define KTilingHeadSizeIdx 14
#define KTilingParaSizeIdx 15
#define KKeyIdx 16
#define KIsLongSeqIdx 17
#define KMaxKVSeqIdx 18
#define KIsAlibiMaskIdx 19
#define KMaskTypeIdx 20
#define KAlibiCompresOffsetIdx 21
#define KAlibiLeftAlignIdx 22
#define KEmbeddingSizeVIdx 23
#define KQuantTypeIdx 24
#define KDataShapeTypeIdx 25
#define KScaleTypeIdx 26
#define KWindowSizeIdx 27
#define KMaxNumBlocksIdx 28
#define KQMaxSeqLenIdx 29
#define KPreTokensIdx 30
#define KNextTokensIdx 31
#define KRazorLenIdx 32
#define KTileQIdx 33
#define KTileKvIdx 34
#define KTextQLenIdx 35
#define KTextKvLenIdx 36
#define KKvLayoutIdx 37
#define KPageSizeIdx 38
#define KWorkspaceIdx 39
#define KHeadLastIdx 40

#define KQSeqLenIdx 0
#define KKvSeqLenIdx 1
#define KMMIdx 2
#define KNNIdx 3
#define KAddrQSeqOffsetHighIdx 4
#define KAddrQSeqOffsetLowIdx 5
#define KAddrKSeqOffsetHighIdx 6
#define KAddrKSeqOffsetLowIdx 7
#define KAddrVSeqOffsetHighIdx 8
#define KAddrVSeqOffsetLowIdx 9
#define KAddrOSeqOffsetHighIdx 10
#define KAddrOSeqOffsetLowIdx 11
#define KSplitIdx 12
#define KTotalQBlockIdx 13
#define KStateIdx 14
#define KParaLastIdx 15

#define KMaxBatch 512

BEGIN_TILING_DATA_DEF(UnpadFaNpdTilingData)
TILING_DATA_FIELD_DEF_ARR(uint32_t, KHeadLastIdx + KParaLastIdx * KMaxBatch, buf);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(UnpadFaNpd, UnpadFaNpdTilingData)
}  // namespace optiling
#endif  // MS_CUSTOM_OPS_OPS_ASCENDC_UNPAD_FA_NPD_OP_HOST_UNPAD_FA_NPD_TILING_H
