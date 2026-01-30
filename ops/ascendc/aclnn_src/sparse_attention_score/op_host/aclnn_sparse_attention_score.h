/*
 * @Description:
 * @Version: 1.0
 * @Author: c00558882 chenjingheng
 * @Date: 2025-06-13 06:25:16
 * @LastEditors: c00558882 chenjingheng
 * @LastEditTime: 2025-06-16 00:50:42
 */
/**
 * Copyright (c) 2023-2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_LEVEL2_ACLNN_SPARSE_ATTENTION_SCORE_H_
#define OP_API_INC_LEVEL2_ACLNN_SPARSE_ATTENTION_SCORE_H_

#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnSparseAttentionScoreMS的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 */
aclnnStatus aclnnSparseAttentionScoreMSGetWorkspaceSize(
  const aclTensor *query, const aclTensor *key, const aclTensor *value, const aclTensor *realShiftOptional,
  const aclTensor *dropMaskOptional, const aclTensor *paddingMaskOptional, const aclTensor *attenMaskOptional,
  const aclIntArray *prefixOptional, const aclTensor *selectBlockIdxOptional, double scaleValueOptional,
  double keepProbOptional, int64_t preTokensOptional, int64_t nextTokensOptional, int64_t headNum, char *inputLayout,
  int64_t innerPreciseOptional, int64_t sparseModeOptional, uint32_t selectBlockLenOptional,
  const aclTensor *softmaxMaxOut, const aclTensor *softmaxSumOut, const aclTensor *softmaxOutOut,
  const aclTensor *attentionOutOut, uint64_t *workspaceSize, aclOpExecutor **executor);

/**
 * @brief aclnnSparseAttentionScoreMS的第二段接口，用于执行计算。
 */
aclnnStatus aclnnSparseAttentionScoreMS(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                      const aclrtStream stream);

/**
 * @brief aclnnSparseAttentionScoreMSV2的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 */
aclnnStatus aclnnSparseAttentionScoreMSV2GetWorkspaceSize(
  const aclTensor *query, const aclTensor *key, const aclTensor *value, const aclTensor *realShiftOptional,
  const aclTensor *dropMaskOptional, const aclTensor *paddingMaskOptional, const aclTensor *attenMaskOptional,
  const aclIntArray *prefixOptional, const aclIntArray *qStartIdxOptional, const aclIntArray *kvStartIdxOptional,
  const aclTensor *selectBlockIdxOptional, double scaleValueOptional, double keepProbOptional,
  int64_t preTokensOptional, int64_t nextTokensOptional, int64_t headNum, char *inputLayout,
  int64_t innerPreciseOptional, int64_t sparseModeOptional, int64_t pseTypeOptional, uint32_t selectBlockLenOptional,
  const aclTensor *softmaxMaxOut, const aclTensor *softmaxSumOut, const aclTensor *softmaxOutOut,
  const aclTensor *attentionOutOut, uint64_t *workspaceSize, aclOpExecutor **executor);

/**
 * @brief aclnnSparseAttentionScoreMSV2的第二段接口，用于执行计算。
 */
aclnnStatus aclnnSparseAttentionScoreMSV2(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                        const aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_LEVEL2_ACLNN_SPARSE_ATTENTION_SCORE_H_
