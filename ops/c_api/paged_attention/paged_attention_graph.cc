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

#include "ops/c_api/paged_attention/paged_attention_common.h"

#include <map>
#include <string>
#include <utility>
#include <vector>

#include "ops/c_api/utils/attention_utils.h"
#include "ops/framework/ms_kernels_internal/graphmode/internal_kernel_mod.h"

namespace ms_custom_ops {
namespace {

inline bool IsKnownDim(int64_t dim) {
  return dim != abstract::Shape::kShapeDimAny;
}

}  // namespace

static void CheckHeadNumbers(int64_t q_head_num, int64_t kv_head_num) {
  MS_CHECK_VALUE(q_head_num > 0,
    CheckAndConvertUtils::FormatCommMsg(
      "For PagedAttention the q_head_num should not be 0, but got 0."));
  MS_CHECK_VALUE(kv_head_num >= 0,
    CheckAndConvertUtils::FormatCommMsg(
      "For PagedAttention the kv_head_num should be greater than or equal to 0, but got ",
      kv_head_num));
  if (kv_head_num != 0) {
    MS_CHECK_VALUE(
      q_head_num % kv_head_num == 0,
      CheckAndConvertUtils::FormatCommMsg(
        "For PagedAttention the q_head_num must be divisible by kv_head_num, but got q_head_num=",
        q_head_num, ", kv_head_num=", kv_head_num));
  }
}

static void CheckScaleTypeLogN(int64_t scale_type, int64_t quant_type,
                               int64_t calc_type, int64_t compress_type) {
  MS_CHECK_VALUE(
    (scale_type >= PAScaleType::kPA_SCALE_TYPE_TOR &&
     scale_type < PAScaleType::kPA_SCALE_TYPE_MAX),
    CheckAndConvertUtils::FormatCommMsg(
      "For PagedAttention the scale_type is invalid, got ", scale_type));

  if (scale_type == PAScaleType::kPA_SCALE_TYPE_LOGN) {
    MS_CHECK_VALUE((quant_type == PAQuantType::kPA_TYPE_QUANT_UNQUANT),
      CheckAndConvertUtils::FormatCommMsg(
        "In PA scale type logn mode, quant_type must be 0(TYPE_QUANT_UNQUANT/"
        "TYPE_QUANT_UNDEFINED), but got ", quant_type));
    MS_CHECK_VALUE(
      (calc_type == PACalcType::kPA_CALC_TYPE_UNDEFINED),
      CheckAndConvertUtils::FormatCommMsg(
        "In PA scale type logn mode, calc_type feature is not supported, but got calc_type=",
        calc_type));
    MS_CHECK_VALUE(
      (compress_type == PACompressType::kPA_COMPRESS_TYPE_UNDEFINED),
      CheckAndConvertUtils::FormatCommMsg(
        "In PA scale type logn mode, compress_type feature is not supported, but got compress_type=",
        compress_type));
  }
}

static void CheckMLAVHeadSize(int64_t mla_v_head_size) {
  constexpr int64_t kMaxMLAVHeadSize = 576;
  MS_CHECK_VALUE(((mla_v_head_size >= 0 && mla_v_head_size <= kMaxMLAVHeadSize)),
    CheckAndConvertUtils::FormatCommMsg(
      "For PagedAttention(MLA mode) the value head size should be [0, 576], but got ",
      mla_v_head_size));
}

static void CheckInputLayoutAndCalcType(int64_t input_layout, int64_t calc_type,
                                        int64_t quant_type, int64_t compress_type,
                                        bool batch_run_status_enable) {
  MS_CHECK_VALUE(
    ((input_layout == PAInputLayout::kPA_INPUT_LAYOUT_BNSD ||
      input_layout == PAInputLayout::kPA_INPUT_LAYOUT_BSND)),
    CheckAndConvertUtils::FormatCommMsg(
      "For PagedAttention the input layout should be 0(BSND)/1(BNSD), but got ",
      input_layout));

  MS_CHECK_VALUE(
    ((calc_type == PACalcType::kPA_CALC_TYPE_SPEC ||
      calc_type == PACalcType::kPA_CALC_TYPE_UNDEFINED)),
    CheckAndConvertUtils::FormatCommMsg(
      "For PagedAttention the calc_type should be 0(disable MTP)/1(enable MTP), but got ",
      calc_type));

  if (calc_type == PACalcType::kPA_CALC_TYPE_SPEC) {
    MS_CHECK_VALUE(
      (quant_type == PAQuantType::kPA_TYPE_QUANT_UNQUANT),
      CheckAndConvertUtils::FormatCommMsg(
        "In PA MTP scene, quant mode should be "
        "0(TYPE_QUANT_UNQUANT/TYPE_QUANT_UNDEFINED), but now got ",
        quant_type));
    MS_CHECK_VALUE(
      (!batch_run_status_enable),
      CheckAndConvertUtils::FormatCommMsg(
        "In PA MTP scene, batch_run_status_enable should be false, but now got true."));
    MS_CHECK_VALUE(
      (compress_type == PACompressType::kPA_COMPRESS_TYPE_UNDEFINED),
      CheckAndConvertUtils::FormatCommMsg(
        "In PA MTP scene, compress_type should be 0(kPA_COMPRESS_TYPE_UNDEFINED), but now got ",
        compress_type));
  }
}

static void CheckBNSDLayout(int64_t input_layout, int64_t calc_type,
                            int64_t compress_type, int64_t quant_type,
                            int64_t scale_type) {
  if (input_layout == PAInputLayout::kPA_INPUT_LAYOUT_BNSD) {
    MS_CHECK_VALUE(
      (calc_type == PACalcType::kPA_CALC_TYPE_UNDEFINED),
      CheckAndConvertUtils::FormatCommMsg(
        "For PagedAttention when input layout is BNSD, calc_type feature is not supported,"
        " but got ", calc_type));
    MS_CHECK_VALUE(
      (compress_type == PACompressType::kPA_COMPRESS_TYPE_UNDEFINED),
      CheckAndConvertUtils::FormatCommMsg(
        "For PagedAttention when input layout is BNSD, compress_type feature is not supported,"
        " but got ", compress_type));
    MS_CHECK_VALUE(
      (quant_type == PAQuantType::kPA_TYPE_QUANT_UNQUANT),
      CheckAndConvertUtils::FormatCommMsg(
        "For PagedAttention when input layout is BNSD, quant_type must be 0(TYPE_QUANT_UNQUANT),"
        " but got ", quant_type));
    MS_CHECK_VALUE(
      (scale_type == PAScaleType::kPA_SCALE_TYPE_TOR),
      CheckAndConvertUtils::FormatCommMsg(
        "For PagedAttention when input layout is BNSD, scale_type must be 0(kPA_SCALE_TYPE_TOR),"
        " but got ", scale_type));
  }
}

static void CheckCompressType(int64_t compress_type, int64_t quant_type,
                               bool batch_run_status_enable, int64_t mask_type) {
  MS_CHECK_VALUE(
    (compress_type != PACompressType::kPA_COMPRESS_TYPE_MAX),
    CheckAndConvertUtils::FormatCommMsg(
      "In PA compress scene, compress type should not be 3(kPA_COMPRESS_TYPE_MAX)."));

  if (compress_type == PACompressType::kPA_COMPRESS_TYPE_KVHEAD ||
      compress_type == PACompressType::kPA_COMPRESS_TYPE_KVHEAD_ROPE) {
    MS_CHECK_VALUE(
      (quant_type == PAQuantType::kPA_TYPE_QUANT_UNQUANT),
      CheckAndConvertUtils::FormatCommMsg(
        "In PA compress scene, quant_type must be 0(TYPE_QUANT_UNQUANT), but now got ",
        quant_type));
  }

  if (compress_type == PACompressType::kPA_COMPRESS_TYPE_KVHEAD_ROPE) {
    MS_CHECK_VALUE(
      (!batch_run_status_enable),
      CheckAndConvertUtils::FormatCommMsg(
        "In PA COMPRESS_TYPE_KVHEAD_ROPE scene, batch_run_status_enable must be false, but now got true."));
    MS_CHECK_VALUE(
      (mask_type == PAMaskType::kPA_MASK_UNDEFINED),
      CheckAndConvertUtils::FormatCommMsg(
        "In PA COMPRESS_TYPE_KVHEAD_ROPE scene, mask type should not be "
        "0(PA_MASK_UNDEFINED), but now got ", mask_type));
  }
}

static void CheckQuantType(int64_t quant_type, int64_t calc_type,
                            int64_t out_data_type, bool has_quant_offset,
                            int64_t compress_type) {
  if (quant_type == PAQuantType::kPA_TYPE_DEQUANT_FUSION) {
    MS_CHECK_VALUE(
      (calc_type == PACalcType::kPA_CALC_TYPE_UNDEFINED),
      CheckAndConvertUtils::FormatCommMsg(
        "For PagedAttention when quant_type is DEQUANT_FUSION, calc_type feature is not supported,"
        " but got ", calc_type));
  }

  if (quant_type == PAQuantType::kPA_TYPE_QUANT_QKV_OFFLINE ||
      quant_type == PAQuantType::kPA_TYPE_QUANT_QKV_ONLINE) {
    MS_CHECK_VALUE(
      (out_data_type == PAOutDataType::kPA_ACL_FLOAT16 ||
       out_data_type == PAOutDataType::kPA_ACL_BF16),
      CheckAndConvertUtils::FormatCommMsg(
        "In PA full quant scene, out_data_type must be 1(Float16) or 27(BFloat16), but got ",
        out_data_type));
    MS_CHECK_VALUE(
      (!has_quant_offset),
      CheckAndConvertUtils::FormatCommMsg(
        "In PA full quant scene, has_quant_offset must be false, but now got true."));
    MS_CHECK_VALUE(
      (compress_type == PACompressType::kPA_COMPRESS_TYPE_UNDEFINED),
      CheckAndConvertUtils::FormatCommMsg(
        "In PA full quant scene, compress_type should be 0(kPA_COMPRESS_TYPE_UNDEFINED),"
        " but now got ", compress_type));
    MS_CHECK_VALUE(
      (calc_type == PACalcType::kPA_CALC_TYPE_UNDEFINED),
      CheckAndConvertUtils::FormatCommMsg(
        "In PA full quant scene, calc_type feature is not supported, but got ",
        calc_type));
  }
}

static void CheckMLAMode(int64_t mla_v_head_size, int64_t mask_type,
                         int64_t compress_type, int64_t quant_type,
                         int64_t scale_type, int64_t input_layout,
                         int64_t kv_head_num, int64_t calc_type,
                         bool batch_run_status_enable) {
  if (mla_v_head_size > 0) {
    MS_CHECK_VALUE(
      (mask_type != PAMaskType::kPA_MASK_TYPE_ALIBI),
      CheckAndConvertUtils::FormatCommMsg(
        "In PA MLA mode, mask type kPA_MASK_TYPE_ALIBI is not supported, but now got ",
        mask_type));
    MS_CHECK_VALUE(
      (compress_type == PACompressType::kPA_COMPRESS_TYPE_UNDEFINED),
      CheckAndConvertUtils::FormatCommMsg(
        "In PA MLA mode, compress_type should be 0(kPA_COMPRESS_TYPE_UNDEFINED), but now got ",
        compress_type));
    MS_CHECK_VALUE(
      (quant_type != PAQuantType::kPA_TYPE_DEQUANT_FUSION),
      CheckAndConvertUtils::FormatCommMsg(
        "In PA MLA mode, quant_type kPA_TYPE_DEQUANT_FUSION is not supported, but now got ",
        quant_type));
    MS_CHECK_VALUE(
      (scale_type == PAScaleType::kPA_SCALE_TYPE_TOR),
      CheckAndConvertUtils::FormatCommMsg(
        "In PA MLA mode, scale_type must be 0(kPA_SCALE_TYPE_TOR), but now got ",
        scale_type));
    MS_CHECK_VALUE(
      (input_layout == PAInputLayout::kPA_INPUT_LAYOUT_BSND),
      CheckAndConvertUtils::FormatCommMsg(
        "In PA MLA mode, input layout must be 0(BSND), but now got ",
        input_layout));
    MS_CHECK_VALUE(
      (kv_head_num == 1),
      CheckAndConvertUtils::FormatCommMsg(
        "In PA MLA mode, kv_head_num should be 1 (MQA), but now got ",
        kv_head_num));

    if (calc_type == PACalcType::kPA_CALC_TYPE_SPEC) {
      MS_CHECK_VALUE(
        (mask_type != PAMaskType::kPA_MASK_TYPE_NORM),
        CheckAndConvertUtils::FormatCommMsg(
          "In PA MLA MTP scene, mask type kPA_MASK_TYPE_NORM is not supported, but now got ",
          mask_type));
      MS_CHECK_VALUE(
        (quant_type == PAQuantType::kPA_TYPE_QUANT_UNQUANT),
        CheckAndConvertUtils::FormatCommMsg(
          "In PA MLA MTP scene, quant_type must be 0(TYPE_QUANT_UNQUANT), but now got ",
          quant_type));
      MS_CHECK_VALUE(
        (!batch_run_status_enable),
        CheckAndConvertUtils::FormatCommMsg(
          "In PA MLA MTP scene, batch_run_status_enable should be false, but now got true."));
    }
  }
}

static void CheckParams(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) {
  (void)primitive;

  // Extract all parameters
  auto q_head_num =
      input_infos[kPagedAttentionInputQHeadNumIndex]->GetScalarValueWithCheck<int64_t>();
  auto kv_head_num =
      input_infos[kPagedAttentionInputKVHeadNumIndex]->GetScalarValueWithCheck<int64_t>();
  auto scale_type =
      input_infos[kPagedAttentionInputScaleTypeIndex]->GetScalarValueWithCheck<int64_t>();
  auto quant_type =
      input_infos[kPagedAttentionInputQuantTypeIndex]->GetScalarValueWithCheck<int64_t>();
  auto mla_v_head_size =
      input_infos[kPagedAttentionInputMlaVDimHeadSizeIndex]->GetScalarValueWithCheck<int64_t>();
  auto input_layout =
      input_infos[kPagedAttentionInputInputLayoutIndex]->GetScalarValueWithCheck<int64_t>();
  auto calc_type =
      input_infos[kPagedAttentionInputCalcTypeIndex]->GetScalarValueWithCheck<int64_t>();
  auto compress_type =
      input_infos[kPagedAttentionInputCompressTypeIndex]->GetScalarValueWithCheck<int64_t>();
  auto mask_type =
      input_infos[kPagedAttentionInputMaskTypeIndex]->GetScalarValueWithCheck<int64_t>();
  auto batch_run_status_enable =
      input_infos[kPagedAttentionInputBatchRunStatusEnableIndex]->GetScalarValueWithCheck<bool>();
  auto has_quant_offset =
      input_infos[kPagedAttentionInputHasQuantOffsetIndex]->GetScalarValueWithCheck<bool>();
  auto out_data_type =
      input_infos[kPagedAttentionInputOutDataTypeIndex]->GetScalarValueWithCheck<int64_t>();

  // Perform validation checks by calling specialized sub-functions
  CheckHeadNumbers(q_head_num, kv_head_num);
  CheckScaleTypeLogN(scale_type, quant_type, calc_type, compress_type);
  CheckMLAVHeadSize(mla_v_head_size);
  CheckInputLayoutAndCalcType(input_layout, calc_type, quant_type, compress_type,
                               batch_run_status_enable);
  CheckBNSDLayout(input_layout, calc_type, compress_type, quant_type, scale_type);
  CheckCompressType(compress_type, quant_type, batch_run_status_enable, mask_type);
  CheckQuantType(quant_type, calc_type, out_data_type, has_quant_offset, compress_type);
  CheckMLAMode(mla_v_head_size, mask_type, compress_type, quant_type, scale_type,
               input_layout, kv_head_num, calc_type, batch_run_status_enable);
}

static void CheckKVCacheConsistency(int64_t mla_v_dim,
                                     int64_t num_blocks_k, int64_t num_blocks_v,
                                     int64_t block_size_k, int64_t block_size_v,
                                     int64_t head_num_k, int64_t head_num_v) {
  if (mla_v_dim == 0) {
    if (IsKnownDim(num_blocks_k) && IsKnownDim(num_blocks_v)) {
      MS_CHECK_VALUE(
        num_blocks_k == num_blocks_v,
        CheckAndConvertUtils::FormatCommMsg(
          "For PagedAttention the num_blocks of key_cache and value_cache must be same, but got ",
          num_blocks_k, " and ", num_blocks_v));
    }
    if (IsKnownDim(block_size_k) && IsKnownDim(block_size_v)) {
      MS_CHECK_VALUE(
        block_size_k == block_size_v,
        CheckAndConvertUtils::FormatCommMsg(
          "For PagedAttention the block_size of key_cache and value_cache must be same, but got ",
          block_size_k, " and ", block_size_v));
    }
    if (IsKnownDim(head_num_k) && IsKnownDim(head_num_v)) {
      MS_CHECK_VALUE(
        head_num_k == head_num_v,
        CheckAndConvertUtils::FormatCommMsg(
          "For PagedAttention the head_num of key_cache and value_cache must be same, but got ",
          head_num_k, " and ", head_num_v));
    }
  }
}

static void CheckQueryKVHeadSizeConsistency(int64_t head_size_k, int64_t head_size_q) {
  if (IsKnownDim(head_size_k) && IsKnownDim(head_size_q)) {
    MS_CHECK_VALUE(
      head_size_k == head_size_q,
      CheckAndConvertUtils::FormatCommMsg(
        "For PagedAttention the head_size of key_cache and query must be same, but got ",
        head_size_k, " and ", head_size_q));
  }
}

static void CheckNonMLAHeadSizeLimits(int64_t mla_v_dim, int64_t head_size_k, int64_t block_size_k) {
  constexpr int64_t kMaxHeadSize910B = 256;
  constexpr int64_t kMaxHeadSizeProd = 128 * 128;

  if (mla_v_dim == 0 && IsKnownDim(head_size_k) && IsKnownDim(block_size_k)) {
    MS_CHECK_VALUE(
      head_size_k > 0 && head_size_k <= kMaxHeadSize910B,
      CheckAndConvertUtils::FormatCommMsg(
        "For PagedAttention on ND path, head_size of key_cache must be in (0, 256], but got ",
        head_size_k));
    MS_CHECK_VALUE(
      block_size_k * head_size_k <= kMaxHeadSizeProd,
      CheckAndConvertUtils::FormatCommMsg(
        "For PagedAttention on ND path, block_size * head_size must be <= 128 * 128, but got ",
        block_size_k * head_size_k));
  }
}

static void CheckMLAHeadSizeConstraints(int64_t mla_v_dim, int64_t head_size_k,
                                        int64_t head_size_v, int64_t block_size_k) {
  if (mla_v_dim > 0) {
    if (IsKnownDim(head_size_k) && IsKnownDim(head_size_v)) {
      constexpr int64_t kMaxMLAHeadSize = 576;
      MS_CHECK_VALUE(
        head_size_k <= kMaxMLAHeadSize && head_size_v <= kMaxMLAHeadSize,
        CheckAndConvertUtils::FormatCommMsg(
          "For PagedAttention MLA mode, head_size of key_cache and value_cache must be <= 576, but got ",
          head_size_k, " and ", head_size_v));
    }
    if (IsKnownDim(head_size_k) && IsKnownDim(head_size_v) && IsKnownDim(block_size_k)) {
      constexpr int64_t kHeadSizeThreshold = 256;
      constexpr int64_t kBlockSizeLimit = 128;
      if ((head_size_k > kHeadSizeThreshold || head_size_v > kHeadSizeThreshold) &&
          block_size_k > kBlockSizeLimit) {
        MS_CHECK_VALUE(
          false,
          CheckAndConvertUtils::FormatCommMsg(
            "For PagedAttention MLA mode, when head_size > 256, block_size must be <= 128, but got "
            "head_size_k=", head_size_k, ", head_size_v=", head_size_v, ", block_size=", block_size_k));
      }
    }
  }
}

static void CheckKVCacheShapeND(const InferInfoPtrList &input_infos) {
  auto query_shape = input_infos[kPagedAttentionInputQueryIndex]->GetShape();
  auto key_cache_shape = input_infos[kPagedAttentionInputKeyCacheIndex]->GetShape();
  auto value_cache_shape = input_infos[kPagedAttentionInputValueCacheIndex]->GetShape();
  auto mla_v_dim =
      input_infos[kPagedAttentionInputMlaVDimHeadSizeIndex]->GetScalarValueWithCheck<int64_t>();

  if (input_infos[kPagedAttentionInputQueryIndex]->IsDynamic() ||
      input_infos[kPagedAttentionInputKeyCacheIndex]->IsDynamic() ||
      input_infos[kPagedAttentionInputValueCacheIndex]->IsDynamic()) {
    return;
  }
  if (query_shape.size() != kPAQShapeRank || key_cache_shape.size() != kPAKVCacheRank ||
      value_cache_shape.size() != kPAKVCacheRank) {
    // Rank mismatch is handled in basic rank checks.
    return;
  }

  // Extract shape dimensions
  auto num_blocks_k = key_cache_shape[0];
  auto block_size_k = key_cache_shape[1];
  auto head_num_k = key_cache_shape[2];
  auto head_size_k = key_cache_shape[3];

  auto num_blocks_v = value_cache_shape[0];
  auto block_size_v = value_cache_shape[1];
  auto head_num_v = value_cache_shape[2];
  auto head_size_v = value_cache_shape[3];

  auto head_size_q = query_shape[2];

  // Perform validation checks by calling specialized sub-functions
  CheckKVCacheConsistency(mla_v_dim, num_blocks_k, num_blocks_v,
                          block_size_k, block_size_v, head_num_k, head_num_v);
  CheckQueryKVHeadSizeConsistency(head_size_k, head_size_q);
  CheckNonMLAHeadSizeLimits(mla_v_dim, head_size_k, block_size_k);
  CheckMLAHeadSizeConstraints(mla_v_dim, head_size_k, head_size_v, block_size_k);
}

static void CheckMaskFreeShape310P(const InferInfoPtrList &input_infos, int64_t input_format) {
  auto mask_type =
      input_infos[kPagedAttentionInputMaskTypeIndex]->GetScalarValueWithCheck<int64_t>();
  if (mask_type != PAMaskType::kPA_MASK_TYPE_MASK_FREE) {
    return;
  }

  // Mask-free only supports 310P (NZ format).
  if (input_format != PAInputFormat::kKVFormatNZ) {
    MS_CHECK_VALUE(
      false,
      CheckAndConvertUtils::FormatCommMsg(
        "For PagedAttention, MASK_FREE is only supported on 310P (NZ format)."));
  }

  const auto &mask_info = input_infos[kPagedAttentionInputAttnMaskIndex];
  if (!mask_info->IsDynamic()) {
    auto mask_shape = mask_info->GetShape();
    constexpr size_t kMaskRank = 3;
    MS_CHECK_VALUE(
      mask_shape.size() == kMaskRank,
      CheckAndConvertUtils::FormatCommMsg(
        "For PagedAttention MASK_FREE on 310P, the rank of mask must be 3, but got shape: ",
        mask_shape));

    constexpr int64_t kExpectedBatch = 1;
    constexpr int64_t kExpectedBlockSize = 128;

    if (IsKnownDim(mask_shape[0])) {
      MS_CHECK_VALUE(
        mask_shape[0] == kExpectedBatch,
        CheckAndConvertUtils::FormatCommMsg(
          "For PagedAttention MASK_FREE on 310P, mask dim[0] must be 1, but got ", mask_shape[0]));
    }
    if (IsKnownDim(mask_shape[1])) {
      MS_CHECK_VALUE(
        mask_shape[1] == kExpectedBlockSize,
        CheckAndConvertUtils::FormatCommMsg(
          "For PagedAttention MASK_FREE on 310P, mask dim[1] must be 128, but got ", mask_shape[1]));
    }
    if (IsKnownDim(mask_shape[2])) {
      MS_CHECK_VALUE(
        mask_shape[2] == kExpectedBlockSize,
        CheckAndConvertUtils::FormatCommMsg(
          "For PagedAttention MASK_FREE on 310P, mask dim[2] must be 128, but got ", mask_shape[2]));
    }
  }
}

static void CheckShape(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) {
  (void)primitive;
  auto query_shape = input_infos[kPagedAttentionInputQueryIndex]->GetShape();
  auto key_cache_shape = input_infos[kPagedAttentionInputKeyCacheIndex]->GetShape();
  auto value_cache_shape = input_infos[kPagedAttentionInputValueCacheIndex]->GetShape();
  auto block_tables_shape = input_infos[kPagedAttentionInputBlockTablesIndex]->GetShape();
  auto context_len_shape = input_infos[kPagedAttentionInputContextLensIndex]->GetShape();
  // Input format: ND (910B) vs NZ (310P)
  auto input_format =
      input_infos[kPagedAttentionInputInputFormatIndex]->GetScalarValueWithCheck<int64_t>();

  if (!input_infos[kPagedAttentionInputQueryIndex]->IsDynamic()) {
    // 910B/ND: rank must be 3 (T, H, Dh)
    // 310P/NZ: rank must be 2 (T, H*Dh)
    auto q_rank = static_cast<int64_t>(query_shape.size());
    constexpr int64_t kPA310PQueryRank = 2;
    if (input_format == PAInputFormat::kKVFormatNZ) {
      MS_CHECK_VALUE(
        q_rank == kPA310PQueryRank,
        CheckAndConvertUtils::FormatCommMsg(
          "For PagedAttention (310P/NZ) the rank of query must be 2 (T, H*Dh), but got shape: ",
          query_shape));
    } else {
      MS_CHECK_VALUE(
        q_rank == kPAQShapeRank,
        CheckAndConvertUtils::FormatCommMsg(
          "For PagedAttention the rank of query must be ", kPAQShapeRank,
          ", but got shape: ", query_shape));
    }
  }

  if (!input_infos[kPagedAttentionInputKeyCacheIndex]->IsDynamic()) {
    // 910B/ND: rank must be 4 (num_blocks, block_size, num_heads, head_size)
    // 310P/NZ: rank must be 3 (num_blocks, block_size, num_heads*head_size)
    auto expected_rank = (input_format == PAInputFormat::kKVFormatNZ
                          ? kPAKVCacheRankAltas
                          : kPAKVCacheRank);
    MS_CHECK_VALUE(
      key_cache_shape.size() == static_cast<size_t>(expected_rank),
      CheckAndConvertUtils::FormatCommMsg(
        "For PagedAttention the rank of key_cache must be ", expected_rank,
        " (3 for 310P/NZ, 4 for 910B/ND), but got shape: ", key_cache_shape));
  }

  if (!input_infos[kPagedAttentionInputValueCacheIndex]->IsDynamic()) {
    auto expected_rank = (input_format == PAInputFormat::kKVFormatNZ
                          ? kPAKVCacheRankAltas
                          : kPAKVCacheRank);
    MS_CHECK_VALUE(
      value_cache_shape.size() == static_cast<size_t>(expected_rank),
      CheckAndConvertUtils::FormatCommMsg(
        "For PagedAttention the rank of value_cache must be ", expected_rank,
        " (3 for 310P/NZ, 4 for 910B/ND), but got shape: ", value_cache_shape));
  }

  if (!input_infos[kPagedAttentionInputBlockTablesIndex]->IsDynamic()) {
    MS_CHECK_VALUE(
      block_tables_shape.size() == kPABlockTableRank,
      CheckAndConvertUtils::FormatCommMsg(
        "For PA The rank of block table must be ", kPABlockTableRank,
        ", but got shape: ", block_tables_shape));
  }

  if (!input_infos[kPagedAttentionInputContextLensIndex]->IsDynamic()) {
    MS_CHECK_VALUE(
      context_len_shape.size() == kPAContextLenRank,
      CheckAndConvertUtils::FormatCommMsg(
        "For PA The rank of context len must be ", kPAContextLenRank,
        ", but got shape: ", context_len_shape));
  }

  // Head_size / block_size / MLA-specific shape checks for ND path (910B).
  if (input_format == PAInputFormat::kKVFormatND) {
    CheckKVCacheShapeND(input_infos);
  }

  CheckMaskFreeShape310P(input_infos, input_format);
}

static void CheckType(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) {
  auto quant_type =
      input_infos[kPagedAttentionInputQuantTypeIndex]->GetScalarValueWithCheck<int64_t>();
  auto query_dtype = input_infos[kPagedAttentionInputQueryIndex]->GetType();
  auto key_cache_dtype = input_infos[kPagedAttentionInputKeyCacheIndex]->GetType();
  auto value_cache_dtype = input_infos[kPagedAttentionInputValueCacheIndex]->GetType();
  if (quant_type == PAQuantType::kPA_TYPE_QUANT_UNQUANT) {
    MS_CHECK_VALUE(
      ((query_dtype == kNumberTypeFloat16) || (query_dtype == kNumberTypeBFloat16)),
      CheckAndConvertUtils::FormatCommMsg(
        "For PA in unquant mode, query dtype must be float16/bfloat16, but got type: ",
        query_dtype));
    MS_CHECK_VALUE(
      ((key_cache_dtype == kNumberTypeFloat16) || (key_cache_dtype == kNumberTypeBFloat16)),
      CheckAndConvertUtils::FormatCommMsg(
        "For PA in unquant mode, key cache dtype must be float16/bfloat16, but got type: ",
        key_cache_dtype));
    MS_CHECK_VALUE(
      ((value_cache_dtype == kNumberTypeFloat16) ||
        (value_cache_dtype == kNumberTypeBFloat16)),
      CheckAndConvertUtils::FormatCommMsg(
        "For PA in unquant mode,value cache dtype must be float16/bfloat16, but got type: ",
        value_cache_dtype));
  } else if (quant_type == PAQuantType::kPA_TYPE_DEQUANT_FUSION) {
    MS_CHECK_VALUE(
      ((query_dtype == kNumberTypeFloat16) || (query_dtype == kNumberTypeBFloat16)),
      CheckAndConvertUtils::FormatCommMsg(
        "For PA in dequant mode, query dtype must be float16/bfloat16, but got type: ",
        query_dtype));
    MS_CHECK_VALUE(
      ((key_cache_dtype == kNumberTypeInt8)),
      CheckAndConvertUtils::FormatCommMsg(
        "For PA in dequant mode, key cache dtype must be int8, but got type: ",
        key_cache_dtype));
    MS_CHECK_VALUE(
      ((value_cache_dtype == kNumberTypeInt8)),
      CheckAndConvertUtils::FormatCommMsg(
        "For PA in dequant mode,value cache dtype must be int8, but got type: ",
        value_cache_dtype));
  } else if (quant_type == PAQuantType::kPA_TYPE_QUANT_QKV_OFFLINE ||
      quant_type == PAQuantType::kPA_TYPE_QUANT_QKV_ONLINE) {
    MS_CHECK_VALUE(
      ((query_dtype == kNumberTypeInt8)),
      CheckAndConvertUtils::FormatCommMsg(
        "For PA in full quant mode, query dtype must be int8, but got type: ",
        query_dtype));
    MS_CHECK_VALUE(
      ((key_cache_dtype == kNumberTypeInt8)),
      CheckAndConvertUtils::FormatCommMsg(
        "For PA in full quant mode, key cache dtype must be int8, but got type: ",
        key_cache_dtype));
    MS_CHECK_VALUE(
      ((value_cache_dtype == kNumberTypeInt8)),
      CheckAndConvertUtils::FormatCommMsg(
        "For PA in full quant mode, value cache dtype must be int8, but got type: ",
        value_cache_dtype));
  }
}

class OPS_API PagedAttentionFuncImpl : public OpFuncImpl {
 public:
  ShapeArray InferShape(const PrimitivePtr &primitive,
                        const InferInfoPtrList &input_infos) const override {
    if (input_infos.size() != kPagedAttentionInputsNum) {
      MS_LOG(EXCEPTION) << "Paged Attention input args should be equal to "
                        << kPagedAttentionInputsNum
                        << ",but now get " << input_infos.size();
    }

    auto &query_info = input_infos[kPagedAttentionInputQueryIndex];
    auto query_shape = query_info->GetShape();

    CheckParams(primitive, input_infos);
    CheckShape(primitive, input_infos);

    if (query_info->IsDynamic() ||
        input_infos[kPagedAttentionInputKeyCacheIndex]->IsDynamic() ||
        input_infos[kPagedAttentionInputValueCacheIndex]->IsDynamic()) {
      return {query_shape};
    }

    auto mla_v_dim =
        input_infos[kPagedAttentionInputMlaVDimHeadSizeIndex]->GetScalarValueWithCheck<int64_t>();
    if (mla_v_dim == 0) {
      return {query_shape};
    }

    auto key_cache_shape = input_infos[kPagedAttentionInputKeyCacheIndex]->GetShape();
    auto key_head_dim = key_cache_shape[key_cache_shape.size() - 1];
    if ((key_head_dim == abstract::Shape::kShapeDimAny) ||
        (query_shape[query_shape.size() - 1] == abstract::Shape::kShapeDimAny)) {
      query_shape[query_shape.size() - 1] = abstract::Shape::kShapeDimAny;
    } else {
      query_shape[query_shape.size() - 1] =
          query_shape[query_shape.size() - 1] / key_head_dim * mla_v_dim;
    }

    return {query_shape};
  }

  std::vector<TypeId> InferType(const PrimitivePtr &primitive,
                                const InferInfoPtrList &input_infos) const override {
    CheckType(primitive, input_infos);
    auto quant_type =
        input_infos[kPagedAttentionInputQuantTypeIndex]->GetScalarValueWithCheck<int64_t>();
    auto query_type = input_infos[kPagedAttentionInputQueryIndex]->GetType();
    if ((quant_type == PAQuantType::kPA_TYPE_QUANT_QKV_ONLINE) ||
        (quant_type == PAQuantType::kPA_TYPE_QUANT_QKV_OFFLINE)) {
      auto out_data_type =
          input_infos[kPagedAttentionInputOutDataTypeIndex]->GetScalarValueWithCheck<int64_t>();

      switch (out_data_type) {
        case PAOutDataType::kPA_ACL_FLOAT16:
          return {kNumberTypeFloat16};
        case PAOutDataType::kPA_ACL_BF16:
          return {kNumberTypeBFloat16};
        default:
          MS_LOG(EXCEPTION)
              << "In PA full quant scene, we should set the output data type:1(Float16) or 27(BFloat16)";
      }
    }

    return {query_type};
  }

  bool GeneralInferRegistered() const override { return true; }

  std::set<int64_t> GetValueDependArgIndices() const override {
    return {
      kPagedAttentionInputContextLensIndex,
      kPagedAttentionInputQSeqLenIndex,
      kPagedAttentionInputQHeadNumIndex,
      kPagedAttentionInputQKScaleIndex,
      kPagedAttentionInputKVHeadNumIndex,
      kPagedAttentionInputMaskTypeIndex,
      kPagedAttentionInputBatchRunStatusEnableIndex,
      kPagedAttentionInputQuantTypeIndex,
      kPagedAttentionInputOutDataTypeIndex,
      kPagedAttentionInputHasQuantOffsetIndex,
      kPagedAttentionInputCompressTypeIndex,
      kPagedAttentionInputCalcTypeIndex,
      kPagedAttentionInputScaleTypeIndex,
      kPagedAttentionInputInputLayoutIndex,
      kPagedAttentionInputMlaVDimHeadSizeIndex,
      kPagedAttentionInputInputFormatIndex
    };
  }
};

class PagedAttention : public InternalKernelMod {
 public:
  PagedAttention() : InternalKernelMod() {}
  ~PagedAttention() override = default;

 protected:
  internal_v2::InternalOpPtr CreateKernel(
      const internal_v2::InputsImmutableInfoList &inputs,
      const internal_v2::OutputsImmutableInfoList &outputs,
      const std::vector<KernelTensor *> &ms_inputs,
      const std::vector<KernelTensor *> &ms_outputs) override {
    param_.q_head_num =
        static_cast<int32_t>(
            ms_inputs[kPagedAttentionInputQHeadNumIndex]->GetValueWithCheck<int64_t>());
    param_.tor = ms_inputs[kPagedAttentionInputQKScaleIndex]->GetValueWithCheck<float>();
    param_.kv_head_num =
        static_cast<int32_t>(
            ms_inputs[kPagedAttentionInputKVHeadNumIndex]->GetValueWithCheck<int64_t>());
    param_.mask_type =
        static_cast<internal_v2::CustomASDPagedAttentionParam::MaskType>(
            ms_inputs[kPagedAttentionInputMaskTypeIndex]->GetValueWithCheck<int64_t>());
    param_.batch_run_status_enable =
        ms_inputs[kPagedAttentionInputBatchRunStatusEnableIndex]->GetValueWithCheck<bool>();
    param_.quant_type =
        static_cast<internal_v2::CustomASDPagedAttentionParam::QuantType>(
            ms_inputs[kPagedAttentionInputQuantTypeIndex]->GetValueWithCheck<int64_t>());
    param_.out_data_type =
        static_cast<internal_v2::CustomASDPagedAttentionParam::OutDataType>(
            ms_inputs[kPagedAttentionInputOutDataTypeIndex]->GetValueWithCheck<int64_t>());
    param_.has_quant_offset =
        ms_inputs[kPagedAttentionInputHasQuantOffsetIndex]->GetValueWithCheck<bool>();
    param_.compress_type =
        static_cast<internal_v2::CustomASDPagedAttentionParam::CompressType>(
            ms_inputs[kPagedAttentionInputCompressTypeIndex]->GetValueWithCheck<int64_t>());
    param_.calc_type =
        static_cast<internal_v2::CustomASDPagedAttentionParam::CaclType>(
            ms_inputs[kPagedAttentionInputCalcTypeIndex]->GetValueWithCheck<int64_t>());
    param_.scale_type =
        static_cast<internal_v2::CustomASDPagedAttentionParam::ScaleType>(
            ms_inputs[kPagedAttentionInputScaleTypeIndex]->GetValueWithCheck<int64_t>());
    param_.input_layout =
        static_cast<internal_v2::CustomASDPagedAttentionParam::InputLayout>(
            ms_inputs[kPagedAttentionInputInputLayoutIndex]->GetValueWithCheck<int64_t>());
    param_.mla_v_dim =
        static_cast<uint32_t>(
            ms_inputs[kPagedAttentionInputMlaVDimHeadSizeIndex]->GetValueWithCheck<int64_t>());
    if (param_.calc_type ==
        internal_v2::CustomASDPagedAttentionParam::CaclType::kPACalcTypeSpec) {
      param_.q_seq_len =
          ms_inputs[kPagedAttentionInputQSeqLenIndex]->GetValueWithCheck<std::vector<int32_t>>();
    }

    // Get input_format parameter: 0 for ND (910B), 1 for NZ (310P)
    auto input_format =
        static_cast<int32_t>(
            ms_inputs[kPagedAttentionInputInputFormatIndex]->GetValueWithCheck<int64_t>());

    // On 310P (input_format==1), context_lens is passed as NPU tensor input, not from param
    // On 910B (input_format==0), context_lens is passed through param
    constexpr int32_t kInputFormatND = 0;
    constexpr int32_t kInputFormatNZ = 1;
    if (input_format == kInputFormatND) {
      param_.kv_seq_len =
          ms_inputs[kPagedAttentionInputContextLensIndex]->GetValueWithCheck<std::vector<int32_t>>();
    }

    if (param_.batch_run_status_enable) {
      param_.batch_run_status =
          ms_inputs[kPagedAttentionInputBatchRunStatusIndex]->GetValueWithCheck<std::vector<int32_t>>();
    }
    created_flag_ = true;

    // NZ format routing for 310P platform when input_format == 1
    if (input_format == kInputFormatNZ) {
      auto inputs_clone = inputs;
      auto outputs_clone = outputs;
      // Set Query, Key Cache, Value Cache and Mask to FRACTAL_NZ format
      inputs_clone[static_cast<size_t>(kPagedAttentionInputQueryIndex)]
          .SetFormat(internal_v2::kFormatFRACTAL_NZ);
      inputs_clone[static_cast<size_t>(kPagedAttentionInputKeyCacheIndex)]
          .SetFormat(internal_v2::kFormatFRACTAL_NZ);
      inputs_clone[static_cast<size_t>(kPagedAttentionInputValueCacheIndex)]
          .SetFormat(internal_v2::kFormatFRACTAL_NZ);
      inputs_clone[static_cast<size_t>(kPagedAttentionInputAttnMaskIndex)]
          .SetFormat(internal_v2::kFormatFRACTAL_NZ);
      outputs_clone[static_cast<size_t>(kPagedAttentionOutputIndex)]
          .SetFormat(internal_v2::kFormatFRACTAL_NZ);
      return internal_v2::CreateCustomPagedAttentionOp(
          inputs_clone, outputs_clone, param_, internal_v2::kInternalCustomPagedAttention);
    }

    // Default ND format for 910B
    return internal_v2::CreateCustomPagedAttentionOp(
        inputs, outputs, param_, internal_v2::kInternalCustomPagedAttention);
  }

  bool UpdateParam(
      const std::vector<KernelTensor *> &inputs,
      const std::vector<KernelTensor *> &outputs) override {
    if (created_flag_) {
      // the q_seq_len and batch_valid_length are inited in CreateKernel,
      // so there is no need to load them again
      created_flag_ = false;
      return true;
    }

    // Get input_format to determine platform (0=910B, 1=310P)
    auto input_format = static_cast<int32_t>(
        inputs[kPagedAttentionInputInputFormatIndex]->GetValueWithCheck<int64_t>());

    bool need_recreate = false;
    if (param_.calc_type ==
        internal_v2::CustomASDPagedAttentionParam::CaclType::kPACalcTypeSpec) {
      auto q_need_recreate =
          GetSeqLenAndCheckUpdate(inputs[kPagedAttentionInputQSeqLenIndex], &param_.q_seq_len);
      need_recreate |= q_need_recreate;
    }

    // On 310P (input_format==1), context_lens is NPU tensor input, not in param
    // On 910B (input_format==0), context_lens is in param and needs update check
    constexpr int32_t kInputFormatND = 0;
    if (input_format == kInputFormatND) {
      auto kv_need_recreate = GetSeqLenAndCheckUpdate(
          inputs[kPagedAttentionInputContextLensIndex], &param_.kv_seq_len);
      need_recreate |= kv_need_recreate;
    }

    if (need_recreate) {
      auto ret = internal_op_->UpdateParam(&param_);
      if (ret != internal_v2::kInternalOk) {
        MS_LOG(ERROR)
            << "ASD PagedAttention UpdateParam failed, kernel_name: " << kernel_name_;
        return false;
      }
      return true;
    }
    return true;
  }

  uint64_t GenerateTilingKey(const std::vector<KernelTensor *> &inputs) override {
    // User defined CacheKey, the inputs should include all the factors which
    // will affect tiling result.
    return InternalTilingCache::GenerateKey(
        kernel_name_, inputs, param_.q_seq_len, param_.kv_seq_len, param_.mla_v_dim);
  }

  void InitKernelInputsOutputsIndex() override {
    kernel_inputs_index_ = {
      kPagedAttentionInputQueryIndex,
      kPagedAttentionInputKeyCacheIndex,
      kPagedAttentionInputValueCacheIndex,
      kPagedAttentionInputBlockTablesIndex,
      kPagedAttentionInputContextLensIndex,
      kPagedAttentionInputAttnMaskIndex,
      kPagedAttentionInputKDescalekIndex,
      kPagedAttentionInputKOffsetIndex,
      kPagedAttentionInputVDescaleIndex,
      kPagedAttentionInputVOffsetIndex,
      kPagedAttentionInputRazorOffsetIndex,
      kPagedAttentionInputPScaleIndex,
      kPagedAttentionInputLogNIndex,
    };
    kernel_outputs_index_ = {kPagedAttentionOutputIndex};
  }

 private:
  bool created_flag_{false};
  internal_v2::CustomASDPagedAttentionParam param_;
};

}  // namespace ms_custom_ops

REG_GRAPH_MODE_OP(
    paged_attention, ms_custom_ops::PagedAttentionFuncImpl, ms_custom_ops::PagedAttention);
