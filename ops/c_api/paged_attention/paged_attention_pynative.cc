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

#include <cstdint>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "mindspore/include/custom_op_api.h"
#include "ops/c_api/utils/attention_utils.h"
#include "ops/framework/ms_kernels_internal/pyboost/internal_pyboost_runner.h"
#include "ops/framework/utils.h"

namespace ms_custom_ops {
namespace {

// ------------------------ Param checks ---------------------------------------------

void CheckHeadNumbers(int64_t q_head_num, int64_t kv_head_num) {
    if (q_head_num <= 0) {
        MS_EXCEPTION(ValueError) << "For PagedAttention the q_head_num should be greater than 0, but got "
                                 << q_head_num << ".";
    }
    if (kv_head_num < 0) {
        MS_EXCEPTION(ValueError)
            << "For PagedAttention the kv_head_num should be greater than or equal to 0, but got "
            << kv_head_num << ".";
    }
    if (kv_head_num != 0 && (q_head_num % kv_head_num != 0)) {
        MS_EXCEPTION(ValueError)
            << "For PagedAttention the q_head_num must be divisible by kv_head_num, but got q_head_num="
            << q_head_num << ", kv_head_num=" << kv_head_num << ".";
    }
}

void CheckScaleTypeLogN(int64_t scale_type, int64_t quant_type, int64_t calc_type, int64_t compress_type) {
    if (!(scale_type >= PAScaleType::kPA_SCALE_TYPE_TOR && scale_type < PAScaleType::kPA_SCALE_TYPE_MAX)) {
        MS_EXCEPTION(ValueError) << "For PagedAttention the scale_type is invalid, got " << scale_type << ".";
    }

    if (scale_type == PAScaleType::kPA_SCALE_TYPE_LOGN) {
        if (quant_type != PAQuantType::kPA_TYPE_QUANT_UNQUANT) {
            MS_EXCEPTION(ValueError)
                << "In PA scale type logn mode, quant_type must be 0(TYPE_QUANT_UNQUANT/"
                << "TYPE_QUANT_UNDEFINED), but got " << quant_type << ".";
        }
        if (calc_type != PACalcType::kPA_CALC_TYPE_UNDEFINED) {
            MS_EXCEPTION(ValueError)
                << "In PA scale type logn mode, calc_type feature is not supported, but got calc_type="
                << calc_type << ".";
        }
        if (compress_type != PACompressType::kPA_COMPRESS_TYPE_UNDEFINED) {
            MS_EXCEPTION(ValueError)
                << "In PA scale type logn mode, compress_type feature is not supported, but got compress_type="
                << compress_type << ".";
        }
    }
}

void CheckMLAVHeadSize(int64_t mla_v_head_size) {
    constexpr int64_t kMaxMLAVHeadSize = 576;
    if (!(mla_v_head_size >= 0 && mla_v_head_size <= kMaxMLAVHeadSize)) {
        MS_EXCEPTION(ValueError)
            << "For PagedAttention(MLA mode) the value head size should be in [0, 576], but got "
            << mla_v_head_size << ".";
    }
}

void CheckInputLayoutAndCalcType(int64_t input_layout, int64_t calc_type, int64_t quant_type,
                                 int64_t compress_type, bool batch_run_status_enable) {
    if (!(input_layout == PAInputLayout::kPA_INPUT_LAYOUT_BNSD ||
          input_layout == PAInputLayout::kPA_INPUT_LAYOUT_BSND)) {
        MS_EXCEPTION(ValueError)
            << "For PagedAttention the input layout should be 0(BSND)/1(BNSD), but got "
            << input_layout << ".";
    }

    if (!(calc_type == PACalcType::kPA_CALC_TYPE_SPEC ||
          calc_type == PACalcType::kPA_CALC_TYPE_UNDEFINED)) {
        MS_EXCEPTION(ValueError)
            << "For PagedAttention the calc_type should be 0(disable MTP)/1(enable MTP), but got "
            << calc_type << ".";
    }

    if (calc_type == PACalcType::kPA_CALC_TYPE_SPEC) {
        if (quant_type != PAQuantType::kPA_TYPE_QUANT_UNQUANT) {
            MS_EXCEPTION(ValueError)
                << "In PA MTP scene, quant mode should be "
                << "0(TYPE_QUANT_UNQUANT/TYPE_QUANT_UNDEFINED), but now got "
                << quant_type << ".";
        }
        if (batch_run_status_enable) {
            MS_EXCEPTION(ValueError)
                << "In PA MTP scene, batch_run_status_enable should be false, but now got true.";
        }
        if (compress_type != PACompressType::kPA_COMPRESS_TYPE_UNDEFINED) {
            MS_EXCEPTION(ValueError)
                << "In PA MTP scene, compress_type should be 0(kPA_COMPRESS_TYPE_UNDEFINED), but now got "
                << compress_type << ".";
        }
    }
}

void CheckBNSDLayout(int64_t input_layout, int64_t calc_type, int64_t compress_type,
                     int64_t quant_type, int64_t scale_type) {
    if (input_layout == PAInputLayout::kPA_INPUT_LAYOUT_BNSD) {
        if (calc_type != PACalcType::kPA_CALC_TYPE_UNDEFINED) {
            MS_EXCEPTION(ValueError)
                << "For PagedAttention when input layout is BNSD, calc_type feature is not supported,"
                << " but got " << calc_type << ".";
        }
        if (compress_type != PACompressType::kPA_COMPRESS_TYPE_UNDEFINED) {
            MS_EXCEPTION(ValueError)
                << "For PagedAttention when input layout is BNSD, compress_type feature is not supported,"
                << " but got " << compress_type << ".";
        }
        if (quant_type != PAQuantType::kPA_TYPE_QUANT_UNQUANT) {
            MS_EXCEPTION(ValueError)
                << "For PagedAttention when input layout is BNSD, quant_type must be 0(TYPE_QUANT_UNQUANT),"
                << " but got " << quant_type << ".";
        }
        if (scale_type != PAScaleType::kPA_SCALE_TYPE_TOR) {
            MS_EXCEPTION(ValueError)
                << "For PagedAttention when input layout is BNSD, scale_type must be 0(kPA_SCALE_TYPE_TOR),"
                << " but got " << scale_type << ".";
        }
    }
}

void CheckCompressType(int64_t compress_type, int64_t quant_type,
                       bool batch_run_status_enable, int64_t mask_type) {
    if (compress_type == PACompressType::kPA_COMPRESS_TYPE_MAX) {
        MS_EXCEPTION(ValueError)
            << "In PA compress scene, compress type should not be 3(kPA_COMPRESS_TYPE_MAX).";
    }

    if (compress_type == PACompressType::kPA_COMPRESS_TYPE_KVHEAD ||
        compress_type == PACompressType::kPA_COMPRESS_TYPE_KVHEAD_ROPE) {
        if (quant_type != PAQuantType::kPA_TYPE_QUANT_UNQUANT) {
            MS_EXCEPTION(ValueError)
                << "In PA compress scene, quant_type must be 0(TYPE_QUANT_UNQUANT), but now got "
                << quant_type << ".";
        }
    }

    if (compress_type == PACompressType::kPA_COMPRESS_TYPE_KVHEAD_ROPE) {
        if (batch_run_status_enable) {
            MS_EXCEPTION(ValueError)
                << "In PA COMPRESS_TYPE_KVHEAD_ROPE scene, batch_run_status_enable must be false, but now got true.";
        }
        if (mask_type == PAMaskType::kPA_MASK_UNDEFINED) {
            MS_EXCEPTION(ValueError)
                << "In PA COMPRESS_TYPE_KVHEAD_ROPE scene, mask type should not be "
                << "0(PA_MASK_UNDEFINED), but now got " << mask_type << ".";
        }
    }
}

void CheckQuantType(int64_t quant_type, int64_t calc_type, int64_t out_data_type,
                    bool has_quant_offset, int64_t compress_type) {
    if (quant_type == PAQuantType::kPA_TYPE_DEQUANT_FUSION) {
        if (calc_type != PACalcType::kPA_CALC_TYPE_UNDEFINED) {
            MS_EXCEPTION(ValueError)
                << "For PagedAttention when quant_type is DEQUANT_FUSION, calc_type feature is not supported,"
                << " but got " << calc_type << ".";
        }
    }

    if (quant_type == PAQuantType::kPA_TYPE_QUANT_QKV_OFFLINE ||
        quant_type == PAQuantType::kPA_TYPE_QUANT_QKV_ONLINE) {
        if (!(out_data_type == PAOutDataType::kPA_ACL_FLOAT16 ||
              out_data_type == PAOutDataType::kPA_ACL_BF16)) {
            MS_EXCEPTION(ValueError)
                << "In PA full quant scene, out_data_type must be 1(Float16) or 27(BFloat16), but got "
                << out_data_type << ".";
        }
        if (has_quant_offset) {
            MS_EXCEPTION(ValueError)
                << "In PA full quant scene, has_quant_offset must be false, but now got true.";
        }
        if (compress_type != PACompressType::kPA_COMPRESS_TYPE_UNDEFINED) {
            MS_EXCEPTION(ValueError)
                << "In PA full quant scene, compress_type should be 0(kPA_COMPRESS_TYPE_UNDEFINED),"
                << " but now got " << compress_type << ".";
        }
        if (calc_type != PACalcType::kPA_CALC_TYPE_UNDEFINED) {
            MS_EXCEPTION(ValueError)
                << "In PA full quant scene, calc_type feature is not supported, but got "
                << calc_type << ".";
        }
    }
}

void CheckMLAMode(int64_t mla_v_head_size, int64_t mask_type, int64_t compress_type,
                  int64_t quant_type, int64_t scale_type, int64_t input_layout,
                  int64_t kv_head_num, int64_t calc_type, bool batch_run_status_enable) {
    if (mla_v_head_size > 0) {
        if (mask_type == PAMaskType::kPA_MASK_TYPE_ALIBI) {
            MS_EXCEPTION(ValueError)
                << "In PA MLA mode, mask type kPA_MASK_TYPE_ALIBI is not supported, but now got "
                << mask_type << ".";
        }
        if (compress_type != PACompressType::kPA_COMPRESS_TYPE_UNDEFINED) {
            MS_EXCEPTION(ValueError)
                << "In PA MLA mode, compress_type should be 0(kPA_COMPRESS_TYPE_UNDEFINED), but now got "
                << compress_type << ".";
        }
        if (quant_type == PAQuantType::kPA_TYPE_DEQUANT_FUSION) {
            MS_EXCEPTION(ValueError)
                << "In PA MLA mode, quant_type kPA_TYPE_DEQUANT_FUSION is not supported, but now got "
                << quant_type << ".";
        }
        if (scale_type != PAScaleType::kPA_SCALE_TYPE_TOR) {
            MS_EXCEPTION(ValueError)
                << "In PA MLA mode, scale_type must be 0(kPA_SCALE_TYPE_TOR), but now got "
                << scale_type << ".";
        }
        if (input_layout != PAInputLayout::kPA_INPUT_LAYOUT_BSND) {
            MS_EXCEPTION(ValueError)
                << "In PA MLA mode, input layout must be 0(BSND), but now got "
                << input_layout << ".";
        }
        if (kv_head_num != 1) {
            MS_EXCEPTION(ValueError)
                << "In PA MLA mode, kv_head_num should be 1 (MQA), but now got "
                << kv_head_num << ".";
        }

        if (calc_type == PACalcType::kPA_CALC_TYPE_SPEC) {
            if (mask_type == PAMaskType::kPA_MASK_TYPE_NORM) {
                MS_EXCEPTION(ValueError)
                    << "In PA MLA MTP scene, mask type kPA_MASK_TYPE_NORM is not supported, but now got "
                    << mask_type << ".";
            }
            if (quant_type != PAQuantType::kPA_TYPE_QUANT_UNQUANT) {
                MS_EXCEPTION(ValueError)
                    << "In PA MLA MTP scene, quant_type must be 0(TYPE_QUANT_UNQUANT), but now got "
                    << quant_type << ".";
            }
            if (batch_run_status_enable) {
                MS_EXCEPTION(ValueError)
                    << "In PA MLA MTP scene, batch_run_status_enable should be false, but now got true.";
            }
        }
    }
}

void CheckParamsPynative(int64_t q_head_num, int64_t kv_head_num, int64_t scale_type,
                         int64_t quant_type, int64_t mla_v_head_size, int64_t input_layout,
                         int64_t calc_type, int64_t compress_type, int64_t mask_type,
                         bool batch_run_status_enable, bool has_quant_offset,
                         int64_t out_data_type) {
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

// ------------------------ Shape checks -----------------------------------------------------------

void CheckKVCacheConsistency(int64_t mla_v_dim,
                             int64_t num_blocks_k, int64_t num_blocks_v,
                             int64_t block_size_k, int64_t block_size_v,
                             int64_t head_num_k, int64_t head_num_v) {
    if (mla_v_dim == 0) {
        if (num_blocks_k != num_blocks_v) {
            MS_EXCEPTION(ValueError)
                << "For PagedAttention the num_blocks of key_cache and value_cache must be same, but got "
                << num_blocks_k << " and " << num_blocks_v << ".";
        }
        if (block_size_k != block_size_v) {
            MS_EXCEPTION(ValueError)
                << "For PagedAttention the block_size of key_cache and value_cache must be same, but got "
                << block_size_k << " and " << block_size_v << ".";
        }
        if (head_num_k != head_num_v) {
            MS_EXCEPTION(ValueError)
                << "For PagedAttention the head_num of key_cache and value_cache must be same, but got "
                << head_num_k << " and " << head_num_v << ".";
        }
    }
}

void CheckQueryKVHeadSizeConsistency(int64_t head_size_k, int64_t head_size_q) {
    if (head_size_k != head_size_q) {
        MS_EXCEPTION(ValueError)
            << "For PagedAttention the head_size of key_cache and query must be same, but got "
            << head_size_k << " and " << head_size_q << ".";
    }
}

void CheckNonMLAHeadSizeLimits(int64_t mla_v_dim, int64_t head_size_k, int64_t block_size_k) {
    constexpr int64_t kMaxHeadSize910B = 256;
    constexpr int64_t kMaxHeadSizeProd = 128 * 128;

    if (mla_v_dim == 0) {
        if (!(head_size_k > 0 && head_size_k <= kMaxHeadSize910B)) {
            MS_EXCEPTION(ValueError)
                << "For PagedAttention on ND path, head_size of key_cache must be in (0, 256], but got "
                << head_size_k << ".";
        }
        if (block_size_k * head_size_k > kMaxHeadSizeProd) {
            MS_EXCEPTION(ValueError)
                << "For PagedAttention on ND path, block_size * head_size must be <= 128 * 128, but got "
                << (block_size_k * head_size_k) << ".";
        }
    }
}

void CheckMLAHeadSizeConstraints(int64_t mla_v_dim, int64_t head_size_k,
                                 int64_t head_size_v, int64_t block_size_k) {
    if (mla_v_dim > 0) {
        constexpr int64_t kMaxMLAHeadSize = 576;
        if (!(head_size_k <= kMaxMLAHeadSize && head_size_v <= kMaxMLAHeadSize)) {
            MS_EXCEPTION(ValueError)
                << "For PagedAttention MLA mode, head_size of key_cache and value_cache must be <= 576, but got "
                << head_size_k << " and " << head_size_v << ".";
        }

        constexpr int64_t kHeadSizeThreshold = 256;
        constexpr int64_t kBlockSizeLimit = 128;
        if ((head_size_k > kHeadSizeThreshold || head_size_v > kHeadSizeThreshold) &&
            block_size_k > kBlockSizeLimit) {
            MS_EXCEPTION(ValueError)
                << "For PagedAttention MLA mode, when head_size > 256, block_size must be <= 128, but got "
                << "head_size_k=" << head_size_k << ", head_size_v=" << head_size_v
                << ", block_size=" << block_size_k << ".";
        }
    }
}

void CheckKVCacheShapeND(const ms::Tensor &query, const ms::Tensor &key_cache,
                         const ms::Tensor &value_cache, int64_t mla_v_dim) {
    const auto &query_shape = query.shape();
    const auto &key_cache_shape = key_cache.shape();
    const auto &value_cache_shape = value_cache.shape();

    if (query_shape.size() != kPAQShapeRank ||
        key_cache_shape.size() != kPAKVCacheRank ||
        value_cache_shape.size() != kPAKVCacheRank) {
        // Rank mismatch will be reported by basic rank checks.
        return;
    }

    const auto num_blocks_k = key_cache_shape[0];
    const auto block_size_k = key_cache_shape[1];
    const auto head_num_k = key_cache_shape[2];
    const auto head_size_k = key_cache_shape[3];

    const auto num_blocks_v = value_cache_shape[0];
    const auto block_size_v = value_cache_shape[1];
    const auto head_num_v = value_cache_shape[2];
    const auto head_size_v = value_cache_shape[3];

    const auto head_size_q = query_shape[2];

    CheckKVCacheConsistency(mla_v_dim, num_blocks_k, num_blocks_v,
                            block_size_k, block_size_v, head_num_k, head_num_v);
    CheckQueryKVHeadSizeConsistency(head_size_k, head_size_q);
    CheckNonMLAHeadSizeLimits(mla_v_dim, head_size_k, block_size_k);
    CheckMLAHeadSizeConstraints(mla_v_dim, head_size_k, head_size_v, block_size_k);
}

void CheckMaskFreeShape310P(const std::optional<ms::Tensor> &attn_mask,
                            int64_t mask_type, int64_t input_format) {
    if (mask_type != PAMaskType::kPA_MASK_TYPE_MASK_FREE) {
        return;
    }

    // MASK_FREE only supports 310P (NZ format).
    if (input_format != PAInputFormat::kKVFormatNZ) {
        MS_EXCEPTION(ValueError)
            << "For PagedAttention, MASK_FREE is only supported on 310P (NZ format).";
    }

    if (!attn_mask.has_value()) {
        MS_EXCEPTION(ValueError)
            << "For PagedAttention MASK_FREE on 310P, attn_mask tensor must be provided.";
    }

    const auto &mask_shape = attn_mask.value().shape();
    constexpr size_t kMaskRank = 3;
    if (mask_shape.size() != kMaskRank) {
        MS_EXCEPTION(ValueError)
            << "For PagedAttention MASK_FREE on 310P, the rank of mask must be 3, but got rank: "
            << mask_shape.size() << ".";
    }

    constexpr int64_t kExpectedBatch = 1;
    constexpr int64_t kExpectedBlockSize = 128;

    if (mask_shape[0] != kExpectedBatch) {
        MS_EXCEPTION(ValueError)
            << "For PagedAttention MASK_FREE on 310P, mask dim[0] must be 1, but got "
            << mask_shape[0] << ".";
    }
    if (mask_shape[1] != kExpectedBlockSize) {
        MS_EXCEPTION(ValueError)
            << "For PagedAttention MASK_FREE on 310P, mask dim[1] must be 128, but got "
            << mask_shape[1] << ".";
    }
    if (mask_shape[2] != kExpectedBlockSize) {
        MS_EXCEPTION(ValueError)
            << "For PagedAttention MASK_FREE on 310P, mask dim[2] must be 128, but got "
            << mask_shape[2] << ".";
    }
}

void CheckShapePynative(const ms::Tensor &query, const ms::Tensor &key_cache,
                        const ms::Tensor &value_cache, const ms::Tensor &block_tables,
                        const ms::Tensor &context_lens,
                        const std::optional<ms::Tensor> &attn_mask,
                        int64_t input_format, int64_t mla_v_dim,
                        int64_t mask_type) {
    const auto &query_shape = query.shape();
    const auto &key_cache_shape = key_cache.shape();
    const auto &value_cache_shape = value_cache.shape();
    const auto &block_tables_shape = block_tables.shape();
    const auto &context_len_shape = context_lens.shape();

    // Query rank check
    const auto q_rank = static_cast<int64_t>(query_shape.size());
    constexpr int64_t kPA310PQueryRank = 2;
    if (input_format == PAInputFormat::kKVFormatNZ) {
        if (q_rank != kPA310PQueryRank) {
            MS_EXCEPTION(ValueError)
                << "For PagedAttention (310P/NZ) the rank of query must be 2 (T, H*Dh), but got "
                << q_rank << ".";
        }
    } else {
        if (q_rank != kPAQShapeRank) {
            MS_EXCEPTION(ValueError)
                << "For PagedAttention the rank of query must be " << kPAQShapeRank
                << ", but got " << q_rank << ".";
        }
    }

    // Key cache rank check
    {
        const auto expected_rank = (input_format == PAInputFormat::kKVFormatNZ
                                    ? kPAKVCacheRankAltas
                                    : kPAKVCacheRank);
        if (static_cast<int64_t>(key_cache_shape.size()) != expected_rank) {
            MS_EXCEPTION(ValueError)
                << "For PagedAttention the rank of key_cache must be " << expected_rank
                << " (3 for 310P/NZ, 4 for 910B/ND), but got "
                << key_cache_shape.size() << ".";
        }
    }

    // Value cache rank check
    {
        const auto expected_rank = (input_format == PAInputFormat::kKVFormatNZ
                                    ? kPAKVCacheRankAltas
                                    : kPAKVCacheRank);
        if (static_cast<int64_t>(value_cache_shape.size()) != expected_rank) {
            MS_EXCEPTION(ValueError)
                << "For PagedAttention the rank of value_cache must be " << expected_rank
                << " (3 for 310P/NZ, 4 for 910B/ND), but got "
                << value_cache_shape.size() << ".";
        }
    }

    // Block tables rank check
    if (block_tables_shape.size() != kPABlockTableRank) {
        MS_EXCEPTION(ValueError)
            << "For PagedAttention the rank of block table must be " << kPABlockTableRank
            << ", but got " << block_tables_shape.size() << ".";
    }

    // Context lens rank check
    if (context_len_shape.size() != kPAContextLenRank) {
        MS_EXCEPTION(ValueError)
            << "For PagedAttention the rank of context len must be " << kPAContextLenRank
            << ", but got " << context_len_shape.size() << ".";
    }

    // ND path: extra KV cache / MLA-specific checks.
    if (input_format == PAInputFormat::kKVFormatND) {
        CheckKVCacheShapeND(query, key_cache, value_cache, mla_v_dim);
    }

    // 310P/NZ MASK_FREE shape checks.
    CheckMaskFreeShape310P(attn_mask, mask_type, input_format);
}

// ------------------------ Type checks ----------------------------------------------------------------

void CheckTypePynative(const ms::Tensor &query, const ms::Tensor &key_cache,
                       const ms::Tensor &value_cache, int64_t quant_type) {
    const auto query_dtype = query.data_type();
    const auto key_cache_dtype = key_cache.data_type();
    const auto value_cache_dtype = value_cache.data_type();

    if (quant_type == PAQuantType::kPA_TYPE_QUANT_UNQUANT) {
        if (!(query_dtype == kNumberTypeFloat16 || query_dtype == kNumberTypeBFloat16)) {
            MS_EXCEPTION(ValueError)
                << "For PagedAttention in unquant mode, query dtype must be float16/bfloat16, but got type: "
                << query_dtype << ".";
        }
        if (!(key_cache_dtype == kNumberTypeFloat16 || key_cache_dtype == kNumberTypeBFloat16)) {
            MS_EXCEPTION(ValueError)
                << "For PagedAttention in unquant mode, key cache dtype must be float16/bfloat16, but got type: "
                << key_cache_dtype << ".";
        }
        if (!(value_cache_dtype == kNumberTypeFloat16 || value_cache_dtype == kNumberTypeBFloat16)) {
            MS_EXCEPTION(ValueError)
                << "For PagedAttention in unquant mode, value cache dtype must be float16/bfloat16, but got type: "
                << value_cache_dtype << ".";
        }
    } else if (quant_type == PAQuantType::kPA_TYPE_DEQUANT_FUSION) {
        if (!(query_dtype == kNumberTypeFloat16 || query_dtype == kNumberTypeBFloat16)) {
            MS_EXCEPTION(ValueError)
                << "For PagedAttention in dequant mode, query dtype must be float16/bfloat16, but got type: "
                << query_dtype << ".";
        }
        if (key_cache_dtype != kNumberTypeInt8) {
            MS_EXCEPTION(ValueError)
                << "For PagedAttention in dequant mode, key cache dtype must be int8, but got type: "
                << key_cache_dtype << ".";
        }
        if (value_cache_dtype != kNumberTypeInt8) {
            MS_EXCEPTION(ValueError)
                << "For PagedAttention in dequant mode, value cache dtype must be int8, but got type: "
                << value_cache_dtype << ".";
        }
    } else if (quant_type == PAQuantType::kPA_TYPE_QUANT_QKV_OFFLINE ||
               quant_type == PAQuantType::kPA_TYPE_QUANT_QKV_ONLINE) {
        if (query_dtype != kNumberTypeInt8) {
            MS_EXCEPTION(ValueError)
                << "For PagedAttention in full quant mode, query dtype must be int8, but got type: "
                << query_dtype << ".";
        }
        if (key_cache_dtype != kNumberTypeInt8) {
            MS_EXCEPTION(ValueError)
                << "For PagedAttention in full quant mode, key cache dtype must be int8, but got type: "
                << key_cache_dtype << ".";
        }
        if (value_cache_dtype != kNumberTypeInt8) {
            MS_EXCEPTION(ValueError)
                << "For PagedAttention in full quant mode, value cache dtype must be int8, but got type: "
                << value_cache_dtype << ".";
        }
    }
}

}  // namespace

class PagedAttentionRunner : public InternalPyboostRunner {
 public:
    explicit PagedAttentionRunner(const std::string &op_name) : InternalPyboostRunner(op_name) {}
    ~PagedAttentionRunner() = default;

    void SetParam(int32_t q_head_num, float qk_scale, int32_t kv_head_num, int32_t mask_type,
            bool batch_run_status_enable, int32_t quant_type, int32_t out_data_type, bool has_quant_offset,
            int32_t compress_type, int32_t calc_type, int32_t scale_type, int32_t input_layout, uint32_t mla_v_dim,
            const std::vector<int32_t> &q_seq_len, const std::vector<int32_t> &kv_seq_len,
            std::vector<int32_t> &batch_run_status) {
        param_.q_head_num = q_head_num;
        param_.tor = qk_scale;
        param_.kv_head_num = kv_head_num;
        param_.batch_run_status_enable = batch_run_status_enable;
        param_.has_quant_offset = has_quant_offset;
        param_.mla_v_dim = mla_v_dim;
        param_.mask_type = static_cast<internal_v2::CustomASDPagedAttentionParam::MaskType>(mask_type);
        param_.quant_type = static_cast<internal_v2::CustomASDPagedAttentionParam::QuantType>(quant_type);
        param_.out_data_type = static_cast<internal_v2::CustomASDPagedAttentionParam::OutDataType>(out_data_type);
        param_.compress_type = static_cast<internal_v2::CustomASDPagedAttentionParam::CompressType>(compress_type);
        param_.calc_type = static_cast<internal_v2::CustomASDPagedAttentionParam::CaclType>(calc_type);
        param_.scale_type = static_cast<internal_v2::CustomASDPagedAttentionParam::ScaleType>(scale_type);
        param_.input_layout = static_cast<internal_v2::CustomASDPagedAttentionParam::InputLayout>(input_layout);
        auto is_q_changed = CheckAndUpdate(q_seq_len, &(param_.q_seq_len));
        auto is_kv_changed = CheckAndUpdate(kv_seq_len, &(param_.kv_seq_len));
        (void)CheckAndUpdate(batch_run_status, &(param_.batch_run_status));

        need_update_param_ = is_q_changed | is_kv_changed;
    }

    void SetInputFormat(PAInputFormat input_format) { input_format_ = input_format; }

 protected:
    bool UpdateParam() override {
        if (created_flag_) {
            // the q_seq_len and kv_seq_len are inited in CreatedKernel, so there is no need to load them again
            created_flag_ = false;
            return true;
        }

        if (need_update_param_) {
            auto ret = internal_op_->UpdateParam(&param_);
            if (ret != internal_v2::kInternalOk) {
                MS_LOG(ERROR) << "ASD PagedAttention UpdateParam failed in MlaRunner.";
                return false;
            }
            return true;
        }
        return true;
    }

    internal_v2::InternalOpPtr CreateKernel(const internal_v2::InputsImmutableInfoList &inputs,
                                            const internal_v2::OutputsImmutableInfoList &outputs) override {
        created_flag_ = true;
        // NZ format routing for 310P platform when input_format == 1
        if (input_format_ == PAInputFormat::kKVFormatNZ) {
            auto inputs_new = inputs;
            auto outputs_new = outputs;
            // Set Query, Key Cache, Value Cache and Mask to FRACTAL_NZ format
            inputs_new[kPagedAttentionInputQueryIndex].SetFormat(internal_v2::kFormatFRACTAL_NZ);
            inputs_new[kPagedAttentionInputKeyCacheIndex].SetFormat(internal_v2::kFormatFRACTAL_NZ);
            inputs_new[kPagedAttentionInputValueCacheIndex].SetFormat(internal_v2::kFormatFRACTAL_NZ);
            inputs_new[kPagedAttentionInputAttnMaskIndex].SetFormat(internal_v2::kFormatFRACTAL_NZ);
            // Set output to FRACTAL_NZ format
            outputs_new[kPagedAttentionOutputIndex].SetFormat(internal_v2::kFormatFRACTAL_NZ);
            return internal_v2::CreateCustomPagedAttentionOp(inputs_new, outputs_new, param_,
                                                             internal_v2::kInternalCustomPagedAttention);
        }
        return internal_v2::CreateCustomPagedAttentionOp(inputs, outputs, param_,
                                                        internal_v2::kInternalCustomPagedAttention);
    }

 private:
    internal_v2::CustomASDPagedAttentionParam param_;
    bool created_flag_{true};
    bool need_update_param_{false};
    PAInputFormat input_format_{kKVFormatND};
};

std::vector<ms::Tensor> paged_attention_atb(
        const ms::Tensor &query, const ms::Tensor &key_cache, const ms::Tensor &value_cache,
        const ms::Tensor &block_tables, const ms::Tensor &context_lens, const std::optional<ms::Tensor> &attn_mask,
        const std::optional<ms::Tensor> &batch_run_status, const std::optional<ms::Tensor> &k_descale,
        const std::optional<ms::Tensor> &k_offset, const std::optional<ms::Tensor> &v_descale,
        const std::optional<ms::Tensor> &v_offset, const std::optional<ms::Tensor> &razor_offset,
        const std::optional<ms::Tensor> &p_scale, const std::optional<ms::Tensor> &log_n,
        const std::optional<ms::Tensor> &q_seq_lens, int64_t q_head_num, double qk_scale, int64_t kv_head_num,
        int64_t mask_type, bool batch_run_status_enable, int64_t quant_type, int64_t out_data_type,
        bool has_quant_offset, int64_t compress_type, int64_t calc_type, int64_t scale_type, int64_t input_layout,
        int64_t mla_v_dim, int64_t input_format) {
    static auto op_name = "PagedAttention";
    auto runner = std::make_shared<PagedAttentionRunner>(op_name);
    MS_EXCEPTION_IF_NULL(runner);

    if (input_format != PAInputFormat::kKVFormatND && input_format != PAInputFormat::kKVFormatNZ) {
        MS_LOG(EXCEPTION) << "For " << op_name << ", the input_format is invalid: " << input_format;
    }

    // Align PyNative checks with graph-mode logic.
    CheckParamsPynative(q_head_num, kv_head_num, scale_type, quant_type, mla_v_dim,
                        input_layout, calc_type, compress_type, mask_type,
                        batch_run_status_enable, has_quant_offset, out_data_type);
    CheckShapePynative(query, key_cache, value_cache, block_tables, context_lens,
                       attn_mask, input_format, mla_v_dim, mask_type);
    CheckTypePynative(query, key_cache, value_cache, quant_type);

    // q_seq_lens: always passed through param (move to CPU) for both 910B and 310P
    std::vector<int32_t> q_seq_lens_value;
    constexpr int64_t kPACalcTypeSpec = 1;
    if (calc_type == kPACalcTypeSpec) {
        if (!q_seq_lens.has_value()) {
            MS_LOG(EXCEPTION) << "For " << op_name << ", calc_type is SPEC(MTP), q_seq_lens must be provided.";
        }
        q_seq_lens_value = GetValueFromTensor<std::vector<int32_t>>(q_seq_lens.value(), op_name, "q_seq_lens");
    }

    // - 910B (ND format): GetValueFromTensor -> passed through param
    // - 310P (NZ format): keep as NPU tensor, DON'T read value
    std::vector<int32_t> context_lens_value;
    if (input_format == PAInputFormat::kKVFormatND) {
        // 910B: safe to read context_lens from CPU tensor
        context_lens_value = GetValueFromTensor<std::vector<int32_t>>(context_lens, op_name, "context_lens");
    }

    std::vector<int32_t> batch_run_status_value;
    if (batch_run_status_enable && batch_run_status.has_value()) {
        batch_run_status_value =
            GetValueFromTensor<std::vector<int32_t>>(batch_run_status.value(), op_name, "batch_run_status");
    }
    runner->SetInputFormat(static_cast<PAInputFormat>(input_format));
    runner->SetParam(static_cast<int32_t>(q_head_num), static_cast<float>(qk_scale), static_cast<int32_t>(kv_head_num),
                     static_cast<int32_t>(mask_type), batch_run_status_enable, static_cast<int32_t>(quant_type),
                     static_cast<int32_t>(out_data_type), has_quant_offset, static_cast<int32_t>(compress_type),
                     static_cast<int32_t>(calc_type), static_cast<int32_t>(scale_type),
                     static_cast<int32_t>(input_layout), static_cast<uint32_t>(mla_v_dim), q_seq_lens_value,
                     context_lens_value, batch_run_status_value);

    runner->Setup(op_name, query, key_cache, value_cache, block_tables, context_lens, attn_mask, batch_run_status,
            k_descale, k_offset, v_descale, v_offset, razor_offset, p_scale, log_n, q_seq_lens, q_head_num,
            qk_scale, kv_head_num, mask_type, batch_run_status_enable, quant_type, out_data_type,
            has_quant_offset, compress_type, calc_type, scale_type, input_layout, mla_v_dim, input_format);

    auto output_data_type = query.data_type();
    if (query.data_type() == kNumberTypeInt8 && out_data_type != PAOutDataType::kPA_ACL_DT_UNDEFINED) {
        if (out_data_type == PAOutDataType::kPA_ACL_FLOAT16) {
            output_data_type = kNumberTypeFloat16;
        } else if (out_data_type == PAOutDataType::kPA_ACL_BF16) {
            output_data_type = kNumberTypeBFloat16;
        }
    }
    auto attn_out = ms::Tensor(output_data_type, query.shape());

    // Construct inputs: include context_lens as tensor input for 310P
    // On 910B, it will be mapped to empty in ms_kernels_internal layer
    std::vector<ms::Tensor> inputs = {query,
                                      key_cache,
                                      value_cache,
                                      block_tables,
                                      context_lens,  // Add context_lens as NPU tensor input
                                      GetTensorOrEmpty(attn_mask),
                                      GetTensorOrEmpty(k_descale),
                                      GetTensorOrEmpty(k_offset),
                                      GetTensorOrEmpty(v_descale),
                                      GetTensorOrEmpty(v_offset),
                                      GetTensorOrEmpty(razor_offset),
                                      GetTensorOrEmpty(p_scale),
                                      GetTensorOrEmpty(log_n)};
    std::vector<ms::Tensor> outputs = {attn_out};
    runner->GetOrCreateKernel(inputs, outputs);
    runner->Run(inputs, outputs);
    return outputs;
}

auto pyboost_paged_attention(const ms::Tensor &query, const ms::Tensor &key_cache, const ms::Tensor &value_cache,
        const ms::Tensor &block_tables, const ms::Tensor &context_lens,
        const std::optional<ms::Tensor> &attn_mask,
        const std::optional<ms::Tensor> &batch_run_status,
        const std::optional<ms::Tensor> &k_descale, const std::optional<ms::Tensor> &k_offset,
        const std::optional<ms::Tensor> &v_descale, const std::optional<ms::Tensor> &v_offset,
        const std::optional<ms::Tensor> &razor_offset, const std::optional<ms::Tensor> &p_scale,
        const std::optional<ms::Tensor> &log_n, const std::optional<ms::Tensor> &q_seq_lens,
        int64_t q_head_num, double qk_scale, int64_t kv_head_num, int64_t mask_type,
        bool batch_run_status_enable, int64_t quant_type, int64_t out_data_type,
        bool has_quant_offset, int64_t compress_type, int64_t calc_type, int64_t scale_type,
        int64_t input_layout, int64_t mla_v_dim, int64_t input_format) {
    return ms::pynative::PyboostRunner::Call<PAOutputIndex::kPagedAttentionOutputNum>(
        paged_attention_atb, query, key_cache, value_cache, block_tables, context_lens, attn_mask, batch_run_status,
        k_descale, k_offset, v_descale, v_offset, razor_offset, p_scale, log_n, q_seq_lens, q_head_num, qk_scale,
        kv_head_num, mask_type, batch_run_status_enable, quant_type, out_data_type, has_quant_offset, compress_type,
        calc_type, scale_type, input_layout, mla_v_dim, input_format);
}
}  // namespace ms_custom_ops

MS_CUSTOM_OPS_EXTENSION_MODULE(m) {
    m.def("paged_attention", &ms_custom_ops::pyboost_paged_attention, "PagedAttention", pybind11::arg("query"),
        pybind11::arg("key_cache"), pybind11::arg("value_cache"), pybind11::arg("block_tables"),
        pybind11::arg("context_lens"), pybind11::arg("attn_mask") = std::nullopt,
        pybind11::arg("batch_run_status") = std::nullopt, pybind11::arg("k_descale") = std::nullopt,
        pybind11::arg("k_offset") = std::nullopt, pybind11::arg("v_descale") = std::nullopt,
        pybind11::arg("v_offset") = std::nullopt, pybind11::arg("razor_offset") = std::nullopt,
        pybind11::arg("p_scale") = std::nullopt, pybind11::arg("log_n") = std::nullopt,
        pybind11::arg("q_seq_lens") = std::nullopt, pybind11::arg("q_head_num") = 0, pybind11::arg("qk_scale") = 1.0,
        pybind11::arg("kv_head_num") = 0, pybind11::arg("mask_type") = 0,
        pybind11::arg("batch_run_status_enable") = false, pybind11::arg("quant_type") = 0,
        pybind11::arg("out_data_type") = -1, pybind11::arg("has_quant_offset") = false,
        pybind11::arg("compress_type") = 0, pybind11::arg("calc_type") = 0, pybind11::arg("scale_type") = 0,
        pybind11::arg("input_layout") = 0, pybind11::arg("mla_v_dim") = 0, pybind11::arg("input_format") = 0);
}
