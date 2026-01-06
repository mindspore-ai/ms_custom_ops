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

#include "ascendc/basic_api/kernel_operator.h"
#include "lib/matmul_intf.h"
#include "asd/unpad_fa/unpad_flash_attention_mix.cce"
#include "asd/unpad_fa/unpad_flashattention_bf16_mix.cce"
__aicore__ inline int64_t GetFftsBaseAddr() { return get_ffts_base_addr(); }

extern "C" __global__ __aicore__ void unpad_fa_npd(GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR attn_mask,
                                                   GM_ADDR q_seq, GM_ADDR kv_seq, GM_ADDR attn_out, GM_ADDR workspace,
                                                   GM_ADDR tiling) {
#if defined(__DAV_C220_CUBE__) || defined(__DAV_C220_VEC__)
  GM_ADDR sync = (__gm__ uint8_t *)GetFftsBaseAddr();
  GM_ADDR q_gm = query;
  GM_ADDR k_gm = key;
  GM_ADDR v_gm = value;
  GM_ADDR layerID_gm = nullptr;
  GM_ADDR mask_gm = attn_mask;
  GM_ADDR alibi_coeff_gm = nullptr;
  GM_ADDR deq_qk_gm = nullptr;
  GM_ADDR off_qk_gm = nullptr;
  GM_ADDR deq_pv_gm = nullptr;
  GM_ADDR off_pv_gm = nullptr;
  GM_ADDR quant_p_gm = nullptr;
  GM_ADDR o_gm = attn_out;
  GM_ADDR tiling_para_gm = tiling;
  uint32_t ws_size = static_cast<uint32_t>(*((__gm__ uint32_t *)tiling_para_gm + 39));
  GM_ADDR s_gm = workspace;
  GM_ADDR p_gm = workspace + ws_size;
  GM_ADDR o_tmp_gm = workspace + 2 * ws_size;
  GM_ADDR upo_tmp_gm = nullptr;
  GM_ADDR logN_gm = nullptr;
#endif
  if (TILING_KEY_IS(0)) {
#ifdef __DAV_C220_CUBE__
    unpda_fa_npd_half::UnpadAttentionDecoderAic<half, half, float, false, false, false> fa_aic_fp16;
    fa_aic_fp16.Run(sync, q_gm, k_gm, v_gm, layerID_gm, mask_gm, alibi_coeff_gm, deq_qk_gm, off_qk_gm, deq_pv_gm,
                    off_pv_gm, quant_p_gm, o_gm, s_gm, p_gm, o_tmp_gm, upo_tmp_gm, tiling_para_gm);
#elif __DAV_C220_VEC__
    unpda_fa_npd_half::UnpadAttentionDecoderAiv<half, half, float, false, false, false> fa_aiv_fp16;
    fa_aiv_fp16.Run(sync, q_gm, k_gm, v_gm, layerID_gm, mask_gm, alibi_coeff_gm, deq_qk_gm, off_qk_gm, deq_pv_gm,
                    off_pv_gm, quant_p_gm, logN_gm, o_gm, s_gm, p_gm, o_tmp_gm, upo_tmp_gm, tiling_para_gm);
#endif
  } else if (TILING_KEY_IS(1)) {
#ifdef __DAV_C220_CUBE__
    unpda_fa_npd_bf16::FlashAttentionEncoderHighPrecision<__bf16> fa_cube(sync, q_gm, k_gm, v_gm, layerID_gm, s_gm,
                                                                          p_gm, o_tmp_gm, tiling_para_gm);
    fa_cube.Run<true>();
#elif __DAV_C220_VEC__
    unpda_fa_npd_bf16::FlashAttentionEncoderHighPrecisionVec<__bf16> fa_vec(
      sync, mask_gm, alibi_coeff_gm, o_gm, s_gm, p_gm, o_tmp_gm, tiling_para_gm, deq_qk_gm, off_qk_gm, quant_p_gm,
      deq_pv_gm, off_pv_gm, logN_gm);
    fa_vec.Run<true>();
#endif
  } else if (TILING_KEY_IS(3)) {
#ifdef __DAV_C220_CUBE__
    unpda_fa_npd_bf16::FlashAttentionEncoderHighPrecisionCubeOpt<__bf16> fa_cube(sync, q_gm, k_gm, v_gm, layerID_gm,
                                                                                 s_gm, p_gm, o_tmp_gm, tiling_para_gm);
    fa_cube.Run();
#elif __DAV_C220_VEC__
    unpda_fa_npd_bf16::FlashAttentionEncoderHighPrecisionVecOpt<float, __bf16, __bf16, __bf16, float,
                                                                unpda_fa_npd_bf16::MaskType::MASK_TYPE_NONE,
                                                                unpda_fa_npd_bf16::ScaleType::SCALE_TOR>
      fa_vec(sync, mask_gm, alibi_coeff_gm, o_gm, s_gm, p_gm, o_tmp_gm, tiling_para_gm, deq_qk_gm, off_qk_gm,
             quant_p_gm, deq_pv_gm, off_pv_gm, logN_gm);
    fa_vec.Run();
#endif
  }
}
