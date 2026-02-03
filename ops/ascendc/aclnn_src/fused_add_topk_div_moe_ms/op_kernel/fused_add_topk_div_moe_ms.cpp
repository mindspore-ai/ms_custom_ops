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
#include "kernel_operator.h"  // NOLINT(build/include_subdir)

// NOLINTBEGIN
class KernelFusedAddTopkDivMoeMS {
public:
  __aicore__ inline KernelFusedAddTopkDivMoeMS() {}
  __aicore__ inline void Init(GM_ADDR logits, GM_ADDR bias, GM_ADDR expert_weight, GM_ADDR expert_index,
                              FusedAddTopkDivMoeMSTilingData *tiling) {
    this->logits = logits;
    this->bias = bias;
    this->expert_weight = expert_weight;
    this->expert_index = expert_index;
    this->num_tokens = tiling->num_tokens;
  }
  __aicore__ inline void Process() {
    fused_add_topk_div_moe_0_kernel(this->logits, this->bias, this->num_tokens, this->expert_weight,
                                    this->expert_index);
  }

private:
__aicore__ void fused_add_topk_div_moe_0_kernel(__gm__ uint8_t *__restrict__ logits, __gm__ uint8_t *__restrict__ bias, int32_t num_tokens, __gm__ uint8_t *__restrict__ expert_weight, __gm__ uint8_t *__restrict__ expert_index) {
  AscendC::TPipe pipe;
  AscendC::TBuf<AscendC::TPosition::VECIN> vecin_buff;
  pipe.InitBuffer(vecin_buff, 262144);
  AscendC::TBuf<AscendC::TPosition::GM> gm_buff;
  pipe.InitBuffer(gm_buff, 4294967295);
  AscendC::TBuf<AscendC::TPosition::VECOUT> vecout_buff;
  pipe.InitBuffer(vecout_buff, 262144);
  AscendC::TBuf<AscendC::TPosition::CO2> co2_buff;
  pipe.InitBuffer(co2_buff, 262144);
  AscendC::TBuf<AscendC::TPosition::CO1> co1_buff;
  pipe.InitBuffer(co1_buff, 262144);
  AscendC::TBuf<AscendC::TPosition::A2> a2_buff;
  pipe.InitBuffer(a2_buff, 65536);
  AscendC::TBuf<AscendC::TPosition::A1> a1_buff;
  pipe.InitBuffer(a1_buff, 1048576);
  AscendC::TBuf<AscendC::TPosition::B2> b2_buff;
  pipe.InitBuffer(b2_buff, 65536);
  AscendC::TBuf<AscendC::TPosition::B1> b1_buff;
  pipe.InitBuffer(b1_buff, 1048576);
  AscendC::DataCopyEnhancedParams enhanceParams_relu({AscendC::BlockMode::BLOCK_MODE_MATRIX,AscendC::DeqScale::DEQ_NONE, 0, 0, true, pad_t::PAD_NONE, 0}); 
  AscendC::DataCopyEnhancedParams enhanceParams({AscendC::BlockMode::BLOCK_MODE_MATRIX,AscendC::DeqScale::DEQ_NONE, 0, 0, false, pad_t::PAD_NONE, 0}); 
  uint8_t padList[4] = {0, 0, 0, 0}; 
  AscendC::GlobalTensor<float>bias_ascendc;
  AscendC::GlobalTensor<float>logits_ascendc;
  AscendC::GlobalTensor<float>expert_weight_ascendc;
  AscendC::GlobalTensor<int32_t>expert_index_ascendc;
  AscendC::LocalTensor<float>current_extra_score_10_ascendc;
  int32_t res_3;
  int32_t res_14;
  int32_t res_16;
  int32_t percore_size_18;
  int32_t res_20;
  int32_t block_idx_0;
  int32_t res_22;
  int32_t res_12;
  int32_t percore_size_24;
  int32_t res_25;
  int32_t token_num_6;
  bool res_26;
  int32_t res_27;
  bool res_28;
  int32_t res_29;
  int32_t res_30;
  int32_t res_35;
  AscendC::LocalTensor<int32_t>ub_range_37_ascendc;
  int32_t res_41;
  AscendC::LocalTensor<float>current_score_50_ascendc;
  int32_t res_38;
  int32_t res_39;
  int32_t res_45;
  int32_t res_46;
  __ubuf__ int32_t * ub_range_44 = (__ubuf__ int32_t *)((uintptr_t)(1536));
  __ubuf__ int32_t * ub_range_37 = (__ubuf__ int32_t *)((uintptr_t)(1536));
  __ubuf__ int32_t * ub_range_53 = (__ubuf__ int32_t *)((uintptr_t)(1536));
  __ubuf__ int32_t * ub_range_57 = (__ubuf__ int32_t *)((uintptr_t)(1536));
  __ubuf__ int32_t * ub_range_62 = (__ubuf__ int32_t *)((uintptr_t)(1536));
  __ubuf__ int32_t * ub_range_66 = (__ubuf__ int32_t *)((uintptr_t)(1536));
  __ubuf__ int32_t * ub_range_71 = (__ubuf__ int32_t *)((uintptr_t)(1536));
  __ubuf__ int32_t * ub_range_75 = (__ubuf__ int32_t *)((uintptr_t)(1536));
  __ubuf__ int32_t * ub_range_79 = (__ubuf__ int32_t *)((uintptr_t)(1536));
  AscendC::LocalTensor<int32_t>values_82_ascendc;
  AscendC::LocalTensor<int32_t>ub_range_79_ascendc;
  AscendC::LocalTensor<int32_t>values_84_ascendc;
  AscendC::LocalTensor<int32_t>values_86_ascendc;
  AscendC::LocalTensor<int32_t>values_88_ascendc;
  AscendC::LocalTensor<int32_t>values_90_ascendc;
  AscendC::LocalTensor<int32_t>values_92_ascendc;
  AscendC::LocalTensor<int32_t>values_94_ascendc;
  AscendC::LocalTensor<int32_t>ub_97_ascendc;
  AscendC::LocalTensor<int32_t>values_99_ascendc;
  AscendC::LocalTensor<int32_t>values_96_ascendc;
  AscendC::LocalTensor<int32_t>ub_indices_102_ascendc;
  AscendC::LocalTensor<int32_t>values_101_ascendc;
  AscendC::LocalTensor<float>ub_indices_103_ascendc;
  AscendC::LocalTensor<float>current_score_54_ascendc;
  AscendC::LocalTensor<float>sum_exp_59_ascendc;
  AscendC::LocalTensor<float>ub_58_ascendc;
  float sum_exp_63;
  __ubuf__ float * sum_exp_59 = (__ubuf__ float *)((uintptr_t)(576));
  float res_68;
  AscendC::LocalTensor<float>current_score_72_ascendc;
  __ubuf__ float * current_score_76 = (__ubuf__ float *)((uintptr_t)(512));
  __ubuf__ float * current_score_72 = (__ubuf__ float *)((uintptr_t)(512));
  AscendC::LocalTensor<float>current_score_80_ascendc;
  AscendC::LocalTensor<float>current_score_76_ascendc;
  AscendC::LocalTensor<float>concat_local_104_ascendc;
  AscendC::LocalTensor<float>sorted_local_105_ascendc;
  AscendC::LocalTensor<float>sorted_0_108_ascendc;
  AscendC::LocalTensor<float>sorted_1_111_ascendc;
  AscendC::LocalTensor<float>sorted_2_114_ascendc;
  AscendC::LocalTensor<float>sorted_3_117_ascendc;
  AscendC::LocalTensor<float>sorted_0_120_ascendc;
  AscendC::LocalTensor<float>sorted_1_123_ascendc;
  AscendC::LocalTensor<float>sorted_2_126_ascendc;
  AscendC::LocalTensor<float>sorted_3_129_ascendc;
  AscendC::LocalTensor<float>sort_tmp_local_134_ascendc;
  AscendC::LocalTensor<float>sort_tmp_local_139_ascendc;
  AscendC::LocalTensor<float>sort_tmp_local_142_ascendc;
  AscendC::LocalTensor<float>current_score_143_ascendc;
  AscendC::LocalTensor<float>ub_indices_144_ascendc;
  AscendC::LocalTensor<float>topk_vals_148_ascendc;
  AscendC::LocalTensor<float>sum_val_153_ascendc;
  AscendC::LocalTensor<float>ub_152_ascendc;
  AscendC::LocalTensor<int32_t>ub_indices_145_ascendc;
  AscendC::LocalTensor<int32_t>topk_indices_151_ascendc;
  float sum_val_157;
  __ubuf__ float * sum_val_153 = (__ubuf__ float *)((uintptr_t)(4192));
  float res_159;
  AscendC::LocalTensor<float>topk_vals_160_ascendc;
  current_extra_score_10_ascendc = vecin_buff.GetWithOffset<float>(128, 0);
  bias_ascendc.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(bias));
  AscendC::DataCopy(current_extra_score_10_ascendc, bias_ascendc, {(uint16_t)(1), (uint16_t)(16), (uint16_t)(0), (uint16_t)(0)});
  res_3 = num_tokens + (int32_t)8;
  res_14 = res_3 - (int32_t)1;
  res_16 = res_14 / (int32_t)8;
  percore_size_18 = res_16 + (int32_t)0;
  res_20 = percore_size_18 + (int32_t)7;
  block_idx_0 = (int32_t)block_idx;
  res_22 = res_20 / (int32_t)8;
  res_12 = block_idx_0 + (int32_t)1;
  percore_size_24 = res_22 * (int32_t)8;
  res_25 = res_12 * percore_size_24;
  token_num_6 = (int32_t)0 + (int32_t)0;
  res_26 = res_25 < num_tokens;
  AscendC::PipeBarrier<PIPE_ALL>();
  if (res_26) {
    token_num_6 = percore_size_24;
  } else {
    res_27 = block_idx_0 * percore_size_24;
    res_28 = res_27 < num_tokens;
    AscendC::PipeBarrier<PIPE_ALL>();
    if (res_28) {
      res_29 = block_idx_0 * percore_size_24;
      res_30 = num_tokens - res_29;
      token_num_6 = res_30;
    } else {
      token_num_6 = (int32_t)0;
    }
  }
  AscendC::PipeBarrier<PIPE_ALL>();
  AscendC::PipeBarrier<PIPE_ALL>();
  for (int dynamic_loop_var_0_40 = 0; dynamic_loop_var_0_40 < token_num_6; dynamic_loop_var_0_40 += 1) {
    res_35 = block_idx_0 * percore_size_24;
    ub_range_37_ascendc = vecin_buff.GetWithOffset<int32_t>(16, 1536);
    AscendC::SetMaskNorm();
    AscendC::SetVectorMask<int32_t, AscendC::MaskMode::NORMAL>(0ULL, (1ULL << (16)) - 1);
    AscendC::Duplicate<int32_t, false>(ub_range_37_ascendc, (int32_t)0, AscendC::MASK_PLACEHOLDER, 1, 1, 8);
    AscendC::SetMaskNorm();
    AscendC::SetVectorMask<int32_t, AscendC::MaskMode::NORMAL>(18446744073709551615ULL, 18446744073709551615ULL);
    res_41 = res_35 + dynamic_loop_var_0_40;
    AscendC::SetFlag<AscendC::HardEvent::S_MTE2>(0);
    AscendC::WaitFlag<AscendC::HardEvent::S_MTE2>(0);
    current_score_50_ascendc = vecin_buff.GetWithOffset<float>(128, 512);
    logits_ascendc.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(logits));
    AscendC::DataCopy(current_score_50_ascendc, logits_ascendc[(res_41) * 128], {(uint16_t)(1), (uint16_t)(16), (uint16_t)(0), (uint16_t)(0)});
    res_38 = block_idx_0 * percore_size_24;
    res_39 = block_idx_0 * percore_size_24;
    res_45 = res_38 + dynamic_loop_var_0_40;
    res_46 = res_39 + dynamic_loop_var_0_40;
    AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
    AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
    ub_range_37[0] = 0;
    ub_range_44[1] = 1;
    ub_range_53[2] = 2;
    ub_range_57[3] = 3;
    ub_range_62[4] = 4;
    ub_range_66[5] = 5;
    ub_range_71[6] = 6;
    ub_range_75[7] = 7;
    AscendC::SetFlag<AscendC::HardEvent::S_V>(0);
    AscendC::WaitFlag<AscendC::HardEvent::S_V>(0);
    values_82_ascendc = vecin_buff.GetWithOffset<int32_t>(16, 1600);
    ub_range_79_ascendc = vecin_buff.GetWithOffset<int32_t>(16, 1536);
    AscendC::SetMaskNorm();
    AscendC::SetVectorMask<int32_t, AscendC::MaskMode::NORMAL>(0ULL, (1ULL << (16)) - 1);
    Adds<int32_t, false>(values_82_ascendc, ub_range_79_ascendc, (int32_t)8, AscendC::MASK_PLACEHOLDER, 1, {1, 1, 8, 8});
    AscendC::SetMaskNorm();
    AscendC::SetVectorMask<int32_t, AscendC::MaskMode::NORMAL>(18446744073709551615ULL, 18446744073709551615ULL);
    AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(0);
    AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(0);
    AscendC::PipeBarrier<PIPE_V>();
    values_84_ascendc = vecin_buff.GetWithOffset<int32_t>(16, 1664);
    AscendC::SetMaskNorm();
    AscendC::SetVectorMask<int32_t, AscendC::MaskMode::NORMAL>(0ULL, (1ULL << (16)) - 1);
    Adds<int32_t, false>(values_84_ascendc, values_82_ascendc, (int32_t)8, AscendC::MASK_PLACEHOLDER, 1, {1, 1, 8, 8});
    AscendC::SetMaskNorm();
    AscendC::SetVectorMask<int32_t, AscendC::MaskMode::NORMAL>(18446744073709551615ULL, 18446744073709551615ULL);
    AscendC::PipeBarrier<PIPE_V>();
    values_86_ascendc = vecin_buff.GetWithOffset<int32_t>(16, 1728);
    AscendC::SetMaskNorm();
    AscendC::SetVectorMask<int32_t, AscendC::MaskMode::NORMAL>(0ULL, (1ULL << (16)) - 1);
    Adds<int32_t, false>(values_86_ascendc, values_84_ascendc, (int32_t)8, AscendC::MASK_PLACEHOLDER, 1, {1, 1, 8, 8});
    AscendC::SetMaskNorm();
    AscendC::SetVectorMask<int32_t, AscendC::MaskMode::NORMAL>(18446744073709551615ULL, 18446744073709551615ULL);
    AscendC::PipeBarrier<PIPE_V>();
    values_88_ascendc = vecin_buff.GetWithOffset<int32_t>(16, 1792);
    AscendC::SetMaskNorm();
    AscendC::SetVectorMask<int32_t, AscendC::MaskMode::NORMAL>(0ULL, (1ULL << (16)) - 1);
    Adds<int32_t, false>(values_88_ascendc, values_86_ascendc, (int32_t)8, AscendC::MASK_PLACEHOLDER, 1, {1, 1, 8, 8});
    AscendC::SetMaskNorm();
    AscendC::SetVectorMask<int32_t, AscendC::MaskMode::NORMAL>(18446744073709551615ULL, 18446744073709551615ULL);
    AscendC::PipeBarrier<PIPE_V>();
    values_90_ascendc = vecin_buff.GetWithOffset<int32_t>(16, 1856);
    AscendC::SetMaskNorm();
    AscendC::SetVectorMask<int32_t, AscendC::MaskMode::NORMAL>(0ULL, (1ULL << (16)) - 1);
    Adds<int32_t, false>(values_90_ascendc, values_88_ascendc, (int32_t)8, AscendC::MASK_PLACEHOLDER, 1, {1, 1, 8, 8});
    AscendC::SetMaskNorm();
    AscendC::SetVectorMask<int32_t, AscendC::MaskMode::NORMAL>(18446744073709551615ULL, 18446744073709551615ULL);
    AscendC::PipeBarrier<PIPE_V>();
    values_92_ascendc = vecin_buff.GetWithOffset<int32_t>(16, 1920);
    AscendC::SetMaskNorm();
    AscendC::SetVectorMask<int32_t, AscendC::MaskMode::NORMAL>(0ULL, (1ULL << (16)) - 1);
    Adds<int32_t, false>(values_92_ascendc, values_90_ascendc, (int32_t)8, AscendC::MASK_PLACEHOLDER, 1, {1, 1, 8, 8});
    AscendC::SetMaskNorm();
    AscendC::SetVectorMask<int32_t, AscendC::MaskMode::NORMAL>(18446744073709551615ULL, 18446744073709551615ULL);
    AscendC::PipeBarrier<PIPE_V>();
    values_94_ascendc = vecin_buff.GetWithOffset<int32_t>(16, 1984);
    AscendC::SetMaskNorm();
    AscendC::SetVectorMask<int32_t, AscendC::MaskMode::NORMAL>(0ULL, (1ULL << (16)) - 1);
    Adds<int32_t, false>(values_94_ascendc, values_92_ascendc, (int32_t)8, AscendC::MASK_PLACEHOLDER, 1, {1, 1, 8, 8});
    AscendC::SetMaskNorm();
    AscendC::SetVectorMask<int32_t, AscendC::MaskMode::NORMAL>(18446744073709551615ULL, 18446744073709551615ULL);
    AscendC::PipeBarrier<PIPE_V>();
    ub_97_ascendc = vecout_buff.GetWithOffset<int32_t>(64, 2112);
    values_82_ascendc = vecin_buff.GetWithOffset<int32_t>(16, 1600);
    values_84_ascendc = vecin_buff.GetWithOffset<int32_t>(16, 1664);
    values_86_ascendc = vecin_buff.GetWithOffset<int32_t>(16, 1728);
    values_88_ascendc = vecin_buff.GetWithOffset<int32_t>(16, 1792);
    values_90_ascendc = vecin_buff.GetWithOffset<int32_t>(16, 1856);
    values_92_ascendc = vecin_buff.GetWithOffset<int32_t>(16, 1920);
    values_94_ascendc = vecin_buff.GetWithOffset<int32_t>(16, 1984);
    AscendC::DataCopy(ub_97_ascendc, ub_range_79_ascendc, {1, 1, 0, 0});
    AscendC::DataCopy(ub_97_ascendc[8], values_82_ascendc, {1, 1, 0, 0});
    AscendC::DataCopy(ub_97_ascendc[16], values_84_ascendc, {1, 1, 0, 0});
    AscendC::DataCopy(ub_97_ascendc[24], values_86_ascendc, {1, 1, 0, 0});
    AscendC::DataCopy(ub_97_ascendc[32], values_88_ascendc, {1, 1, 0, 0});
    AscendC::DataCopy(ub_97_ascendc[40], values_90_ascendc, {1, 1, 0, 0});
    AscendC::DataCopy(ub_97_ascendc[48], values_92_ascendc, {1, 1, 0, 0});
    AscendC::DataCopy(ub_97_ascendc[56], values_94_ascendc, {1, 1, 0, 0});
    AscendC::PipeBarrier<PIPE_V>();
    values_99_ascendc = vecin_buff.GetWithOffset<int32_t>(64, 1536);
    ub_97_ascendc = vecin_buff.GetWithOffset<int32_t>(64, 2112);
    Adds<int32_t, false>(values_99_ascendc, ub_97_ascendc, (int32_t)64, AscendC::MASK_PLACEHOLDER, 1, {1, 1, 8, 8});
    values_96_ascendc = vecin_buff.GetWithOffset<int32_t>(16, 2048);
    AscendC::SetMaskNorm();
    AscendC::SetVectorMask<int32_t, AscendC::MaskMode::NORMAL>(0ULL, (1ULL << (16)) - 1);
    Adds<int32_t, false>(values_96_ascendc, values_94_ascendc, (int32_t)8, AscendC::MASK_PLACEHOLDER, 1, {1, 1, 8, 8});
    AscendC::SetMaskNorm();
    AscendC::SetVectorMask<int32_t, AscendC::MaskMode::NORMAL>(18446744073709551615ULL, 18446744073709551615ULL);
    AscendC::PipeBarrier<PIPE_V>();
    ub_indices_102_ascendc = vecout_buff.GetWithOffset<int32_t>(128, 2368);
    values_99_ascendc = vecin_buff.GetWithOffset<int32_t>(64, 1536);
    AscendC::DataCopy(ub_indices_102_ascendc, ub_97_ascendc, {1, 8, 0, 0});
    AscendC::DataCopy(ub_indices_102_ascendc[64], values_99_ascendc, {1, 8, 0, 0});
    AscendC::PipeBarrier<PIPE_V>();
    values_101_ascendc = vecin_buff.GetWithOffset<int32_t>(64, 1792);
    Adds<int32_t, false>(values_101_ascendc, values_99_ascendc, (int32_t)64, AscendC::MASK_PLACEHOLDER, 1, {1, 1, 8, 8});
    AscendC::PipeBarrier<PIPE_V>();
    ub_indices_103_ascendc = vecin_buff.GetWithOffset<float>(128, 1536);
    ub_indices_102_ascendc = vecin_buff.GetWithOffset<int32_t>(128, 2368);
    AscendC::Cast<float, int32_t, false>(ub_indices_103_ascendc, ub_indices_102_ascendc, AscendC::RoundMode::CAST_NONE, AscendC::MASK_PLACEHOLDER, 2, {1, 1, 8, 8});
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
    current_score_54_ascendc = vecin_buff.GetWithOffset<float>(128, 1024);
    Exp<float, false>(current_score_54_ascendc, current_score_50_ascendc, AscendC::MASK_PLACEHOLDER, 2, {1, 1, 8, 8});
    AscendC::PipeBarrier<PIPE_V>();
    sum_exp_59_ascendc = vecin_buff.GetWithOffset<float>(1, 576);
    ub_58_ascendc = vecin_buff.GetWithOffset<float>(16, 512);
    for (int rep = 0; rep < 1; ++rep) {
      AscendC::BlockReduceSum<float, false>(ub_58_ascendc.ReinterpretCast<float>(), current_score_54_ascendc[rep * 128].ReinterpretCast<float>(), 2, AscendC::MASK_PLACEHOLDER, 1, 1, 8);
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::SetMaskNorm();
      AscendC::SetVectorMask<float, AscendC::MaskMode::NORMAL>(0ULL, (1ULL << (16)) - 1);
      AscendC::BlockReduceSum<float, false>(ub_58_ascendc.ReinterpretCast<float>(), ub_58_ascendc.ReinterpretCast<float>(), 1, AscendC::MASK_PLACEHOLDER, 1, 1, 8);
      AscendC::SetMaskNorm();
      AscendC::SetVectorMask<float, AscendC::MaskMode::NORMAL>(18446744073709551615ULL, 18446744073709551615ULL);
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::SetMaskNorm();
      AscendC::SetVectorMask<float, AscendC::MaskMode::NORMAL>(0ULL, (1ULL << (2)) - 1);
      AscendC::BlockReduceSum<float, false>(sum_exp_59_ascendc[rep].ReinterpretCast<float>(), ub_58_ascendc.ReinterpretCast<float>(), 1, AscendC::MASK_PLACEHOLDER, 1, 1, 8);
      AscendC::SetMaskNorm();
      AscendC::SetVectorMask<float, AscendC::MaskMode::NORMAL>(18446744073709551615ULL, 18446744073709551615ULL);
    }
    AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
    AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
    sum_exp_63 = *(__ubuf__ float *)sum_exp_59;
    res_68 = float((float)1.0000000000) / float(sum_exp_63);
    AscendC::SetFlag<AscendC::HardEvent::S_V>(0);
    AscendC::WaitFlag<AscendC::HardEvent::S_V>(0);
    current_score_72_ascendc = vecin_buff.GetWithOffset<float>(128, 512);
    Muls<float, false>(current_score_72_ascendc, current_score_54_ascendc, res_68, AscendC::MASK_PLACEHOLDER, 2, {1, 1, 8, 8});
    AscendC::PipeBarrier<PIPE_V>();
    current_score_80_ascendc = vecin_buff.GetWithOffset<float>(128, 1024);
    current_score_76_ascendc = vecin_buff.GetWithOffset<float>(128, 512);
    Add<float, false>(current_score_80_ascendc, current_score_76_ascendc, current_extra_score_10_ascendc, AscendC::MASK_PLACEHOLDER, 2, {1, 1, 1, 8, 8, 8});
    AscendC::PipeBarrier<PIPE_V>();
    concat_local_104_ascendc = vecin_buff.GetWithOffset<float>(1024, 2048);
    AscendC::ProposalConcat<float>(concat_local_104_ascendc, current_score_80_ascendc, 8, 4);
    AscendC::ProposalConcat<float>(concat_local_104_ascendc, ub_indices_103_ascendc, 8, 5);
    AscendC::PipeBarrier<PIPE_V>();
    sorted_local_105_ascendc = vecin_buff.GetWithOffset<float>(1024, 6144);
    AscendC::RpSort16<float>(sorted_local_105_ascendc, concat_local_104_ascendc, 8);
    AscendC::PipeBarrier<PIPE_V>();
    sorted_0_108_ascendc = vecout_buff.GetWithOffset<float>(128, 1568);
    AscendC::DataCopy(sorted_0_108_ascendc, sorted_local_105_ascendc, {(uint16_t)(1), (uint16_t)(16), (uint16_t)(0), (uint16_t)(0)});
    sorted_1_111_ascendc = vecout_buff.GetWithOffset<float>(128, 2080);
    AscendC::DataCopy(sorted_1_111_ascendc, sorted_local_105_ascendc[128], {(uint16_t)(1), (uint16_t)(16), (uint16_t)(0), (uint16_t)(0)});
    sorted_2_114_ascendc = vecout_buff.GetWithOffset<float>(128, 2592);
    AscendC::DataCopy(sorted_2_114_ascendc, sorted_local_105_ascendc[256], {(uint16_t)(1), (uint16_t)(16), (uint16_t)(0), (uint16_t)(0)});
    sorted_3_117_ascendc = vecout_buff.GetWithOffset<float>(128, 3104);
    AscendC::DataCopy(sorted_3_117_ascendc, sorted_local_105_ascendc[384], {(uint16_t)(1), (uint16_t)(16), (uint16_t)(0), (uint16_t)(0)});
    sorted_0_120_ascendc = vecout_buff.GetWithOffset<float>(128, 3616);
    AscendC::DataCopy(sorted_0_120_ascendc, sorted_local_105_ascendc[512], {(uint16_t)(1), (uint16_t)(16), (uint16_t)(0), (uint16_t)(0)});
    sorted_1_123_ascendc = vecout_buff.GetWithOffset<float>(128, 544);
    AscendC::DataCopy(sorted_1_123_ascendc, sorted_local_105_ascendc[640], {(uint16_t)(1), (uint16_t)(16), (uint16_t)(0), (uint16_t)(0)});
    sorted_2_126_ascendc = vecout_buff.GetWithOffset<float>(128, 1056);
    AscendC::DataCopy(sorted_2_126_ascendc, sorted_local_105_ascendc[768], {(uint16_t)(1), (uint16_t)(16), (uint16_t)(0), (uint16_t)(0)});
    sorted_3_129_ascendc = vecout_buff.GetWithOffset<float>(128, 10240);
    AscendC::DataCopy(sorted_3_129_ascendc, sorted_local_105_ascendc[896], {(uint16_t)(1), (uint16_t)(16), (uint16_t)(0), (uint16_t)(0)});
    AscendC::PipeBarrier<PIPE_V>();
    sort_tmp_local_134_ascendc = vecin_buff.GetWithOffset<float>(512, 4128);
    sorted_0_108_ascendc = vecin_buff.GetWithOffset<float>(128, 1568);
    sorted_1_111_ascendc = vecin_buff.GetWithOffset<float>(128, 2080);
    sorted_2_114_ascendc = vecin_buff.GetWithOffset<float>(128, 2592);
    sorted_3_117_ascendc = vecin_buff.GetWithOffset<float>(128, 3104);
    uint16_t sorted_0_108_lengths[4] = {16, 16, 16, 16};
    struct AscendC::MrgSortSrcList<float>sorted_0_108_list(sorted_0_108_ascendc, sorted_1_111_ascendc, sorted_2_114_ascendc, sorted_3_117_ascendc);
    struct AscendC::MrgSort4Info src_info_sorted_0_108(sorted_0_108_lengths, false, 15, 1);
    AscendC::MrgSort4(sort_tmp_local_134_ascendc, sorted_0_108_list, src_info_sorted_0_108);
    sort_tmp_local_139_ascendc = vecin_buff.GetWithOffset<float>(512, 6176);
    sorted_0_120_ascendc = vecin_buff.GetWithOffset<float>(128, 3616);
    sorted_1_123_ascendc = vecin_buff.GetWithOffset<float>(128, 544);
    sorted_2_126_ascendc = vecin_buff.GetWithOffset<float>(128, 1056);
    sorted_3_129_ascendc = vecin_buff.GetWithOffset<float>(128, 10240);
    uint16_t sorted_0_120_lengths[4] = {16, 16, 16, 16};
    struct AscendC::MrgSortSrcList<float>sorted_0_120_list(sorted_0_120_ascendc, sorted_1_123_ascendc, sorted_2_126_ascendc, sorted_3_129_ascendc);
    struct AscendC::MrgSort4Info src_info_sorted_0_120(sorted_0_120_lengths, false, 15, 1);
    AscendC::MrgSort4(sort_tmp_local_139_ascendc, sorted_0_120_list, src_info_sorted_0_120);
    AscendC::PipeBarrier<PIPE_V>();
    sort_tmp_local_142_ascendc = vecin_buff.GetWithOffset<float>(512, 512);
    uint16_t sort_tmp_local_134_lengths[4] = {32, 32, 0, 0};
    struct AscendC::MrgSortSrcList<float>sort_tmp_local_134_list(sort_tmp_local_134_ascendc, sort_tmp_local_139_ascendc, sort_tmp_local_139_ascendc, sort_tmp_local_139_ascendc);
    struct AscendC::MrgSort4Info src_info_sort_tmp_local_134(sort_tmp_local_134_lengths, false, 3, 1);
    AscendC::MrgSort4(sort_tmp_local_142_ascendc, sort_tmp_local_134_list, src_info_sort_tmp_local_134);
    AscendC::PipeBarrier<PIPE_V>();
    current_score_143_ascendc = vecin_buff.GetWithOffset<float>(64, 4160);
    ProposalExtract(current_score_143_ascendc, sort_tmp_local_142_ascendc, 4, 4);
    ub_indices_144_ascendc = vecin_buff.GetWithOffset<float>(64, 3904);
    ProposalExtract(ub_indices_144_ascendc, sort_tmp_local_142_ascendc, 4, 5);
    AscendC::PipeBarrier<PIPE_V>();
    topk_vals_148_ascendc = vecout_buff.GetWithOffset<float>(8, 768);
    AscendC::DataCopy(topk_vals_148_ascendc, current_score_143_ascendc, {(uint16_t)(1), (uint16_t)(1), (uint16_t)(0), (uint16_t)(0)});
    AscendC::PipeBarrier<PIPE_V>();
    sum_val_153_ascendc = vecin_buff.GetWithOffset<float>(1, 4192);
    topk_vals_148_ascendc = vecin_buff.GetWithOffset<float>(8, 768);
    ub_152_ascendc = vecin_buff.GetWithOffset<float>(1, 4160);
    AscendC::SetMaskNorm();
    AscendC::SetVectorMask<float, AscendC::MaskMode::NORMAL>(0ULL, (1ULL << (8)) - 1);
    AscendC::BlockReduceSum<float, false>(sum_val_153_ascendc.ReinterpretCast<float>(), topk_vals_148_ascendc.ReinterpretCast<float>(), 1, AscendC::MASK_PLACEHOLDER, 1, 1, 8);
    AscendC::SetMaskNorm();
    AscendC::SetVectorMask<float, AscendC::MaskMode::NORMAL>(18446744073709551615ULL, 18446744073709551615ULL);
    ub_indices_145_ascendc = vecin_buff.GetWithOffset<int32_t>(64, 512);
    AscendC::Cast<int32_t, float, false>(ub_indices_145_ascendc, ub_indices_144_ascendc, AscendC::RoundMode::CAST_ROUND, AscendC::MASK_PLACEHOLDER, 1, {1, 1, 8, 8});
    AscendC::PipeBarrier<PIPE_V>();
    topk_indices_151_ascendc = vecout_buff.GetWithOffset<int32_t>(8, 3904);
    AscendC::DataCopy(topk_indices_151_ascendc, ub_indices_145_ascendc, {(uint16_t)(1), (uint16_t)(1), (uint16_t)(0), (uint16_t)(0)});
    AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
    AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
    sum_val_157 = *(__ubuf__ float *)sum_val_153;
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
    expert_index_ascendc.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(expert_index));
    AscendC::DataCopy(expert_index_ascendc[(res_46) * 8], topk_indices_151_ascendc, {(uint16_t)(1), (uint16_t)(1), (uint16_t)(0), (uint16_t)(0)});
    res_159 = float((float)1.0000000000) / float(sum_val_157);
    AscendC::SetFlag<AscendC::HardEvent::S_V>(0);
    AscendC::WaitFlag<AscendC::HardEvent::S_V>(0);
    topk_vals_160_ascendc = vecin_buff.GetWithOffset<float>(8, 4192);
    AscendC::SetMaskNorm();
    AscendC::SetVectorMask<float, AscendC::MaskMode::NORMAL>(0ULL, (1ULL << (8)) - 1);
    Muls<float, false>(topk_vals_160_ascendc, topk_vals_148_ascendc, res_159, AscendC::MASK_PLACEHOLDER, 1, {1, 1, 8, 8});
    AscendC::SetMaskNorm();
    AscendC::SetVectorMask<float, AscendC::MaskMode::NORMAL>(18446744073709551615ULL, 18446744073709551615ULL);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
    expert_weight_ascendc.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(expert_weight));
    topk_vals_160_ascendc = vecout_buff.GetWithOffset<float>(8, 4192);
    AscendC::DataCopy(expert_weight_ascendc[(res_45) * 8], topk_vals_160_ascendc, {(uint16_t)(1), (uint16_t)(1), (uint16_t)(0), (uint16_t)(0)});
    AscendC::PipeBarrier<PIPE_ALL>();
  }
}
  __aicore__ inline void CopyIn(int32_t index) {
  }
  __aicore__ inline void Compute() {
  }
  __aicore__ inline void CopyOut(int32_t index) {
  }
private:
  GM_ADDR logits;
  GM_ADDR bias;
  GM_ADDR expert_weight;
  GM_ADDR expert_index;
  int32_t num_tokens;
};

extern "C" __global__ __aicore__ void fused_add_topk_div_moe_ms(GM_ADDR logits, GM_ADDR bias, GM_ADDR expert_weight,
                                                              GM_ADDR expert_index, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tilingData, tiling);
    KernelFusedAddTopkDivMoeMS op;
    op.Init(logits, bias, expert_weight, expert_index, &tilingData);
    op.Process();
}

#ifndef ASCENDC_CPU_DEBUG
void fused_add_topk_div_moe_ms_do(uint32_t blockDim, void *l2ctrl, void *stream, uint8_t *logits, uint8_t *bias,
                              uint8_t *expert_weight, uint8_t *expert_index, uint8_t *workspace, uint8_t *tiling) {
  fused_add_topk_div_moe_ms<<<blockDim, l2ctrl, stream>>>(logits, bias, expert_weight, expert_index, workspace, tiling);
}
#endif
// NOLINTEND