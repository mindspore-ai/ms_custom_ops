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

#include "ops/c_api/flash_attention_encoder/flash_attention_encoder.h"
#include "ops/c_api/utils/attention_utils.h"
#include "ops/framework/utils.h"

namespace ms_custom_ops {

ShapeArray FlashAttentionEncoderOpFuncImpl::InferShape(const PrimitivePtr &primitive,
                                                       const InferInfoPtrList &input_infos) const {
  (void)primitive;
  auto query_shape = input_infos[kQueryIdx]->GetShape();
  return {query_shape};
}

std::vector<TypeId> FlashAttentionEncoderOpFuncImpl::InferType(const PrimitivePtr &primitive,
                                                               const InferInfoPtrList &input_infos) const {
  (void)primitive;
  auto query_type = input_infos[kQueryIdx]->GetType();
  return {query_type};
}

internal_v2::InternalOpPtr CustomFlashAttentionEncoder::CreateKernel(
  const internal_v2::InputsImmutableInfoList &inputs_ii, const internal_v2::OutputsImmutableInfoList &outputs_ii,
  const std::vector<KernelTensor *> &ms_inputs, const std::vector<KernelTensor *> &ms_outputs) {
  param_.head_num = static_cast<int32_t>(ms_inputs[kHeadNumIdx]->GetValueWithCheck<int64_t>());
  param_.qk_scale = ms_inputs[kScaleValueIdx]->GetValueWithCheck<float>();
  param_.kv_head_num = static_cast<int32_t>(ms_inputs[kKvHeadNumIdx]->GetValueWithCheck<int64_t>());
  // 固定为 PA_ENCODER 路径
  param_.calc_type = internal_v2::SelfAttentionParam::CalcType::PA_ENCODER;
  param_.kernel_type =
    static_cast<internal_v2::SelfAttentionParam::KernelType>(ms_inputs[kKernelTypeIdx]->GetValueWithCheck<int64_t>());
  param_.mask_type =
    static_cast<internal_v2::SelfAttentionParam::MaskType>(ms_inputs[kMaskTypeIdx]->GetValueWithCheck<int64_t>());
  param_.window_size = static_cast<uint32_t>(ms_inputs[kWindowSizeIdx]->GetValueWithCheck<int64_t>());
  param_.cache_type =
    static_cast<internal_v2::SelfAttentionParam::CacheType>(ms_inputs[kCacheTypeIdx]->GetValueWithCheck<int64_t>());
  // input_format: default 0 ND, 1 force NZ
  param_.input_format = static_cast<int32_t>(ms_inputs[kInputFormatIdx]->GetValueWithCheck<int64_t>());

  if (ms_inputs[kQSeqLenIdx]->dtype_id() != TypeId::kNumberTypeInt32) {
    MS_LOG(EXCEPTION) << "q_seq_len must be int32";
  }
  param_.q_seq_len = ms_inputs[kQSeqLenIdx]->GetValueWithCheck<std::vector<int32_t>>();

  if (ms_inputs[kKVSeqLenIdx]->dtype_id() != TypeId::kNumberTypeInt32) {
    MS_LOG(EXCEPTION) << "kv_seq_len must be int32";
  }
  param_.kv_seq_len = ms_inputs[kKVSeqLenIdx]->GetValueWithCheck<std::vector<int32_t>>();

  created_flag_ = true;
  // 当input_format为NZ时，设置Q/K/V和输出为FRACTAL_NZ格式
  if (param_.input_format == 1) {
    auto inputs_clone = inputs_ii;
    auto outputs_clone = outputs_ii;
    inputs_clone[static_cast<size_t>(kQueryIdx)].SetFormat(internal_v2::kFormatFRACTAL_NZ);
    inputs_clone[static_cast<size_t>(kKeyIdx)].SetFormat(internal_v2::kFormatFRACTAL_NZ);
    inputs_clone[static_cast<size_t>(kValueIdx)].SetFormat(internal_v2::kFormatFRACTAL_NZ);
    inputs_clone[static_cast<size_t>(kMaskIdx)].SetFormat(internal_v2::kFormatFRACTAL_NZ);
    outputs_clone[static_cast<size_t>(kAttentionOutIdx)].SetFormat(internal_v2::kFormatFRACTAL_NZ);
    return internal_v2::CreateFlashAttentionEncoderOp(inputs_clone, outputs_clone, param_,
                                                      internal_v2::kInternalFlashAttentionEncoderOpName);
  }

  return internal_v2::CreateFlashAttentionEncoderOp(inputs_ii, outputs_ii, param_,
                                                    internal_v2::kInternalFlashAttentionEncoderOpName);
}

bool CustomFlashAttentionEncoder::UpdateParam(const std::vector<KernelTensor *> &inputs,
                                              const std::vector<KernelTensor *> &outputs) {
  if (created_flag_) {
    created_flag_ = false;
    return true;
  }

  auto q_need_recreate = GetSeqLenAndCheckUpdate(inputs[kQSeqLenIdx], &param_.q_seq_len);
  auto kv_need_recreate = GetSeqLenAndCheckUpdate(inputs[kKVSeqLenIdx], &param_.kv_seq_len);
  if (q_need_recreate || kv_need_recreate) {
    auto ret = internal_op_->UpdateParam(&param_);
    if (ret != internal_v2::kInternalOk) {
      MS_LOG(ERROR) << "CustomFlashAttentionEncoder UpdateParam failed, kernel_name: " << kernel_name_;
      return false;
    }
    return true;
  }
  return true;
}

void CustomFlashAttentionEncoder::InitKernelInputsOutputsIndex() {
  kernel_inputs_index_ = {kQueryIdx,      kKeyIdx,         kValueIdx,      kLayerIdIdx,     kMaskIdx,   kAlibiCoeffIdx,
                          kDeqScaleQkIdx, kDeqOffsetQkIdx, kDeqScalePvIdx, kDeqOffsetPvIdx, kQuantPIdx, kLogNIdx};
  kernel_outputs_index_ = {kAttentionOutIdx};
}

uint64_t CustomFlashAttentionEncoder::GenerateTilingKey(const std::vector<KernelTensor *> &inputs) {
  return InternalTilingCache::GenerateKey(kernel_name_, inputs, param_.q_seq_len, param_.kv_seq_len);
}

}  // namespace ms_custom_ops

REG_GRAPH_MODE_OP(flash_attention_encoder, ms_custom_ops::FlashAttentionEncoderOpFuncImpl,
                  ms_custom_ops::CustomFlashAttentionEncoder);
