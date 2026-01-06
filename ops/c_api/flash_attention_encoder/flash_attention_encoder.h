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

#ifndef CCSRC_OPS_MS_KERNELS_INTERNAL_FLASH_ATTENTION_ENCODER_FLASH_ATTENTION_ENCODER_H_
#define CCSRC_OPS_MS_KERNELS_INTERNAL_FLASH_ATTENTION_ENCODER_FLASH_ATTENTION_ENCODER_H_

#include <vector>
#include <memory>
#include <set>
#include <cstdint>
#include "ops/framework/ms_kernels_internal/graphmode/internal_kernel_mod.h"
#include "ops/framework/utils.h"
#include "ops/c_api/utils/attention_utils.h"
namespace ms_custom_ops {

enum FlashAttentionEncoderInputIndex : int {
  kQueryIdx = 0,
  kKeyIdx,
  kValueIdx,
  kLayerIdIdx,
  kMaskIdx,
  kAlibiCoeffIdx,
  kDeqScaleQkIdx,
  kDeqOffsetQkIdx,
  kDeqScalePvIdx,
  kDeqOffsetPvIdx,
  kQuantPIdx,
  kLogNIdx,
  kQSeqLenIdx,
  kKVSeqLenIdx,
  kHeadNumIdx,
  kScaleValueIdx,
  kKvHeadNumIdx,
  kMaskTypeIdx,
  kKernelTypeIdx,
  kWindowSizeIdx,
  kCacheTypeIdx,
  kInputFormatIdx,
  kInputNums
};

enum FlashAttentionEncoderOutputIndex : int { kAttentionOutIdx = 0, kOutputNums };

class OPS_API FlashAttentionEncoderOpFuncImpl : public OpFuncImpl {
 public:
  ShapeArray InferShape(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const override;
  std::vector<TypeId> InferType(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const override;
  bool GeneralInferRegistered() const override { return true; }
  std::set<int64_t> GetValueDependArgIndices() const override {
    return {kQSeqLenIdx,  kKVSeqLenIdx,   kHeadNumIdx,    kScaleValueIdx, kKvHeadNumIdx,
            kMaskTypeIdx, kKernelTypeIdx, kWindowSizeIdx, kCacheTypeIdx,  kInputFormatIdx};
  };
};

class CustomFlashAttentionEncoder : public InternalKernelMod {
 public:
  CustomFlashAttentionEncoder() = default;
  ~CustomFlashAttentionEncoder() override = default;

 protected:
  void InitKernelInputsOutputsIndex() override;
  internal_v2::InternalOpPtr CreateKernel(const internal_v2::InputsImmutableInfoList &inputs,
                                          const internal_v2::OutputsImmutableInfoList &outputs,
                                          const std::vector<KernelTensor *> &ms_inputs,
                                          const std::vector<KernelTensor *> &ms_outputs) override;
  bool UpdateParam(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;
  uint64_t GenerateTilingKey(const std::vector<KernelTensor *> &inputs) override;

 private:
  bool created_flag_{false};
  internal_v2::SelfAttentionParam param_{};
};

}  // namespace ms_custom_ops

#endif  // CCSRC_OPS_MS_KERNELS_INTERNAL_FLASH_ATTENTION_ENCODER_FLASH_ATTENTION_ENCODER_H_
