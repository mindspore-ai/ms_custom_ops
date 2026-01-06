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

#include <string>
#include <vector>
#include "ops/c_api/sparse_flash_attention/sparse_flash_attention_common.h"
#include "ops/framework/aclnn/graphmode/aclnn_kernel_mod.h"
#include "ops/framework/utils.h"
#include "mindspore/include/custom_op_api.h"

namespace ms_custom_ops {
class OPS_API SparseFlashAttentionFuncImpl : public OpFuncImpl {
 public:
  ShapeArray InferShape(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const override {
    return {input_infos[kQueryIndex]->GetShape()};
  }

  std::vector<TypeId> InferType(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const override {
    auto q_dtype = input_infos[kQueryIndex]->GetType();
    return {q_dtype};
  }

  bool GeneralInferRegistered() const override { return true; }
};

class SparseFlashAttentionAscendC : public AclnnCustomKernelMod {
 public:
  SparseFlashAttentionAscendC() : AclnnCustomKernelMod(std::move("aclnnSparseFlashAttention")) {}
  ~SparseFlashAttentionAscendC() = default;

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs, void *stream_ptr) override {
    RunOp(stream_ptr, workspace, inputs[kQueryIndex], inputs[kKeyIndex], inputs[kValueIndex],
          inputs[kSparseIndicesIndex], inputs[kBlockTable], inputs[kQSeqLenIndex], inputs[kKSeqLenIndex],
          inputs[kQRopeIndex], inputs[kKRopeIndex], scale_value_, sparse_block_size_, layout_query_, layout_kv_,
          sparse_mode_, outputs[0]);
    return true;
  }

  void GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                        const std::vector<KernelTensor *> &outputs) override {
    scale_value_ = static_cast<double>(inputs[kScaleValueIndex]->GetValueWithCheck<float>());
    sparse_block_size_ = inputs[kSparseBlockSizeIndex]->GetValueWithCheck<int64_t>();
    sparse_mode_ = inputs[kSparseModeIndex]->GetValueWithCheck<int64_t>();

    auto layout_query_int = inputs[kLayoutQueryIndex]->GetValueWithCheck<int64_t>();
    auto layout_kv_int = inputs[kLayoutKVIndex]->GetValueWithCheck<int64_t>();

    CheckLayout(layout_query_int, layout_kv_int);

    layout_query_ = kQueryLayouts[static_cast<size_t>(layout_query_int)];
    layout_kv_ = kKVLayouts[static_cast<size_t>(layout_kv_int)];

    GetWorkspaceForResize(inputs[kQueryIndex], inputs[kKeyIndex], inputs[kValueIndex], inputs[kSparseIndicesIndex],
                          inputs[kBlockTable], inputs[kQSeqLenIndex], inputs[kKSeqLenIndex], inputs[kQRopeIndex],
                          inputs[kKRopeIndex], scale_value_, sparse_block_size_, layout_query_, layout_kv_,
                          sparse_mode_, outputs[0]);
    return;
  }

 private:
  DEFINE_GET_WORKSPACE_FOR_RESIZE();
  std::string layout_query_;
  std::string layout_kv_;
  double scale_value_{0};
  int64_t sparse_block_size_{0};
  int64_t sparse_mode_{0};
};
}  // namespace ms_custom_ops

REG_GRAPH_MODE_OP(sparse_flash_attention, ms_custom_ops::SparseFlashAttentionFuncImpl,
                  ms_custom_ops::SparseFlashAttentionAscendC);
