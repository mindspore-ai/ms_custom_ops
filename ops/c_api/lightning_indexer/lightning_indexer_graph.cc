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
#include "ops/c_api/lightning_indexer/lightning_indexer_common.h"
#include "ops/framework/aclnn/graphmode/aclnn_kernel_mod.h"
#include "ops/framework/utils.h"

namespace ms_custom_ops {
class OPS_API LightningIndexerFuncImpl : public OpFuncImpl {
 public:
  ShapeArray InferShape(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const override {
    auto query_shape = input_infos[kQueryIndex]->GetShape();
    auto key_shape = input_infos[kKeyIndex]->GetShape();

    auto layout_query = input_infos[kLayoutQueryIndex]->GetScalarValueWithCheck<int64_t>();
    auto layout_key = input_infos[kLayoutKeyIndex]->GetScalarValueWithCheck<int64_t>();
    auto sparse_count = input_infos[kSparseCountIndex]->GetScalarValueWithCheck<int64_t>();

    auto out_shape = InferShapeForLightningIndexer(query_shape, key_shape, layout_query, layout_key, sparse_count);
    return {out_shape};
  }

  std::vector<TypeId> InferType(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const override {
    auto seq_k_dtype = input_infos[kKSeqLenIndex]->GetType();

    return {seq_k_dtype};
  }

  bool GeneralInferRegistered() const override { return true; }
};

class LightningIndexerAscendC : public AclnnCustomKernelMod {
 public:
  LightningIndexerAscendC() : AclnnCustomKernelMod(std::move("aclnnLightningIndexer")) {}
  ~LightningIndexerAscendC() = default;

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs, void *stream_ptr) override {
    RunOp(stream_ptr, workspace, inputs[kQueryIndex], inputs[kKeyIndex], inputs[kWeightsIndex], inputs[kQSeqLenIndex],
          inputs[kKSeqLenIndex], inputs[kBlockTable], layout_query_, layout_key_, sparse_count_, sparse_mode_,
          outputs[0]);
    return true;
  }

  void GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                        const std::vector<KernelTensor *> &outputs) override {
    auto layout_query_int = inputs[kLayoutQueryIndex]->GetValueWithCheck<int64_t>();
    auto layout_key_int = inputs[kLayoutKeyIndex]->GetValueWithCheck<int64_t>();
    layout_query_ = kQueryLayouts[static_cast<size_t>(layout_query_int)];
    layout_key_ = kKeyLayouts[static_cast<size_t>(layout_key_int)];

    sparse_count_ = inputs[kSparseCountIndex]->GetValueWithCheck<int64_t>();
    sparse_mode_ = inputs[kSparseModeIndex]->GetValueWithCheck<int64_t>();

    GetWorkspaceForResize(inputs[kQueryIndex], inputs[kKeyIndex], inputs[kWeightsIndex], inputs[kQSeqLenIndex],
                          inputs[kKSeqLenIndex], inputs[kBlockTable], layout_query_, layout_key_, sparse_count_,
                          sparse_mode_, outputs[0]);
    return;
  }

 private:
  DEFINE_GET_WORKSPACE_FOR_RESIZE();
  std::string layout_query_;
  std::string layout_key_;
  int64_t sparse_count_{0};
  int64_t sparse_mode_{0};
};
}  // namespace ms_custom_ops

REG_GRAPH_MODE_OP(lightning_indexer, ms_custom_ops::LightningIndexerFuncImpl, ms_custom_ops::LightningIndexerAscendC);
