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

// =============================================================================
// GRAPH MODE IMPLEMENTATION
// =============================================================================

#include <vector>
#include "ops/framework/ms_kernels_internal/graphmode/internal_kernel_mod.h"
#include "ops/framework/utils.h"

namespace ms_custom_ops {
enum class InputIndex : size_t {
  kInputTokenIndex = 0,
  kInputIdxArrIndex = 1,
  kInputGroupNumIndex = 2,
  kInputKIndex = 3,
  kInputKInnerIndex = 4,
};

enum class OutputIndex : size_t { kOutputIndex = 0 };

class OPS_API CustomGroupTopkOpFuncImpl : public OpFuncImpl {
 public:
  ShapeArray InferShape(const PrimitivePtr &primitive,
                        const InferInfoPtrList &input_infos) const override {
    return kFakeOutTensorShapes;
  }

  std::vector<TypeId> InferType(const PrimitivePtr &primitive,
                                const InferInfoPtrList &input_infos) const override {
    return kFakeOutTensorTypes;
  }

  bool GeneralInferRegistered() const override { return true; }
};

class CustomGroupTopk : public InternalKernelMod {
 public:
  CustomGroupTopk() : InternalKernelMod() {}
  ~CustomGroupTopk() = default;

  void InitKernelInputsOutputsIndex() override {
    kernel_inputs_index_ = {
      static_cast<size_t>(InputIndex::kInputTokenIndex), static_cast<size_t>(InputIndex::kInputIdxArrIndex)};
    kernel_outputs_index_ = {static_cast<size_t>(OutputIndex::kOutputIndex)};
  }

 protected:
  internal_v2::InternalOpPtr CreateKernel(const internal_v2::InputsImmutableInfoList &inputs,
                                          const internal_v2::OutputsImmutableInfoList &outputs,
                                          const std::vector<KernelTensor *> &ms_inputs,
                                          const std::vector<KernelTensor *> &ms_outputs) override {
    internal_v2::GroupTopkParam param;
    auto input_group_num = ms_inputs.at(static_cast<size_t>(InputIndex::kInputGroupNumIndex));
    auto input_k = ms_inputs.at(static_cast<size_t>(InputIndex::kInputKIndex));
    auto input_k_inner = ms_inputs.at(static_cast<size_t>(InputIndex::kInputKInnerIndex));

    MS_EXCEPTION_IF_NULL(input_group_num);
    MS_EXCEPTION_IF_NULL(input_k);
    MS_EXCEPTION_IF_NULL(input_k_inner);

    if (input_group_num->dtype_id() == TypeId::kNumberTypeInt64 && input_k->dtype_id() == TypeId::kNumberTypeInt64 &&
        input_k_inner->dtype_id() == TypeId::kNumberTypeInt64) {
      param.group_num = static_cast<int32_t>(input_group_num->GetValue<int64_t>().value());
      param.k = static_cast<int32_t>(input_k->GetValue<int64_t>().value());
      param.k_inner = static_cast<int32_t>(input_k_inner->GetValue<int64_t>().value());
    } else {
      MS_LOG(EXCEPTION) << "GroupTopk [group_num, k, k_inner]'s dtype should all be kNumberTypeInt64, but is ["
                        << TypeIdToString(input_group_num->dtype_id()) << ", "
                        << TypeIdToString(input_k->dtype_id()) << ","
                        << TypeIdToString(input_k_inner->dtype_id()) << "]";
    }
    return internal_v2::CreateGroupTopkOp(inputs, outputs, param, internal_v2::kInternalGroupTopkOpName);
  }
};
}  // namespace ms_custom_ops

REG_GRAPH_MODE_OP(group_topk, ms_custom_ops::CustomGroupTopkOpFuncImpl,
                  ms_custom_ops::CustomGroupTopk);
