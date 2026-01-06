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

#include "ops/framework/ms_kernels_internal/graphmode/internal_kernel_mod.h"
#include "ops/c_api/fa_update/fa_update_common.h"

namespace ms_custom_ops {
class OPS_API CustomFaUpdateOpFuncImpl : public OpFuncImpl {
 public:
  ShapeArray InferShape(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const override {
    if (input_infos[kFaUpdateInputLocalOutIndex]->IsDynamicRank()) {
        return {{abstract::Shape::kShapeRankAny}};
    }
    MS_CHECK_VALUE(input_infos[kFaUpdateInputLocalOutIndex]->GetShape().size() == kFaUpdateLocalOutShapeRank,
                   CheckAndConvertUtils::FormatCommMsg(
                    "For FaUpdate, local_out dim must be 3, but got : ",
                    input_infos[kFaUpdateInputLocalOutIndex]->GetShape().size()));

    int64_t dim1 = input_infos[kFaUpdateInputLocalOutIndex]->GetShape()[kIndex1];
    int64_t dim2 = input_infos[kFaUpdateInputLocalOutIndex]->GetShape()[kIndex2];
    MS_CHECK_VALUE(dim2 >= 8 && dim2 <= 512 && ALIGN_8(dim2),
                   CheckAndConvertUtils::FormatCommMsg(
                    "For FaUpdate, head_size must be in range [8, 512] and be the multiple of 8, but got : ", dim2));
    ShapeVector output_shape{dim1, dim2};
    return {output_shape};
  }
  std::vector<TypeId> InferType(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const override {
    return {TypeId::kNumberTypeFloat32};
  }

  bool GeneralInferRegistered() const override { return true; }
};

class CustomFaUpdate : public InternalKernelMod {
 public:
  CustomFaUpdate() : InternalKernelMod() {}
  ~CustomFaUpdate() = default;

  void InitKernelInputsOutputsIndex() override {
    kernel_inputs_index_ = {kFaUpdateInputLseIndex, kFaUpdateInputLocalOutIndex};
    kernel_outputs_index_ = {kFaUpdateOutputIndex};
  }

 protected:
  internal_v2::InternalOpPtr CreateKernel(const internal_v2::InputsImmutableInfoList &inputs,
                                          const internal_v2::OutputsImmutableInfoList &outputs,
                                          const std::vector<KernelTensor *> &ms_inputs,
                                          const std::vector<KernelTensor *> &ms_outputs) override {
    internal_v2::FaUpdateParam param;
    auto fa_update_type = ms_inputs.at(kFaUpdateParamFaUpdateTypeIndex);
    auto sp = ms_inputs.at(kFaUpdateParamSpIndex);
    param.fa_update_type = fa_update_type->GetValue<uint64_t>().value();
    param.sp = sp->GetValue<uint64_t>().value();
    return internal_v2::CreateFaUpdateOp(inputs, outputs, param, internal_v2::kInternalFaUpdateOpName);
  }
};
}  // namespace ms_custom_ops
REG_GRAPH_MODE_OP(fa_update, ms_custom_ops::CustomFaUpdateOpFuncImpl, ms_custom_ops::CustomFaUpdate);
