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

#include <map>
#include <memory>
#include <string>
#include <set>
#include <vector>
#include "ops/framework/aclnn/graphmode/aclnn_kernel_mod.h"
#include "ops/framework/utils.h"
#include "ops/c_api/utils/common_utils.h"
#include "ops/c_api/utils/check_utils.h"

namespace ms_custom_ops {
enum class ScatterNdUpdateInputIndex : size_t {
  kScatterNdUpdateInputIndex = 0,
  kScatterNdUpdateIndicesIndex,
  kScatterNdUpdateUpdatesIndex,
  kScatterNdUpdateInputsNum,
};

class OPS_API ScatterNdUpdateCustomOpFuncImpl : public OpFuncImpl {
 public:
  ShapeArray InferShape(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const override {
    MS_EXCEPTION_IF_NULL(primitive);
    auto op_name = primitive->name();

    if ((input_infos[static_cast<size_t>(ScatterNdUpdateInputIndex::kScatterNdUpdateInputIndex)]->IsDynamicRank()) ||
        (input_infos[static_cast<size_t>(ScatterNdUpdateInputIndex::kScatterNdUpdateIndicesIndex)]->IsDynamicRank()) ||
        (input_infos[static_cast<size_t>(ScatterNdUpdateInputIndex::kScatterNdUpdateUpdatesIndex)]->IsDynamicRank())) {
      return kFakeOutTensorShapes;
    }
    auto indices_shape =
      input_infos[static_cast<size_t>(ScatterNdUpdateInputIndex::kScatterNdUpdateIndicesIndex)]->GetShape();

    CHECK_TENSOR_DIM_GE(indices_shape, kDim2, op_name);
    auto input_shape =
      input_infos[static_cast<size_t>(ScatterNdUpdateInputIndex::kScatterNdUpdateInputIndex)]->GetShape();
    // otherwise its dims should be 1~8.
    CHECK_TENSOR_DIM_RANGE(input_shape, kDim1, kDim8, op_name, "input tensor",
                           static_cast<size_t>(ScatterNdUpdateInputIndex::kScatterNdUpdateInputIndex));

    return kFakeOutTensorShapes;
  }

  std::vector<TypeId> InferType(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const override {
    MS_EXCEPTION_IF_NULL(primitive);
    auto op_name = primitive->name();
    auto input_dtype =
      input_infos[static_cast<size_t>(ScatterNdUpdateInputIndex::kScatterNdUpdateInputIndex)]->GetType();
    auto updates_dtype =
      input_infos[static_cast<size_t>(ScatterNdUpdateInputIndex::kScatterNdUpdateUpdatesIndex)]->GetType();
    if (input_dtype != updates_dtype) {
      MS_LOG(EXCEPTION) << "input dtype should be same with updates dtype, but input_dtype:" << input_dtype
                        << ", updates dtype:" << updates_dtype;
    }
    auto ms_context = MsContext::GetInstance();
    MS_EXCEPTION_IF_NULL(ms_context);
    const auto &soc_version = ms_context->ascend_soc_version();

    if (IsSoc910BC()) {
      const std::set<TypeId> valid_types = {
        kNumberTypeFloat16, kNumberTypeBFloat16, kNumberTypeFloat32, kNumberTypeInt64, kNumberTypeBool, kNumberTypeInt8,
      };
      CheckAndConvertUtils::CheckTypeIdValid("input", input_dtype, valid_types, op_name);
      CheckAndConvertUtils::CheckTypeIdValid("updates", updates_dtype, valid_types, op_name);
    } else if (IsSoc310p()) {
      const std::set<TypeId> valid_types = {kNumberTypeFloat16, kNumberTypeFloat32, kNumberTypeBool};
      CheckAndConvertUtils::CheckTypeIdValid("input", input_dtype, valid_types, op_name);
      CheckAndConvertUtils::CheckTypeIdValid("updates", updates_dtype, valid_types, op_name);
    } else {
      MS_LOG(EXCEPTION) << "'scatter_nd_update' only support [" << kAscendVersion910b << ", " << kAscendVersion910_93
                        << ", " << kAscendVersion310p << "], but got " << soc_version;
    }

    return kFakeOutTensorTypes;
  }

  bool GeneralInferRegistered() const override { return true; }
};

class ScatterNdUpdateCustomAscend : public AclnnCustomKernelMod {
 public:
  ScatterNdUpdateCustomAscend() : AclnnCustomKernelMod("aclnnScatterNdUpdate") {}
  ~ScatterNdUpdateCustomAscend() = default;

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs, void *stream_ptr) override {
    MS_EXCEPTION_IF_NULL(stream_ptr);
    RunOp(stream_ptr, workspace, inputs[static_cast<size_t>(ScatterNdUpdateInputIndex::kScatterNdUpdateInputIndex)],
          inputs[static_cast<size_t>(ScatterNdUpdateInputIndex::kScatterNdUpdateIndicesIndex)],
          inputs[static_cast<size_t>(ScatterNdUpdateInputIndex::kScatterNdUpdateUpdatesIndex)]);
    return true;
  }

  void GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                        const std::vector<KernelTensor *> &outputs) override {
    GetWorkspaceForResize(inputs[static_cast<size_t>(ScatterNdUpdateInputIndex::kScatterNdUpdateInputIndex)],
                          inputs[static_cast<size_t>(ScatterNdUpdateInputIndex::kScatterNdUpdateIndicesIndex)],
                          inputs[static_cast<size_t>(ScatterNdUpdateInputIndex::kScatterNdUpdateUpdatesIndex)]);
    return;
  }

 private:
  DEFINE_GET_WORKSPACE_FOR_RESIZE();
};
}  // namespace ms_custom_ops

REG_GRAPH_MODE_OP(scatter_nd_update, ms_custom_ops::ScatterNdUpdateCustomOpFuncImpl,
                  ms_custom_ops::ScatterNdUpdateCustomAscend);

// =============================================================================
// PYBOOST MODE IMPLEMENTATION
// =============================================================================

namespace ms_custom_ops {
void scatter_nd_update_custom(const ms::Tensor &input, const ms::Tensor &indices, const ms::Tensor &updates) {
  std::vector<ms::Tensor> outputs = {};
  auto runner = std::make_shared<ms::pynative::AclnnOpRunner>("aclnnScatterNdUpdate");
  runner->SetLaunchFunc(LAUNCH_ACLNN_FUNC(aclnnScatterNdUpdate, input, indices, updates));
  // only set tensor.
  runner->Run({input, indices, updates}, outputs);
  return;
}
}  // namespace ms_custom_ops

auto pyboost_scatter_nd_update(const ms::Tensor &input, const ms::Tensor &indices, const ms::Tensor &updates) {
  return ms::pynative::PyboostRunner::Call<ms_custom_ops::kNumber0>(ms_custom_ops::scatter_nd_update_custom, input,
                                                                    indices, updates);
}

MS_CUSTOM_OPS_EXTENSION_MODULE(m) {
  m.def("scatter_nd_update", &pyboost_scatter_nd_update, "Scatter Nd Update", pybind11::arg("input"),
        pybind11::arg("indices"), pybind11::arg("updates"));
}
