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
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "ops/framework/ms_kernels_internal/graphmode/internal_kernel_mod.h"
#include "ops/framework/utils.h"

namespace ms_custom_ops {
enum class GroupedMatmulW4InputIndex : size_t {
  kGmmW4XIndex = 0,
  kGmmW4WeightIndex,
  kGmmW4GroupListIndex,
  kGmmW4BiasIndex,
  kGmmW4XScaleIndex,
  kGmmW4WeightScaleIndex,
  kGmmW4InputsNum,
  kGmmW4UnusedScaleIndex,
};
enum class GroupedMatmulW4OutputIndex : size_t {
  kGmmW4OutputIndex = 0,
  kGmmW4OutputsNum,
};

// =============================================================================
// UTILITY FUNCTIONS
// =============================================================================
ShapeVector InferGroupedMatmulW4OutputShape(const ShapeVector &x_shape, const ShapeVector &weight_shape,
                                            const ShapeVector &group_list_shape) {
  auto x_shape_size = x_shape.size();
  auto weight_shape_size = weight_shape.size();
  auto group_list_shape_size = group_list_shape.size();
  // check input rank
  if (MS_UNLIKELY(x_shape_size != kDim2)) {
    MS_LOG(EXCEPTION) << "For GroupedMatmulW4, input 'x' must be 2D, but got:" << x_shape_size;
  }
  if (MS_UNLIKELY(weight_shape_size != kDim3)) {
    MS_LOG(EXCEPTION) << "For GroupedMatmulW4, input 'weight' must be 3D, but got:" << weight_shape_size;
  }
  if (MS_UNLIKELY(group_list_shape_size != kDim1)) {
    MS_LOG(EXCEPTION) << "For GroupedMatmulW4, input 'group_list' must be 1D, but got:" << group_list_shape_size;
  }
  // get input dimensions
  auto x_m_dim = x_shape[kIndex0];
  auto x_k_dim = x_shape[kIndex1];
  auto weight_n_dim = weight_shape[kIndex1];
  auto weight_k_dim = weight_shape[kIndex2];

  // infer output shape when weight_n_dim is dynamic
  if (weight_n_dim == abstract::Shape::kShapeDimAny) {
    return ShapeVector{x_m_dim, weight_n_dim};
  }

  // For qint4x2 weight type, adjust the logical dimensions based on storage layout
  auto weight_real_k_dim = weight_k_dim;
  // Stored as [e, n, k/2], so restore k dimension
  weight_real_k_dim <<= 1;

  if (MS_UNLIKELY(x_k_dim != abstract::Shape::kShapeDimAny && weight_k_dim != abstract::Shape::kShapeDimAny &&
                  x_k_dim != weight_real_k_dim)) {
    MS_LOG(EXCEPTION) << "For GroupedMatmulW4, input 'x' and 'weight' must have the same dimension of 'k', but got:"
                      << x_k_dim << " and " << weight_real_k_dim;
  }

  return ShapeVector{x_m_dim, weight_n_dim};
}

class OPS_API CustomGroupedMatmulW4OpFuncImpl : public OpFuncImpl {
 public:
  ShapeArray InferShape(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const override {
    // dynamic rank
    if (MS_UNLIKELY(input_infos[static_cast<size_t>(GroupedMatmulW4InputIndex::kGmmW4XIndex)]->IsDynamicRank() ||
                    input_infos[static_cast<size_t>(GroupedMatmulW4InputIndex::kGmmW4WeightIndex)]->IsDynamicRank())) {
      auto output_shape = ShapeVector{abstract::Shape::kShapeRankAny};
      return {output_shape};
    }
    // check input size
    if (MS_UNLIKELY(input_infos.size() != static_cast<size_t>(GroupedMatmulW4InputIndex::kGmmW4InputsNum))) {
      MS_LOG(EXCEPTION) << "For GroupedMatmulW4, input size must be " << GroupedMatmulW4InputIndex::kGmmW4InputsNum
                        << ", but got " << input_infos.size();
    }

    // get input shapes
    auto x_shape = input_infos[static_cast<size_t>(GroupedMatmulW4InputIndex::kGmmW4XIndex)]->GetShape();
    auto weight_shape = input_infos[static_cast<size_t>(GroupedMatmulW4InputIndex::kGmmW4WeightIndex)]->GetShape();
    auto group_list_shape =
      input_infos[static_cast<size_t>(GroupedMatmulW4InputIndex::kGmmW4GroupListIndex)]->GetShape();
    auto output_shape = InferGroupedMatmulW4OutputShape(x_shape, weight_shape, group_list_shape);
    return {output_shape};
  }

  std::vector<TypeId> InferType(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const override {
    MS_EXCEPTION_IF_NULL(primitive);
    auto ms_context = MsContext::GetInstance();
    MS_EXCEPTION_IF_NULL(ms_context);
    const auto &device_target = ms_context->ascend_soc_version();
    if (device_target != kAscendVersion310p) {
      MS_LOG(EXCEPTION) << "For GroupedMatmulW4, only support Ascend device, but got:" << device_target;
    }
    auto x_dtype = input_infos[static_cast<size_t>(GroupedMatmulW4InputIndex::kGmmW4XIndex)]->GetType();
    auto weight_dtype = input_infos[static_cast<size_t>(GroupedMatmulW4InputIndex::kGmmW4WeightIndex)]->GetType();
    auto group_list_dtype =
      input_infos[static_cast<size_t>(GroupedMatmulW4InputIndex::kGmmW4GroupListIndex)]->GetType();
    auto bias_dtype = input_infos[static_cast<size_t>(GroupedMatmulW4InputIndex::kGmmW4BiasIndex)]->GetType();
    auto x_scale_dtype = input_infos[static_cast<size_t>(GroupedMatmulW4InputIndex::kGmmW4XScaleIndex)]->GetType();
    auto weight_scale_dtype =
      input_infos[static_cast<size_t>(GroupedMatmulW4InputIndex::kGmmW4WeightScaleIndex)]->GetType();
    // check input types
    if (MS_UNLIKELY(x_dtype != kNumberTypeInt8)) {
      MS_LOG(EXCEPTION) << "For GroupedMatmulW4, input 'x' must be int8, but got:" << x_dtype;
    }
    if (MS_UNLIKELY(weight_dtype != kNumberTypeInt4)) {
      MS_LOG(EXCEPTION) << "For GroupedMatmulW4, input 'weight' must be qint4x2, but got:" << weight_dtype;
    }
    if (MS_UNLIKELY(group_list_dtype != kNumberTypeInt32)) {
      MS_LOG(EXCEPTION) << "For GroupedMatmulW4, input 'group_list' must be int32, but got:" << group_list_dtype;
    }
    if (MS_UNLIKELY(bias_dtype != kNumberTypeFloat32)) {
      MS_LOG(EXCEPTION) << "For GroupedMatmulW4, input 'bias' must be float32, but got:" << bias_dtype;
    }
    if (MS_UNLIKELY(x_scale_dtype != kNumberTypeFloat32)) {
      MS_LOG(EXCEPTION) << "For GroupedMatmulW4, input 'x_scale' must be float32, but got:" << x_scale_dtype;
    }
    if (MS_UNLIKELY(weight_scale_dtype != kNumberTypeFloat32)) {
      MS_LOG(EXCEPTION) << "For GroupedMatmulW4, input 'weight_scale' must be float32, but got:" << weight_scale_dtype;
    }
    // infer output types
    auto output_dtype = kNumberTypeFloat16;
    return {output_dtype};
  }

  bool GeneralInferRegistered() const override { return true; }
};

class CustomGroupedMatmulW4 : public InternalKernelMod {
 public:
  CustomGroupedMatmulW4() : InternalKernelMod() {}
  ~CustomGroupedMatmulW4() = default;

  void InitKernelInputsOutputsIndex() override {
    kernel_inputs_index_ = {static_cast<size_t>(GroupedMatmulW4InputIndex::kGmmW4XIndex),
                            static_cast<size_t>(GroupedMatmulW4InputIndex::kGmmW4WeightIndex),
                            static_cast<size_t>(GroupedMatmulW4InputIndex::kGmmW4UnusedScaleIndex),
                            static_cast<size_t>(GroupedMatmulW4InputIndex::kGmmW4BiasIndex),
                            static_cast<size_t>(GroupedMatmulW4InputIndex::kGmmW4GroupListIndex),
                            static_cast<size_t>(GroupedMatmulW4InputIndex::kGmmW4XScaleIndex),
                            static_cast<size_t>(GroupedMatmulW4InputIndex::kGmmW4WeightScaleIndex)};
    kernel_outputs_index_ = {static_cast<size_t>(GroupedMatmulW4OutputIndex::kGmmW4OutputIndex)};
  }

 protected:
  internal_v2::InternalOpPtr CreateKernel(const internal_v2::InputsImmutableInfoList &inputs,
                                          const internal_v2::OutputsImmutableInfoList &outputs,
                                          const std::vector<KernelTensor *> &ms_inputs,
                                          const std::vector<KernelTensor *> &ms_outputs) override {
    internal_v2::MatmulParam param;
    param.transpose_a = false;
    param.transpose_b = true;  // For w4, weight is always transposed
    param.with_bias = true;
    param.enable_shuffle = false;  // the real definition is in internal
    auto inputs_clone = inputs;
    inputs_clone[static_cast<size_t>(GroupedMatmulW4InputIndex::kGmmW4WeightIndex)].SetFormat(
      internal_v2::TensorFormat::kFormatFRACTAL_NZ);
    return internal_v2::CreateGroupedMatmulOp(inputs_clone, outputs, param, internal_v2::kInternalGroupedMatmulOpName);
  }
};
}  // namespace ms_custom_ops

REG_GRAPH_MODE_OP(grouped_matmul_w4, ms_custom_ops::CustomGroupedMatmulW4OpFuncImpl,
                  ms_custom_ops::CustomGroupedMatmulW4);

// =============================================================================
// PYBOOST MODE IMPLEMENTATION
// =============================================================================

#include "ops/framework/ms_kernels_internal/pyboost/internal_pyboost_runner.h"

namespace ms_custom_ops {
class GroupedMatmulW4Runner : public InternalPyboostRunner {
 public:
  using InternalPyboostRunner::InternalPyboostRunner;

  void SetParam() {
    param_.transpose_a = false;
    param_.transpose_b = true;  // For w4, weight is always transposed
    param_.with_bias = true;
    param_.enable_shuffle = false;  // the real definition is in internal
  }

 protected:
  internal_v2::InternalOpPtr CreateKernel(const internal_v2::InputsImmutableInfoList &inputs,
                                          const internal_v2::OutputsImmutableInfoList &outputs) override {
    auto inputs_clone = inputs;
    inputs_clone[static_cast<size_t>(GroupedMatmulW4InputIndex::kGmmW4WeightIndex)].SetFormat(
      internal_v2::TensorFormat::kFormatFRACTAL_NZ);
    return internal_v2::CreateGroupedMatmulOp(inputs_clone, outputs, param_, internal_v2::kInternalGroupedMatmulOpName);
  }

 private:
  internal_v2::MatmulParam param_;
};

std::vector<ms::Tensor> npu_grouped_matmul_w4(const ms::Tensor &x, const ms::Tensor &weight,
                                              const ms::Tensor &group_list, const ms::Tensor &bias,
                                              const ms::Tensor &x_scale, const ms::Tensor &weight_scale) {
  auto op_name = "GroupedMatmulW4";
  auto runner = std::make_shared<GroupedMatmulW4Runner>(op_name);
  MS_EXCEPTION_IF_NULL(runner);

  // Set param for internal kernel
  runner->SetParam();
  // Setup the runner with all parameters (including hash calculation)
  runner->Setup(op_name, x, weight, group_list, bias, x_scale, weight_scale);

  // if you need infer shape and type, you can use this
  std::vector<ms::Tensor> inputs = {x, weight, ms::Tensor(), bias, group_list, x_scale, weight_scale};
  auto out_shape = InferGroupedMatmulW4OutputShape(x.shape(), weight.shape(), group_list.shape());
  std::vector<ms::Tensor> outputs = {ms::Tensor(TypeId::kNumberTypeFloat16, out_shape)};
  runner->GetOrCreateKernel(inputs, outputs);
  runner->Run(inputs, outputs);
  return outputs;
}
}  // namespace ms_custom_ops

auto pyboost_grouped_matmul_w4(const ms::Tensor &x, const ms::Tensor &weight, const ms::Tensor &group_list,
                               const ms::Tensor &bias, const ms::Tensor &x_scale, const ms::Tensor &weight_scale) {
  return ms::pynative::PyboostRunner::Call<ms_custom_ops::kNumber1>(ms_custom_ops::npu_grouped_matmul_w4, x, weight,
                                                                    group_list, bias, x_scale, weight_scale);
}

MS_CUSTOM_OPS_EXTENSION_MODULE(m) {
  m.def("grouped_matmul_w4", &pyboost_grouped_matmul_w4, "GroupedMatmulW4", pybind11::arg("x"), pybind11::arg("weight"),
        pybind11::arg("group_list"), pybind11::arg("bias"), pybind11::arg("x_scale"), pybind11::arg("weight_scale"));
}
