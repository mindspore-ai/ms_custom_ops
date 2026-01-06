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
enum class GroupedMatmulInputIndex : size_t {
  kGmmXIndex = 0,
  kGmmWeightIndex,
  kGmmGroupListIndex,
  kGmmBiasIndex,
  kGmmScaleIndex,
  kGmmPerTokenScaleIndex,
  kGmmAntiquantScaleIndex,
  kGmmTransposeAIndex,
  kGmmTransposeBIndex,
  kGmmXFormatIndex,
  kGmmInputsNum,
};
enum class GroupedMatmulOutputIndex : size_t {
  kGmmOutputIndex = 0,
  kGmmOutputsNum,
};

// =============================================================================
// UTILITY FUNCTIONS
// =============================================================================
void CheckInputRank(const ShapeVector &x_shape, const ShapeVector &weight_shape, const ShapeVector &group_list_shape) {
  auto x_shape_size = x_shape.size();
  auto weight_shape_size = weight_shape.size();
  auto group_list_shape_size = group_list_shape.size();
  if (MS_UNLIKELY(x_shape_size != kDim2)) {
    MS_LOG(EXCEPTION) << "For GroupedMatmul, input 'x' must be 2D, but got:" << x_shape_size;
  }
  if (MS_UNLIKELY(weight_shape_size != kDim3)) {
    MS_LOG(EXCEPTION) << "For GroupedMatmul, input 'weight' must be 3D, but got:" << weight_shape_size;
  }
  if (MS_UNLIKELY(group_list_shape_size != kDim1)) {
    MS_LOG(EXCEPTION) << "For GroupedMatmul, input 'group_list' must be 1D, but got:" << group_list_shape_size;
  }
}

ShapeVector InferGroupedMatmulOutputShape(const ShapeVector &x_shape, const ShapeVector &weight_shape,
                                          const ShapeVector &group_list_shape, bool transpose_a, bool transpose_b,
                                          TypeId x_dtype) {
  CheckInputRank(x_shape, weight_shape, group_list_shape);
  // get input dimensions
  auto x_m_dim = transpose_a ? x_shape[kIndex1] : x_shape[kIndex0];
  auto x_k_dim = transpose_a ? x_shape[kIndex0] : x_shape[kIndex1];
  auto weight_e_dim = weight_shape[kIndex0];
  auto weight_n_dim = transpose_b ? weight_shape[kIndex1] : weight_shape[kIndex2];
  auto weight_k_dim = transpose_b ? weight_shape[kIndex2] : weight_shape[kIndex1];

  // infer output shape when weight_n_dim is dynamic
  if (weight_n_dim == abstract::Shape::kShapeDimAny) {
    return ShapeVector{x_m_dim, weight_n_dim};
  }

  if (MS_UNLIKELY(x_k_dim != abstract::Shape::kShapeDimAny && weight_k_dim != abstract::Shape::kShapeDimAny &&
                  x_k_dim != weight_k_dim)) {
    MS_LOG(EXCEPTION) << "For GroupedMatmul, input 'x' and 'weight' must have the same dimension of 'k', but got:"
                      << x_k_dim << " and " << weight_k_dim;
  }
  // Check shape alignment requirements
  // Check K dimension alignment
  if (x_k_dim != abstract::Shape::kShapeDimAny) {
    const auto kFloat16KAlign16 = 16;
    const auto kInt8KAlign32 = 32;
    int64_t k_align = (x_dtype == kNumberTypeFloat16) ? kFloat16KAlign16 : kInt8KAlign32;
    if (x_k_dim % k_align != 0) {
      MS_LOG(EXCEPTION) << "For GroupedMatmul, input 'x' K dimension must be aligned to " << k_align << " for "
                        << (x_dtype == kNumberTypeFloat16 ? "float16" : "int8") << " input, but got: " << x_k_dim;
    }
  }

  // Check N dimension alignment
  if (MS_UNLIKELY(weight_n_dim != abstract::Shape::kShapeDimAny && weight_n_dim % 16 != 0)) {
    MS_LOG(EXCEPTION) << "For GroupedMatmul, input 'weight' N dimension must be aligned to 16, but got: "
                      << weight_n_dim;
  }

  // Check E dimension alignment
  if (MS_UNLIKELY(weight_e_dim != abstract::Shape::kShapeDimAny && weight_e_dim % 8 != 0)) {
    MS_LOG(EXCEPTION) << "For GroupedMatmul, input 'weight' E dimension (group count) must be aligned to 8, but got: "
                      << weight_e_dim;
  }

  return ShapeVector{x_m_dim, weight_n_dim};
}

class OPS_API CustomGroupedMatmulOpFuncImpl : public OpFuncImpl {
 public:
  ShapeArray InferShape(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const override {
    // dynamic rank
    if (MS_UNLIKELY(input_infos[static_cast<size_t>(GroupedMatmulInputIndex::kGmmXIndex)]->IsDynamicRank() ||
                    input_infos[static_cast<size_t>(GroupedMatmulInputIndex::kGmmWeightIndex)]->IsDynamicRank())) {
      auto output_shape = ShapeVector{abstract::Shape::kShapeRankAny};
      return {output_shape};
    }
    // check input size
    if (MS_UNLIKELY(input_infos.size() != static_cast<size_t>(GroupedMatmulInputIndex::kGmmInputsNum))) {
      MS_LOG(EXCEPTION) << "For GroupedMatmul, input size must be "
                        << static_cast<size_t>(GroupedMatmulInputIndex::kGmmInputsNum) << ", but got "
                        << input_infos.size();
    }

    // get input shapes
    auto x_shape = input_infos[static_cast<size_t>(GroupedMatmulInputIndex::kGmmXIndex)]->GetShape();
    auto weight_shape = input_infos[static_cast<size_t>(GroupedMatmulInputIndex::kGmmWeightIndex)]->GetShape();
    auto group_list_shape = input_infos[static_cast<size_t>(GroupedMatmulInputIndex::kGmmGroupListIndex)]->GetShape();
    auto transpose_a =
      input_infos[static_cast<size_t>(GroupedMatmulInputIndex::kGmmTransposeAIndex)]->GetScalarValueWithCheck<bool>();
    auto transpose_b =
      input_infos[static_cast<size_t>(GroupedMatmulInputIndex::kGmmTransposeBIndex)]->GetScalarValueWithCheck<bool>();
    auto x_dtype = input_infos[static_cast<size_t>(GroupedMatmulInputIndex::kGmmXIndex)]->GetType();
    auto output_shape =
      InferGroupedMatmulOutputShape(x_shape, weight_shape, group_list_shape, transpose_a, transpose_b, x_dtype);
    return {output_shape};
  }

  std::vector<TypeId> InferType(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const override {
    MS_EXCEPTION_IF_NULL(primitive);
    auto ms_context = MsContext::GetInstance();
    MS_EXCEPTION_IF_NULL(ms_context);
    const auto &device_target = ms_context->ascend_soc_version();
    if (device_target != kAscendVersion310p) {
      MS_LOG(EXCEPTION) << "For GroupedMatmul, only support Ascend device, but got:" << device_target;
    }
    auto x_dtype = input_infos[static_cast<size_t>(GroupedMatmulInputIndex::kGmmXIndex)]->GetType();
    auto weight_dtype = input_infos[static_cast<size_t>(GroupedMatmulInputIndex::kGmmWeightIndex)]->GetType();
    auto group_list_dtype = input_infos[static_cast<size_t>(GroupedMatmulInputIndex::kGmmGroupListIndex)]->GetType();
    // check input types
    if (MS_UNLIKELY(x_dtype != kNumberTypeFloat16 && x_dtype != kNumberTypeInt8)) {
      MS_LOG(EXCEPTION) << "For GroupedMatmul, input 'x' must be float16 or int8, but got:" << x_dtype;
    }
    if (MS_UNLIKELY(weight_dtype != x_dtype)) {
      MS_LOG(EXCEPTION) << "For GroupedMatmul, input 'weight' must be the same type as 'x', but got:" << weight_dtype;
    }
    if (MS_UNLIKELY(group_list_dtype != kNumberTypeInt32)) {
      MS_LOG(EXCEPTION) << "For GroupedMatmul, input 'group_list' must be int32, but got:" << group_list_dtype;
    }
    auto check_optional_input_valid_dtypes = [](const InferInfoPtr &input_info, std::vector<TypeId> expected_types,
                                                const std::string &input_name) {
      if (input_info->IsNone()) {
        return;
      }
      if (std::find(expected_types.begin(), expected_types.end(), input_info->GetType()) == expected_types.end()) {
        MS_LOG(EXCEPTION) << "For GroupedMatmul, input '" << input_name << "' must be one of " << expected_types
                          << ", but got:" << input_info->GetType();
      }
    };
    check_optional_input_valid_dtypes(input_infos[static_cast<size_t>(GroupedMatmulInputIndex::kGmmBiasIndex)],
                                      {kNumberTypeFloat16, kNumberTypeInt32}, "bias");
    check_optional_input_valid_dtypes(input_infos[static_cast<size_t>(GroupedMatmulInputIndex::kGmmScaleIndex)],
                                      {kNumberTypeInt64, kNumberTypeUInt64, kNumberTypeFloat32}, "scale");
    check_optional_input_valid_dtypes(input_infos[static_cast<size_t>(GroupedMatmulInputIndex::kGmmPerTokenScaleIndex)],
                                      {kNumberTypeFloat32}, "per_token_scale");
    check_optional_input_valid_dtypes(
      input_infos[static_cast<size_t>(GroupedMatmulInputIndex::kGmmAntiquantScaleIndex)],
      {kNumberTypeFloat16, kNumberTypeFloat32}, "antiquant_scale");

    // infer output types
    auto output_dtype = kNumberTypeFloat16;
    return {output_dtype};
  }

  bool GeneralInferRegistered() const override { return true; }
};

class CustomGroupedMatmul : public InternalKernelMod {
 public:
  CustomGroupedMatmul() : InternalKernelMod() {}
  ~CustomGroupedMatmul() = default;

  void InitKernelInputsOutputsIndex() override {
    kernel_inputs_index_ = {static_cast<size_t>(GroupedMatmulInputIndex::kGmmXIndex),
                            static_cast<size_t>(GroupedMatmulInputIndex::kGmmWeightIndex),
                            static_cast<size_t>(GroupedMatmulInputIndex::kGmmBiasIndex),
                            static_cast<size_t>(GroupedMatmulInputIndex::kGmmScaleIndex),
                            static_cast<size_t>(GroupedMatmulInputIndex::kGmmGroupListIndex),
                            static_cast<size_t>(GroupedMatmulInputIndex::kGmmPerTokenScaleIndex),
                            static_cast<size_t>(GroupedMatmulInputIndex::kGmmAntiquantScaleIndex)};
    kernel_outputs_index_ = {static_cast<size_t>(GroupedMatmulOutputIndex::kGmmOutputIndex)};
  }

 protected:
  internal_v2::InternalOpPtr CreateKernel(const internal_v2::InputsImmutableInfoList &inputs,
                                          const internal_v2::OutputsImmutableInfoList &outputs,
                                          const std::vector<KernelTensor *> &ms_inputs,
                                          const std::vector<KernelTensor *> &ms_outputs) override {
    param_.transpose_a =
      ms_inputs.at(static_cast<size_t>(GroupedMatmulInputIndex::kGmmTransposeAIndex))->GetValueWithCheck<bool>();
    param_.transpose_b =
      ms_inputs.at(static_cast<size_t>(GroupedMatmulInputIndex::kGmmTransposeBIndex))->GetValueWithCheck<bool>();
    param_.with_bias =
      !(ms_inputs.at(static_cast<size_t>(GroupedMatmulInputIndex::kGmmBiasIndex))->GetType()->isa<TypeNone>());
    param_.enable_shuffle = false;  // the real definition is in internal
    auto inputs_clone = inputs;
    auto x_format = static_cast<DataFormat>(
      ms_inputs.at(static_cast<size_t>(GroupedMatmulInputIndex::kGmmXFormatIndex))->GetValueWithCheck<int64_t>());
    if (x_format == DataFormat::FRACTAL_NZ) {
      inputs_clone[static_cast<size_t>(GroupedMatmulInputIndex::kGmmXIndex)].SetFormat(
        internal_v2::TensorFormat::kFormatFRACTAL_NZ);
    }
    inputs_clone[static_cast<size_t>(GroupedMatmulInputIndex::kGmmWeightIndex)].SetFormat(
      internal_v2::TensorFormat::kFormatFRACTAL_NZ);
    return internal_v2::CreateGroupedMatmulOp(inputs_clone, outputs, param_, internal_v2::kInternalGroupedMatmulOpName);
  }

  uint64_t GenerateTilingKey(const std::vector<KernelTensor *> &inputs) override {
    return InternalTilingCache::GenerateKey(kernel_name_, inputs, param_);
  }

 private:
  internal_v2::MatmulParam param_;
};
}  // namespace ms_custom_ops

REG_GRAPH_MODE_OP(grouped_matmul, ms_custom_ops::CustomGroupedMatmulOpFuncImpl, ms_custom_ops::CustomGroupedMatmul);

// =============================================================================
// PYBOOST MODE IMPLEMENTATION
// =============================================================================

#include "ops/framework/ms_kernels_internal/pyboost/internal_pyboost_runner.h"

namespace ms_custom_ops {
class GroupedMatmulRunner : public InternalPyboostRunner {
 public:
  using InternalPyboostRunner::InternalPyboostRunner;

  void SetParam(const bool &transpose_a, const bool &transpose_b, const bool &with_bias) {
    param_.transpose_a = transpose_a;
    param_.transpose_b = transpose_b;
    param_.with_bias = with_bias;
    param_.enable_shuffle = false;  // the real definition is in internal
  }
  void SetXFormat(const DataFormat &x_format) { this->x_format_ = x_format; }

 protected:
  internal_v2::InternalOpPtr CreateKernel(const internal_v2::InputsImmutableInfoList &inputs,
                                          const internal_v2::OutputsImmutableInfoList &outputs) override {
    auto inputs_clone = inputs;
    if (x_format_ == DataFormat::FRACTAL_NZ) {
      inputs_clone[static_cast<size_t>(GroupedMatmulInputIndex::kGmmXIndex)].SetFormat(
        internal_v2::TensorFormat::kFormatFRACTAL_NZ);
    }
    inputs_clone[static_cast<size_t>(GroupedMatmulInputIndex::kGmmWeightIndex)].SetFormat(
      internal_v2::TensorFormat::kFormatFRACTAL_NZ);
    return internal_v2::CreateGroupedMatmulOp(inputs_clone, outputs, param_, internal_v2::kInternalGroupedMatmulOpName);
  }

 private:
  internal_v2::MatmulParam param_;
  DataFormat x_format_{DataFormat::ND};
};

std::vector<ms::Tensor> npu_grouped_matmul(const ms::Tensor &x, const ms::Tensor &weight, const ms::Tensor &group_list,
                                           const std::optional<ms::Tensor> &bias,
                                           const std::optional<ms::Tensor> &scale,
                                           const std::optional<ms::Tensor> &per_token_scale,
                                           const std::optional<ms::Tensor> &antiquant_scale, const bool &transpose_a,
                                           const bool &transpose_b, const int &x_format) {
  auto op_name = "GroupedMatmul";
  auto runner = std::make_shared<GroupedMatmulRunner>(op_name);
  MS_EXCEPTION_IF_NULL(runner);

  // Set param for internal kernel
  runner->SetParam(transpose_a, transpose_b, bias.has_value());
  runner->SetXFormat(static_cast<DataFormat>(x_format));
  // Setup the runner with all parameters (including hash calculation)
  runner->Setup(op_name, x, weight, group_list, bias, scale, per_token_scale, antiquant_scale, transpose_a,
                transpose_b);

  // if you need infer shape and type, you can use this
  std::vector<ms::Tensor> inputs = {x,
                                    weight,
                                    GetTensorOrEmpty(bias),
                                    GetTensorOrEmpty(scale),
                                    group_list,
                                    GetTensorOrEmpty(per_token_scale),
                                    GetTensorOrEmpty(antiquant_scale)};
  auto out_shape = InferGroupedMatmulOutputShape(x.shape(), weight.shape(), group_list.shape(), transpose_a,
                                                 transpose_b, x.data_type());
  std::vector<ms::Tensor> outputs = {ms::Tensor(TypeId::kNumberTypeFloat16, out_shape)};
  runner->GetOrCreateKernel(inputs, outputs);
  runner->Run(inputs, outputs);
  return outputs;
}
}  // namespace ms_custom_ops

auto pyboost_grouped_matmul(const ms::Tensor &x, const ms::Tensor &weight, const ms::Tensor &group_list,
                            const std::optional<ms::Tensor> &bias, const std::optional<ms::Tensor> &scale,
                            const std::optional<ms::Tensor> &per_token_scale,
                            const std::optional<ms::Tensor> &antiquant_scale, const bool &transpose_a,
                            const bool &transpose_b, const int &x_format) {
  return ms::pynative::PyboostRunner::Call<ms_custom_ops::kNumber1>(
    ms_custom_ops::npu_grouped_matmul, x, weight, group_list, bias, scale, per_token_scale, antiquant_scale,
    transpose_a, transpose_b, x_format);
}

MS_CUSTOM_OPS_EXTENSION_MODULE(m) {
  m.def("grouped_matmul", &pyboost_grouped_matmul, "GroupedMatmul", pybind11::arg("x"), pybind11::arg("weight"),
        pybind11::arg("group_list"), pybind11::arg("bias") = std::nullopt, pybind11::arg("scale") = std::nullopt,
        pybind11::arg("per_token_scale") = std::nullopt, pybind11::arg("antiquant_scale") = std::nullopt,
        pybind11::arg("transpose_a") = false, pybind11::arg("transpose_b") = false, pybind11::arg("x_format") = 0);
}
