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

#include <cstddef>

#include "ops/framework/ms_kernels_internal/graphmode/internal_kernel_mod.h"
#include "ops/framework/utils.h"

namespace ms_custom_ops {

constexpr size_t kMatmulMatSize = 2;

enum class MatmulInputIndex : size_t {
  kMatmulInputX1Index = 0,
  kMatmulInputX2Index,
  kMatmulInputTransposeAIndex,
  kMatmulInputTransposeBIndex,
  kMatmulInputX1FormatIndex,
  kMatmulInputX2FormatIndex,
  kMatmulInputOutputFormatIndex,
  kMatmulInputsNum,
};

enum class MatmulOutputIndex : size_t { kMatmulOutputIndex = 0 };

ShapeVector MatMulMakeShape(const ShapeVector x1_shape, const ShapeVector x2_shape,
                            bool transpose_a, bool transpose_b, size_t offset) {
  if (x1_shape.size() != kMatmulMatSize || x2_shape.size() != kMatmulMatSize) {
    MS_LOG(EXCEPTION) << "For 'MatMul', the dimension of 'x1' and 'x2' should be 2, "
                      << "but got " << x1_shape.size() << " and " << x2_shape.size();
  }
  ShapeVector out_shape;
  auto x_col = x1_shape[(transpose_a ? kIndex0 : kIndex1)];
  auto y_row = x2_shape[(transpose_b ? kIndex1 : kIndex0)];
  if (x_col != y_row && x_col >= 0 && y_row >= 0) {
    MS_LOG(EXCEPTION) << "For 'MatMul' the input dimensions must be equal, "
                      << "but got 'x1_col': " << x_col << " and 'x2_row': " << y_row << ".";
  }

  ShapeVector ret_shape;
  auto make_shape = [&transpose_a, &transpose_b](ShapeVector &output,
                                                  const ShapeVector xshp,
                                                  const ShapeVector yshp) -> void {
    if (!xshp.empty() && !yshp.empty()) {
      output.push_back(xshp[(transpose_a ? kIndex1 : kIndex0)]);
      output.push_back(yshp[(transpose_b ? kIndex0 : kIndex1)]);
    }
    return;
  };
  make_shape(ret_shape, x1_shape, x2_shape);
  return ret_shape;
}

inline internal_v2::InternalOpPtr CreateMatmulOpWithParam(
    const internal_v2::InputsImmutableInfoList &inputs,
    const internal_v2::OutputsImmutableInfoList &outputs, const bool &transpose_a,
    const bool &transpose_b, const DataFormat &x1_format, const DataFormat &x2_format,
    const DataFormat &output_format) {
  internal_v2::MatmulParam param;
  param.transpose_a = transpose_a;
  param.transpose_b = transpose_b;

  // Map format to internal_v2 enum and set appropriate format
  auto inputs_clone = inputs;
  auto outputs_clone = outputs;

  inputs_clone[static_cast<size_t>(MatmulInputIndex::kMatmulInputX1Index)].SetFormat(
      internal_v2::TensorFormat::kFormatND);
  inputs_clone[static_cast<size_t>(MatmulInputIndex::kMatmulInputX2Index)].SetFormat(
      internal_v2::TensorFormat::kFormatND);
  outputs_clone[static_cast<size_t>(MatmulOutputIndex::kMatmulOutputIndex)].SetFormat(
      internal_v2::TensorFormat::kFormatND);
  if (x1_format == DataFormat::FRACTAL_NZ) {
    inputs_clone[static_cast<size_t>(MatmulInputIndex::kMatmulInputX1Index)].SetFormat(
        internal_v2::TensorFormat::kFormatFRACTAL_NZ);
  }
  if (x2_format == DataFormat::FRACTAL_NZ) {
    inputs_clone[static_cast<size_t>(MatmulInputIndex::kMatmulInputX2Index)].SetFormat(
        internal_v2::TensorFormat::kFormatFRACTAL_NZ);
  }
  if (output_format == DataFormat::FRACTAL_NZ) {
    outputs_clone[static_cast<size_t>(MatmulOutputIndex::kMatmulOutputIndex)].SetFormat(
        internal_v2::TensorFormat::kFormatFRACTAL_NZ);
  }

  return internal_v2::CreateMatmulOp(inputs_clone, outputs_clone, param,
                                      internal_v2::kInternalMatMulOpName);
}

class OPS_API CustomMatMulOpFuncImpl : public OpFuncImpl {
 public:
  ShapeArray InferShape(const PrimitivePtr &primitive,
                        const InferInfoPtrList &input_infos) const override {
    auto x1_shape =
        input_infos[static_cast<size_t>(MatmulInputIndex::kMatmulInputX1Index)]->GetShape();
    auto x2_shape =
        input_infos[static_cast<size_t>(MatmulInputIndex::kMatmulInputX2Index)]->GetShape();
    if (IsDynamicRank(x1_shape) || IsDynamicRank(x2_shape)) {
      return {ShapeVector({abstract::Shape::kShapeRankAny})};
    }
    bool transpose_a = input_infos[static_cast<size_t>(
        MatmulInputIndex::kMatmulInputTransposeAIndex)]->GetScalarValueWithCheck<bool>();
    bool transpose_b = input_infos[static_cast<size_t>(
        MatmulInputIndex::kMatmulInputTransposeBIndex)]->GetScalarValueWithCheck<bool>();
    ShapeVector out_shape =
        MatMulMakeShape(x1_shape, x2_shape, transpose_a, transpose_b, kMatmulMatSize);
    return {out_shape};
  }

  std::vector<TypeId> InferType(const PrimitivePtr &primitive,
                                const InferInfoPtrList &input_infos) const override {
    // Use the first input's type as output type
    return {input_infos[static_cast<size_t>(
        MatmulInputIndex::kMatmulInputX1Index)]->GetType()};
  }

  bool GeneralInferRegistered() const override { return true; }
};

class CustomMatMul : public InternalKernelMod {
 public:
  CustomMatMul() : InternalKernelMod() {}
  ~CustomMatMul() = default;

  void InitKernelInputsOutputsIndex() override {
    kernel_inputs_index_ = {
        static_cast<size_t>(MatmulInputIndex::kMatmulInputX1Index),
        static_cast<size_t>(MatmulInputIndex::kMatmulInputX2Index)};
    kernel_outputs_index_ = {
        static_cast<size_t>(MatmulOutputIndex::kMatmulOutputIndex)};
  }

  bool Init(const std::vector<KernelTensor *> &inputs,
            const std::vector<KernelTensor *> &outputs) override {
    bool result = InternalKernelMod::Init(inputs, outputs);

    auto output_format =
        inputs.at(static_cast<size_t>(MatmulInputIndex::kMatmulInputOutputFormatIndex));
    auto output_format_val =
        static_cast<DataFormat>(output_format->GetValueWithCheck<int64_t>());

    if (output_format_val == DataFormat::FRACTAL_NZ) {
      ClearNzOutputIndices();
      AddNzOutputIndex(static_cast<size_t>(MatmulOutputIndex::kMatmulOutputIndex));
    }

    return result;
  }

 protected:
  internal_v2::InternalOpPtr CreateKernel(
      const internal_v2::InputsImmutableInfoList &inputs,
      const internal_v2::OutputsImmutableInfoList &outputs,
      const std::vector<KernelTensor *> &ms_inputs,
      const std::vector<KernelTensor *> &ms_outputs) override {
    auto transpose_a = ms_inputs.at(static_cast<size_t>(
        MatmulInputIndex::kMatmulInputTransposeAIndex))->GetValueWithCheck<bool>();
    auto transpose_b = ms_inputs.at(static_cast<size_t>(
        MatmulInputIndex::kMatmulInputTransposeBIndex))->GetValueWithCheck<bool>();
    this->x1_format_ = static_cast<DataFormat>(
        ms_inputs.at(static_cast<size_t>(MatmulInputIndex::kMatmulInputX1FormatIndex))
            ->GetValueWithCheck<int64_t>());
    this->x2_format_ = static_cast<DataFormat>(
        ms_inputs.at(static_cast<size_t>(MatmulInputIndex::kMatmulInputX2FormatIndex))
            ->GetValueWithCheck<int64_t>());
    this->output_format_ = static_cast<DataFormat>(
        ms_inputs.at(static_cast<size_t>(MatmulInputIndex::kMatmulInputOutputFormatIndex))
            ->GetValueWithCheck<int64_t>());
    return CreateMatmulOpWithParam(inputs, outputs, transpose_a, transpose_b,
                                    this->x1_format_, this->x2_format_,
                                    this->output_format_);
  }

  uint64_t GenerateTilingKey(const std::vector<KernelTensor *> &inputs) override {
    return InternalTilingCache::GenerateKey(kernel_name_, inputs, this->x1_format_,
                                             this->x2_format_, this->output_format_);
  }

 private:
  DataFormat x1_format_{DataFormat::ND};
  DataFormat x2_format_{DataFormat::ND};
  DataFormat output_format_{DataFormat::ND};
};
}  // namespace ms_custom_ops

REG_GRAPH_MODE_OP(mat_mul, ms_custom_ops::CustomMatMulOpFuncImpl,
                  ms_custom_ops::CustomMatMul);

// =============================================================================
// PYBOOST MODE IMPLEMENTATION
// =============================================================================

#include "ops/framework/ms_kernels_internal/pyboost/internal_pyboost_runner.h"

namespace ms_custom_ops {
class MatMulRunner : public InternalPyboostRunner {
 public:
  using InternalPyboostRunner::InternalPyboostRunner;

  void SetTransposeA(const bool &transpose_a) { this->transpose_a_ = transpose_a; }
  void SetTransposeB(const bool &transpose_b) { this->transpose_b_ = transpose_b; }
  void SetX1Format(const DataFormat &x1_format) { this->x1_format_ = x1_format; }
  void SetX2Format(const DataFormat &x2_format) { this->x2_format_ = x2_format; }
  void SetOutputFormat(const DataFormat &output_format) { this->output_format_ = output_format; }

 protected:
  internal_v2::InternalOpPtr CreateKernel(
      const internal_v2::InputsImmutableInfoList &inputs,
      const internal_v2::OutputsImmutableInfoList &outputs) override {
    return CreateMatmulOpWithParam(inputs, outputs, this->transpose_a_,
                                   this->transpose_b_, this->x1_format_,
                                   this->x2_format_, this->output_format_);
  }

 private:
  bool transpose_a_{false};
  bool transpose_b_{false};
  DataFormat x1_format_{DataFormat::ND};
  DataFormat x2_format_{DataFormat::ND};
  DataFormat output_format_{DataFormat::ND};
};

ms::Tensor npu_matmul(const ms::Tensor &x1, const ms::Tensor &x2,
                      std::optional<bool> transpose_a, std::optional<bool> transpose_b,
                      std::optional<int64_t> x1_format, std::optional<int64_t> x2_format,
                      std::optional<int64_t> output_format) {
  auto op_name = "MatMul";
  auto runner = std::make_shared<ms_custom_ops::MatMulRunner>(op_name);
  MS_EXCEPTION_IF_NULL(runner);

  runner->SetTransposeA(transpose_a.value_or(false));
  runner->SetTransposeB(transpose_b.value_or(false));
  runner->SetX1Format(static_cast<DataFormat>(x1_format.value_or(0)));
  runner->SetX2Format(static_cast<DataFormat>(x2_format.value_or(0)));
  runner->SetOutputFormat(static_cast<DataFormat>(output_format.value_or(0)));

  // Setup the runner with all parameters (including hash calculation)
  runner->Setup(op_name, x1, x2, transpose_a.value_or(false), transpose_b.value_or(false),
                x1_format.value_or(0), x2_format.value_or(0), output_format.value_or(0));

  // Infer output shape and type
  auto transpose_a_val = transpose_a.value_or(false);
  auto transpose_b_val = transpose_b.value_or(false);
  auto output_shape = MatMulMakeShape(x1.shape(), x2.shape(), transpose_a_val,
                                       transpose_b_val, kMatmulMatSize);
  if (output_format.has_value() &&
      static_cast<DataFormat>(output_format.value()) == DataFormat::FRACTAL_NZ) {
    CheckShapeHWAlignment(output_shape, x1.data_type());
  }
  std::vector<ms::Tensor> inputs = {x1, x2};
  std::vector<ms::Tensor> outputs = {ms::Tensor(x1.data_type(), output_shape)};
  runner->GetOrCreateKernel(inputs, outputs);
  runner->Run(inputs, outputs);
  return outputs[0];
}
}  // namespace ms_custom_ops

auto pyboost_matmul(const ms::Tensor &x1, const ms::Tensor &x2,
                    std::optional<bool> transpose_a, std::optional<bool> transpose_b,
                    std::optional<int64_t> x1_format, std::optional<int64_t> x2_format,
                    std::optional<int64_t> output_format) {
  return ms::pynative::PyboostRunner::Call<1>(ms_custom_ops::npu_matmul, x1, x2, transpose_a,
                                              transpose_b, x1_format, x2_format,
                                              output_format);
}

MS_CUSTOM_OPS_EXTENSION_MODULE(m) {
  m.def("mat_mul", &pyboost_matmul, "MatMul", pybind11::arg("x1"), pybind11::arg("x2"),
        pybind11::arg("transpose_a") = false, pybind11::arg("transpose_b") = false,
        pybind11::arg("x1_format") = 0, pybind11::arg("x2_format") = 0,
        pybind11::arg("output_format") = 0);
}
