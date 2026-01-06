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

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "ops/framework/utils.h"
#include "ops/framework/ms_kernels_internal/graphmode/internal_kernel_mod.h"
#include "ops/c_api/utils/check_utils.h"

namespace ms_custom_ops {

// ============================================================================
// TransData-specific Types
// ============================================================================

enum class InputIndex : size_t {
  kInputIndex = 0,
  kTransDataFormatIndex = 1,
};

enum class OutputIndex : size_t { kOutputIndex = 0 };

// Helper function to extract int64 value from KernelTensor with type validation
inline int32_t GetInt64ValueAsInt32(KernelTensor *tensor, const std::string &param_name) {
  auto type_id = tensor->dtype_id();
  // Accept both concrete int64 and abstract Number type (graph build phase may have kObjectTypeNumber)
  if (type_id != mindspore::TypeId::kNumberTypeInt64 && type_id != mindspore::TypeId::kObjectTypeNumber) {
    MS_LOG(EXCEPTION) << "TransData [" << param_name << "]'s dtype wrong, expect int64 or Number, but got: " << type_id;
  }
  return static_cast<int32_t>(tensor->GetValue<int64_t>().value());
}

// Helper function to extract and validate transdata_type from KernelTensor
inline TransDataFormat GetAndValidateTransDataType(const int32_t &type_val) {
  // Validate that transdata_type is either FRACTAL_NZ_TO_ND (0) or ND_TO_FRACTAL_NZ (1)
  if (type_val != static_cast<int32_t>(TransDataFormat::FRACTAL_NZ_TO_ND) &&
      type_val != static_cast<int32_t>(TransDataFormat::ND_TO_FRACTAL_NZ)) {
    MS_LOG(EXCEPTION) << "Invalid transdata_type val: " << type_val
                      << ". Valid values are: 0 (FRACTAL_NZ_TO_ND) or 1 (ND_TO_FRACTAL_NZ)";
  }

  return static_cast<TransDataFormat>(type_val);
}

inline internal_v2::InternalOpPtr CreateTransDataOpWithParam(const internal_v2::InputsImmutableInfoList &inputs,
                                                             const internal_v2::OutputsImmutableInfoList &outputs,
                                                             TransDataFormat transdata_type) {
  internal_v2::TransDataParam param;

  // Map transdata_type to internal_v2 enum and set appropriate input format
  auto inputs_clone = inputs;
  auto outputs_clone = outputs;

  if (transdata_type == TransDataFormat::FRACTAL_NZ_TO_ND) {
    param.transdataType = internal_v2::TransDataParam::FRACTAL_NZ_TO_ND;
    // For FRACTAL_NZ_TO_ND: input should be FRACTAL_NZ format
    inputs_clone[kIndex0].SetFormat(internal_v2::TensorFormat::kFormatFRACTAL_NZ);
    outputs_clone[kIndex0].SetFormat(internal_v2::TensorFormat::kFormatND);
  } else if (transdata_type == TransDataFormat::ND_TO_FRACTAL_NZ) {
    param.transdataType = internal_v2::TransDataParam::ND_TO_FRACTAL_NZ;
    // For ND_TO_FRACTAL_NZ: input should be ND format
    inputs_clone[kIndex0].SetFormat(internal_v2::TensorFormat::kFormatND);
    outputs_clone[kIndex0].SetFormat(internal_v2::TensorFormat::kFormatFRACTAL_NZ);
  } else {
    MS_LOG(EXCEPTION) << "TransData: Invalid transdata_type " << static_cast<int32_t>(transdata_type)
                      << ", valid values are: 0 (FRACTAL_NZ_TO_ND), 1 (ND_TO_FRACTAL_NZ)";
  }

  param.specialTransdata = internal_v2::TransDataParam::NORMAL;

  return internal_v2::CreateTransDataOp(inputs_clone, outputs_clone, param, internal_v2::kInternalTransDataOpName);
}

// =============================================================================
// GRAPH MODE IMPLEMENTATION
// =============================================================================

class OPS_API CustomTransDataOpFuncImpl : public OpFuncImpl {
 public:
  ShapeArray InferShape(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const override {
    return {input_infos[static_cast<size_t>(InputIndex::kInputIndex)]->GetShape()};
  }

  std::vector<TypeId> InferType(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const override {
    return {input_infos[static_cast<size_t>(InputIndex::kInputIndex)]->GetType()};
  }

  bool GeneralInferRegistered() const override { return true; }
};

class CustomTransData : public InternalKernelMod {
 public:
  CustomTransData() : InternalKernelMod(), skip_execution_(false) {}
  ~CustomTransData() = default;

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override {
    bool result = InternalKernelMod::Init(inputs, outputs);

    auto transdata_type = inputs.at(static_cast<size_t>(InputIndex::kTransDataFormatIndex));
    auto transdata_type_val = GetAndValidateTransDataType(GetInt64ValueAsInt32(transdata_type, "transdata_type"));

    if (transdata_type_val == TransDataFormat::ND_TO_FRACTAL_NZ) {
      // In graph mode, for ND_TO_FRACTAL_NZ, we need set NzOutputIndex to malloc output.
      ClearNzOutputIndices();
      AddNzOutputIndex(static_cast<size_t>(OutputIndex::kOutputIndex));
    }

    return result;
  }

  void InitKernelInputsOutputsIndex() override {
    kernel_inputs_index_ = {static_cast<size_t>(InputIndex::kInputIndex)};
    kernel_outputs_index_ = {static_cast<size_t>(OutputIndex::kOutputIndex)};
  }

  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override {
    // Check if any input has shape containing 0
    for (const auto &input : inputs) {
      if (input == nullptr) continue;
      auto shape = input->GetShapeVector();
      bool has_zero = std::any_of(shape.begin(), shape.end(), [](const auto &dim) { return dim == 0; });
      if (has_zero) {
        MS_LOG(INFO) << "TransData: Skipping execution due to zero dimension in input shape: " << shape;
        skip_execution_ = true;
        return KernelMod::Resize(inputs, outputs);  // Skip execution
      }
    }

    skip_execution_ = false;
    return InternalKernelMod::Resize(inputs, outputs);
  }

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs, void *stream_ptr) override {
    if (skip_execution_) {
      return true;
    }

    return InternalKernelMod::Launch(inputs, workspace, outputs, stream_ptr);
  }

 protected:
  internal_v2::InternalOpPtr CreateKernel(const internal_v2::InputsImmutableInfoList &inputs,
                                          const internal_v2::OutputsImmutableInfoList &outputs,
                                          const std::vector<KernelTensor *> &ms_inputs,
                                          const std::vector<KernelTensor *> &ms_outputs) override {
    auto transdata_type = ms_inputs.at(static_cast<size_t>(InputIndex::kTransDataFormatIndex));
    auto transdata_type_val = GetAndValidateTransDataType(GetInt64ValueAsInt32(transdata_type, "transdata_type"));

    return CreateTransDataOpWithParam(inputs, outputs, transdata_type_val);
  }

 private:
  bool skip_execution_{false};  // Flag to skip execution when shape contains 0
};
}  // namespace ms_custom_ops

REG_GRAPH_MODE_OP(trans_data, ms_custom_ops::CustomTransDataOpFuncImpl, ms_custom_ops::CustomTransData);

// =============================================================================
// PYBOOST MODE IMPLEMENTATION
// =============================================================================

#include "ops/framework/ms_kernels_internal/pyboost/internal_pyboost_runner.h"

namespace ms_custom_ops {
class TransDataRunner : public InternalPyboostRunner {
 public:
  using InternalPyboostRunner::InternalPyboostRunner;

  void SetTransDataFormat(TransDataFormat transdata_type) { this->transdata_type_ = transdata_type; }

 protected:
  internal_v2::InternalOpPtr CreateKernel(const internal_v2::InputsImmutableInfoList &inputs,
                                          const internal_v2::OutputsImmutableInfoList &outputs) override {
    return CreateTransDataOpWithParam(inputs, outputs, this->transdata_type_);
  }

 private:
  TransDataFormat transdata_type_{TransDataFormat::FRACTAL_NZ_TO_ND};
};

ms::Tensor npu_trans_data(const ms::Tensor &input, std::optional<int64_t> transdata_type) {
  auto op_name = "TransData";
  auto runner = std::make_shared<ms_custom_ops::TransDataRunner>(op_name);
  MS_EXCEPTION_IF_NULL(runner);
  // Validate input shape's H, W dimensions are aligned for FRACTAL_NZ format
  CheckShapeHWAlignment(input.shape(), input.data_type());

  auto transdata_type_val = GetAndValidateTransDataType(transdata_type.value_or(0));
  runner->SetTransDataFormat(transdata_type_val);
  // Setup the runner with all parameters (including hash calculation)
  runner->Setup(op_name, input, transdata_type);
  auto output = ms::Tensor(input.data_type(), input.shape());

  // Create input and output tensors
  std::vector<ms::Tensor> inputs = {input};
  std::vector<ms::Tensor> outputs = {output};
  runner->GetOrCreateKernel(inputs, outputs);
  runner->Run(inputs, outputs);
  return output;
}
}  // namespace ms_custom_ops

auto pyboost_trans_data(const ms::Tensor &input, std::optional<int64_t> transdata_type) {
  return ms::pynative::PyboostRunner::Call<ms_custom_ops::kNumber1>(ms_custom_ops::npu_trans_data, input,
                                                                    transdata_type);
}

MS_CUSTOM_OPS_EXTENSION_MODULE(m) {
  m.def("trans_data", &pyboost_trans_data, "Trans Data", pybind11::arg("input"), pybind11::arg("transdata_type") = 0);
}
