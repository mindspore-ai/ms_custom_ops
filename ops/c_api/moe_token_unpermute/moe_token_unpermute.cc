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

#include <map>
#include <string>
#include <utility>
#include <memory>
#include <vector>

#include "ops/framework/ms_kernels_internal/graphmode/internal_kernel_mod.h"
#include "ops/framework/utils.h"

// =============================================================================
// COMMON FUNCTION
// =============================================================================

namespace ms_custom_ops {

enum class InputIndex : size_t {
  kInputPermutedTokensIndex = 0,
  kInputSortedIndicesIndex = 1,
  kInputProbsIndex = 2,
  kInputPaddedModeIndex = 3,
  kInputRestoreShapeIndex = 4,
};

enum class OutputIndex : size_t { kOutputIndex = 0 };

ShapeVector MoeTokenUnpermuteMakeShape(const ShapeVector &permuted_tokens_shape, const ShapeVector &probs_shape) {
  if (permuted_tokens_shape.size() != kDim2) {
    MS_LOG(EXCEPTION) << "For MoeTokenUnpermute, permuted_tokens must be a 2D tensor, but got dimension: "
                      << permuted_tokens_shape.size();
  }
  if (probs_shape.size() != kDim2) {
    MS_LOG(EXCEPTION) << "For MoeTokenUnpermute, probs must be a 2D tensor, but got dimension: " << probs_shape.size();
  }
  ShapeVector out_shape{probs_shape[0], permuted_tokens_shape[1]};
  return out_shape;
}
// =============================================================================
// GRAPH MODE IMPLEMENTATION
// =============================================================================

class MoeTokenUnpermuteFuncImpl : public OpFuncImpl {
 public:
  MoeTokenUnpermuteFuncImpl() : OpFuncImpl() {}
  ~MoeTokenUnpermuteFuncImpl() = default;

  ShapeArray InferShape(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const override {
    auto &permuted_tokens = input_infos[static_cast<size_t>(InputIndex::kInputPermutedTokensIndex)];
    auto &probs = input_infos[static_cast<size_t>(InputIndex::kInputProbsIndex)];
    ShapeVector out_shape = {abstract::Shape::kShapeRankAny};
    if (permuted_tokens->IsDynamicRank() || probs->IsDynamicRank()) {
      return {out_shape};
    }
    out_shape = MoeTokenUnpermuteMakeShape(permuted_tokens->GetShape(), probs->GetShape());
    return {out_shape};
  }

  std::vector<TypeId> InferType(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const override {
    auto permuted_tokens_type = input_infos[static_cast<size_t>(InputIndex::kInputPermutedTokensIndex)]->GetType();
    if (permuted_tokens_type != TypeId::kNumberTypeFloat16) {
      MS_LOG(EXCEPTION) << "For MoeTokenUnpermute, permuted_tokens must be a float16 tensor, but got: "
                        << permuted_tokens_type;
    }
    return {permuted_tokens_type};
  }

  bool GeneralInferRegistered() const override { return true; }
};

class MoeTokenUnpermute : public InternalKernelMod {
 public:
  MoeTokenUnpermute() : InternalKernelMod() {}
  ~MoeTokenUnpermute() = default;

  void InitKernelInputsOutputsIndex() override {
    kernel_inputs_index_ = {
      static_cast<size_t>(InputIndex::kInputPermutedTokensIndex),
      static_cast<size_t>(InputIndex::kInputSortedIndicesIndex),
      static_cast<size_t>(InputIndex::kInputProbsIndex), static_cast<size_t>(InputIndex::kInputPaddedModeIndex),
      static_cast<size_t>(InputIndex::kInputRestoreShapeIndex)};
    kernel_outputs_index_ = {static_cast<size_t>(OutputIndex::kOutputIndex)};
  }

 protected:
  internal_v2::InternalOpPtr CreateKernel(const internal_v2::InputsImmutableInfoList &inputs,
                                          const internal_v2::OutputsImmutableInfoList &outputs,
                                          const std::vector<KernelTensor *> &ms_inputs,
                                          const std::vector<KernelTensor *> &ms_outputs) override {
    return internal_v2::CreateMoeTokenUnpermuteOp(inputs, outputs, internal_v2::kInternalMoeTokenUnpermuteOpName);
  }
};
}  // namespace ms_custom_ops

REG_GRAPH_MODE_OP(moe_token_unpermute, ms_custom_ops::MoeTokenUnpermuteFuncImpl, ms_custom_ops::MoeTokenUnpermute);
// =============================================================================
// PYBOOST MODE IMPLEMENTATION
// =============================================================================

#include "ops/framework/ms_kernels_internal/pyboost/internal_pyboost_runner.h"

namespace ms_custom_ops {
class MoeTokenUnpermuteRunner : public InternalPyboostRunner {
 public:
  explicit MoeTokenUnpermuteRunner(const std::string &op_name) : InternalPyboostRunner(op_name) {}
  ~MoeTokenUnpermuteRunner() = default;

 protected:
  internal_v2::InternalOpPtr CreateKernel(const internal_v2::InputsImmutableInfoList &inputs,
                                          const internal_v2::OutputsImmutableInfoList &outputs) override {
    return internal_v2::CreateMoeTokenUnpermuteOp(inputs, outputs, internal_v2::kInternalMoeTokenUnpermuteOpName);
  }
};

std::vector<ms::Tensor> npu_moe_token_unpermute(const ms::Tensor &permuted_tokens, const ms::Tensor &sorted_indices,
                                                const ms::Tensor &probs, const std::optional<bool> &padded_mode,
                                                const std::optional<std::vector<int32_t>> &restore_shape) {
  auto op_name = "MoeTokenUnpermute";
  auto runner = std::make_shared<MoeTokenUnpermuteRunner>(op_name);
  MS_EXCEPTION_IF_NULL(runner);

  // Setup the runner with all parameters (including hash calculation)
  runner->Setup(op_name, permuted_tokens, sorted_indices, probs, padded_mode, restore_shape);

  // if you need infer shape and type, you need create output tensors.
  ShapeVector output_shape = MoeTokenUnpermuteMakeShape(permuted_tokens.shape(), probs.shape());
  auto output = ms::Tensor(permuted_tokens.data_type(), output_shape);
  std::vector<ms::Tensor> inputs = {permuted_tokens, sorted_indices, probs};
  std::vector<ms::Tensor> outputs = {output};
  runner->GetOrCreateKernel(inputs, outputs);
  runner->Run(inputs, outputs);
  return outputs;
}
}  // namespace ms_custom_ops

auto pyboost_moe_token_unpermute(const ms::Tensor &permuted_tokens, const ms::Tensor &sorted_indices,
                                 const ms::Tensor &probs, const std::optional<bool> &padded_mode,
                                 const std::optional<std::vector<int32_t>> &restore_shape) {
  return ms::pynative::PyboostRunner::Call<1>(ms_custom_ops::npu_moe_token_unpermute, permuted_tokens, sorted_indices,
                                              probs, padded_mode, restore_shape);
}

MS_CUSTOM_OPS_EXTENSION_MODULE(m) {
  m.def("moe_token_unpermute", &pyboost_moe_token_unpermute, "MoeTokenUnpermute", pybind11::arg("permuted_tokens"),
        pybind11::arg("sorted_indices"), pybind11::arg("probs"), pybind11::arg("padded_mode") = false,
        pybind11::arg("restore_shape") = std::nullopt);
}
