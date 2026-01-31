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
#include <string>
#include <vector>
#include <set>
#include "ops/framework/aclnn/graphmode/aclnn_kernel_mod.h"
#include "ops/framework/utils.h"
#include "ops/c_api/utils/common_utils.h"

namespace ms_custom_ops {

// Input indices
constexpr size_t kMatmulBIInputX1 = 0;
constexpr size_t kMatmulBIInputX2 = 1;
constexpr size_t kMatmulBICubeMathType = 2;
constexpr size_t kMatmulBIOutputY = 0;

// Shape constants
constexpr size_t kMatmulMinShapeSize = 2;
constexpr size_t kMatmulMaxShapeSize = 6;
constexpr int64_t kUnknownDim = -1;

// =============================================================================
// Common InferShape function for both Graph and PyBoost modes
// =============================================================================

/**
 * @brief Broadcast batch dimensions of two shapes.
 *        Follows standard NumPy/PyTorch broadcasting rules.
 *
 * @param batch1 Batch dimensions of first tensor (excluding last 2 dims)
 * @param batch2 Batch dimensions of second tensor (excluding last 2 dims)
 * @param op_name Operator name for error messages
 * @return ShapeVector The broadcasted batch shape
 */
ShapeVector BroadcastBatchShape(const ShapeVector &batch1, const ShapeVector &batch2,
                                const std::string &op_name) {
  size_t max_len = std::max(batch1.size(), batch2.size());
  ShapeVector result(max_len);

  for (size_t i = 0; i < max_len; ++i) {
    int64_t dim1 = 1;
    int64_t dim2 = 1;

    if (i < batch1.size()) {
      dim1 = batch1[batch1.size() - 1 - i];
    }
    if (i < batch2.size()) {
      dim2 = batch2[batch2.size() - 1 - i];
    }

    if (dim1 == kUnknownDim || dim2 == kUnknownDim) {
      result[max_len - 1 - i] = kUnknownDim;
    } else if (dim1 == dim2) {
      result[max_len - 1 - i] = dim1;
    } else if (dim1 == 1) {
      result[max_len - 1 - i] = dim2;
    } else if (dim2 == 1) {
      result[max_len - 1 - i] = dim1;
    } else {
      MS_LOG(EXCEPTION) << "For '" << op_name << "', batch dimensions cannot be broadcast: "
                        << dim1 << " vs " << dim2;
    }
  }

  return result;
}

/**
 * @brief Infer output shape for matmul batch invariant operation.
 *        Supports 1D to 6D inputs with batch dimension broadcasting.
 *        Follows the logic from mat_mul_v3_batch_invariant_infershape.cpp
 *
 * @param x1_shape Shape of left matrix
 * @param x2_shape Shape of right matrix
 * @param op_name Operator name for error messages
 * @return ShapeVector The inferred output shape
 */
ShapeVector MatmulBatchInvariantInferShape(const ShapeVector &x1_shape, const ShapeVector &x2_shape,
                                           const std::string &op_name) {
  ShapeVector shape_x1 = x1_shape;
  ShapeVector shape_x2 = x2_shape;
  bool x1_is_1d = false;
  bool x2_is_1d = false;

  // Handle 1D input: complement to 2D
  // For x1: [K] -> [1, K]
  if (shape_x1.size() == 1 && shape_x1[0] > 0) {
    x1_is_1d = true;
    int64_t ori_dim = shape_x1[0];
    shape_x1 = {1, ori_dim};
  }

  // For x2: [K] -> [K, 1]
  if (shape_x2.size() == 1 && shape_x2[0] > 0) {
    x2_is_1d = true;
    int64_t ori_dim = shape_x2[0];
    shape_x2 = {ori_dim, 1};
  }

  // Validate shape dimensions (after complement, should be 2-6D)
  size_t dim_num_x1 = shape_x1.size();
  size_t dim_num_x2 = shape_x2.size();

  if (dim_num_x1 < kMatmulMinShapeSize || dim_num_x1 > kMatmulMaxShapeSize) {
    MS_LOG(EXCEPTION) << "For '" << op_name << "', x1 shape dimension [" << dim_num_x1
                      << "] must be between " << kMatmulMinShapeSize << " and " << kMatmulMaxShapeSize;
  }

  if (dim_num_x2 < kMatmulMinShapeSize || dim_num_x2 > kMatmulMaxShapeSize) {
    MS_LOG(EXCEPTION) << "For '" << op_name << "', x2 shape dimension [" << dim_num_x2
                      << "] must be between " << kMatmulMinShapeSize << " and " << kMatmulMaxShapeSize;
  }

  // Get M, K from x1 (last two dimensions: [..., M, K])
  int64_t m = shape_x1[dim_num_x1 - 2];
  int64_t k_x1 = shape_x1[dim_num_x1 - 1];

  // Get K, N from x2 (last two dimensions: [..., K, N])
  int64_t k_x2 = shape_x2[dim_num_x2 - 2];
  int64_t n = shape_x2[dim_num_x2 - 1];

  // Validate K dimension consistency
  if (k_x1 != kUnknownDim && k_x2 != kUnknownDim && k_x1 != k_x2) {
    MS_LOG(EXCEPTION) << "For '" << op_name << "', the K-axis of x1(" << k_x1
                      << ") and x2(" << k_x2 << ") tensors must be the same";
  }

  // Extract batch dimensions (all dimensions except last 2)
  ShapeVector batch_x1(shape_x1.begin(), shape_x1.end() - 2);
  ShapeVector batch_x2(shape_x2.begin(), shape_x2.end() - 2);

  // Broadcast batch dimensions
  ShapeVector batch_out = BroadcastBatchShape(batch_x1, batch_x2, op_name);

  // Build output shape: [...batch..., M, N]
  ShapeVector out_shape = batch_out;
  out_shape.push_back(m);
  out_shape.push_back(n);

  // Handle output shape complement for 1D inputs
  if (x1_is_1d && !x2_is_1d) {
    // x1 was 1D, remove M dimension: [..., 1, N] -> [..., N]
    out_shape.erase(out_shape.end() - 2);
  } else if (!x1_is_1d && x2_is_1d) {
    // x2 was 1D, remove N dimension: [..., M, 1] -> [..., M]
    out_shape.pop_back();
  }

  return out_shape;
}

class OPS_API MatmulBatchInvariantCustomOpFuncImpl : public OpFuncImpl {
 public:
  ShapeArray InferShape(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const override {
    auto x1_shape = input_infos[kMatmulBIInputX1]->GetShape();
    auto x2_shape = input_infos[kMatmulBIInputX2]->GetShape();

    if (IsDynamicRank(x1_shape) || IsDynamicRank(x2_shape)) {
      return {ShapeVector({abstract::Shape::kShapeRankAny})};
    }

    auto op_name = primitive->name();
    ShapeVector out_shape = MatmulBatchInvariantInferShape(x1_shape, x2_shape, op_name);
    return {out_shape};
  }

  std::vector<TypeId> InferType(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const override {
    auto op_name = primitive->name();
    auto x1_dtype = input_infos[kMatmulBIInputX1]->GetType();
    auto x2_dtype = input_infos[kMatmulBIInputX2]->GetType();

    // Validate input types
    const std::set<TypeId> valid_types = {kNumberTypeFloat16, kNumberTypeBFloat16, kNumberTypeFloat32};
    CheckAndConvertUtils::CheckTypeIdValid("x1", x1_dtype, valid_types, op_name);
    CheckAndConvertUtils::CheckTypeIdValid("x2", x2_dtype, valid_types, op_name);

    // Output type is same as x1
    return {x1_dtype};
  }

  bool GeneralInferRegistered() const override { return true; }
};

class MatmulBatchInvariantCustomAscend : public AclnnCustomKernelMod {
 public:
  MatmulBatchInvariantCustomAscend() : AclnnCustomKernelMod("aclnnMatmulBatchInvariant") {}
  ~MatmulBatchInvariantCustomAscend() = default;

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs, void *stream_ptr) override {
    MS_EXCEPTION_IF_NULL(stream_ptr);
    auto x1 = inputs[kMatmulBIInputX1];
    auto x2 = inputs[kMatmulBIInputX2];
    auto out = outputs[kMatmulBIOutputY];

    RunOp(stream_ptr, workspace, x1, x2, out, cube_math_type_);
    return true;
  }

  void GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                        const std::vector<KernelTensor *> &outputs) override {
    cube_math_type_ = static_cast<int8_t>(inputs[kMatmulBICubeMathType]->GetValueWithCheck<int64_t>());
    GetWorkspaceForResize(inputs[kMatmulBIInputX1], inputs[kMatmulBIInputX2], outputs[kMatmulBIOutputY],
                          cube_math_type_);
  }

 private:
  DEFINE_GET_WORKSPACE_FOR_RESIZE();
  int8_t cube_math_type_{0};
};
}  // namespace ms_custom_ops

REG_GRAPH_MODE_OP(matmul_batch_invariant, ms_custom_ops::MatmulBatchInvariantCustomOpFuncImpl,
                  ms_custom_ops::MatmulBatchInvariantCustomAscend);

// =============================================================================
// PYBOOST MODE IMPLEMENTATION
// =============================================================================

namespace ms_custom_ops {

ms::Tensor matmul_batch_invariant_custom(const ms::Tensor &x1, const ms::Tensor &x2, int64_t cube_math_type) {
  auto x1_shape = x1.shape();
  auto x2_shape = x2.shape();

  // Infer output shape using common function
  auto output_shape = MatmulBatchInvariantInferShape(x1_shape, x2_shape, "matmul_batch_invariant");

  // Output dtype is same as x1
  TypeId out_dtype = x1.data_type();
  auto out = ms::Tensor(out_dtype, output_shape);

  int8_t cube_math_type_int8 = static_cast<int8_t>(cube_math_type);

  auto runner = std::make_shared<ms::pynative::AclnnOpRunner>("aclnnMatmulBatchInvariant");
  runner->SetLaunchFunc(LAUNCH_ACLNN_FUNC(aclnnMatmulBatchInvariant, x1, x2, out, cube_math_type_int8));
  runner->Run({x1, x2}, {out});

  return out;
}
}  // namespace ms_custom_ops

auto pyboost_matmul_batch_invariant(const ms::Tensor &x1, const ms::Tensor &x2, int64_t cube_math_type) {
  return ms::pynative::PyboostRunner::Call<ms_custom_ops::kNumber1>(ms_custom_ops::matmul_batch_invariant_custom, x1, x2,
                                                                    cube_math_type);
}

MS_CUSTOM_OPS_EXTENSION_MODULE(m) {
  m.def("matmul_batch_invariant", &pyboost_matmul_batch_invariant, "Matmul Batch Invariant", pybind11::arg("x1"),
        pybind11::arg("x2"), pybind11::arg("cube_math_type") = 0);
}
