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
#include <memory>
#include <vector>
#include "ops/framework/aclnn/graphmode/aclnn_kernel_mod.h"
#include "ops/framework/utils.h"
#include "ops/c_api/utils/common_utils.h"

namespace ms_custom_ops {

constexpr size_t kBatchMatmulMatSize = 2;  // Last 2 dimensions for matrix multiplication
constexpr size_t kBatchMatmulDims = 3;     // Must be exactly 3D tensor: [batch, M, K] or [batch, K, N]

enum class BatchMatmulInputIndex : size_t {
  kBatchMatmulInputX1Index = 0,
  kBatchMatmulInputX2Index,
  kBatchMatmulInputCubeMathTypeIndex,
  kBatchMatmulInputsNum,
};

enum class BatchMatmulOutputIndex : size_t { kBatchMatmulOutputIndex = 0 };

/**
 * @brief Batch matrix multiplication operator function implementation for shape/type inference
 * This class handles the shape and type inference logic for batch matmul operations in graph mode.
 */
class OPS_API BatchMatMulCustomOpFuncImpl : public OpFuncImpl {
 public:
  /**
   * @brief Infer the output shape of batch matmul operation
   * @param primitive The primitive object containing operator information
   * @param input_infos List of input information including shapes
   * @return ShapeArray containing the inferred output shape
   */
  ShapeArray InferShape(const PrimitivePtr &primitive,
                        const InferInfoPtrList &input_infos) const override {
    auto x1_shape =
        input_infos[static_cast<size_t>(BatchMatmulInputIndex::kBatchMatmulInputX1Index)]->GetShape();
    auto x2_shape =
        input_infos[static_cast<size_t>(BatchMatmulInputIndex::kBatchMatmulInputX2Index)]->GetShape();

    if (IsDynamicRank(x1_shape) || IsDynamicRank(x2_shape)) {
      return {ShapeVector({abstract::Shape::kShapeRankAny})};
    }

    // Check that inputs must be exactly 3D tensors
    // In Graph mode, MS_LOG(EXCEPTION) will be converted to Python exception by MindSpore framework
    if (x1_shape.size() != kBatchMatmulDims || x2_shape.size() != kBatchMatmulDims) {
      MS_LOG(EXCEPTION) << "For 'BatchMatMul', inputs must be exactly 3D tensors, "
                        << "but got x1 with " << x1_shape.size() << "D and x2 with " << x2_shape.size() << "D";
    }

    // aclnnBatchMatMul does not support transpose, so we pass false
    ShapeVector out_shape =
        BatchMatMulMakeShape(x1_shape, x2_shape, false, false, kBatchMatmulMatSize, "BatchMatMul");
    return {out_shape};
  }

  /**
   * @brief Infer the output data type of batch matmul operation
   * @param primitive The primitive object containing operator information
   * @param input_infos List of input information including data types
   * @return std::vector<TypeId> containing the inferred output data type
   */
  std::vector<TypeId> InferType(const PrimitivePtr &primitive,
                                const InferInfoPtrList &input_infos) const override {
    // Check data types of both inputs
    TypeId x1_type = input_infos[static_cast<size_t>(
        BatchMatmulInputIndex::kBatchMatmulInputX1Index)]->GetType();
    TypeId x2_type = input_infos[static_cast<size_t>(
        BatchMatmulInputIndex::kBatchMatmulInputX2Index)]->GetType();

    // Check if data types are supported
    CheckBatchMatMulDataType(x1_type, "BatchMatMul");
    CheckBatchMatMulDataType(x2_type, "BatchMatMul");

    // Check if both inputs have the same type
    if (x1_type != x2_type) {
      MS_LOG(EXCEPTION) << "For 'BatchMatMul', inputs x1 and x2 must have the same data type, "
                        << "but got x1 with " << TypeIdToString(x1_type)
                        << " and x2 with " << TypeIdToString(x2_type);
    }

    // Use the first input's type as output type
    return {x1_type};
  }

  bool GeneralInferRegistered() const override { return true; }
};

/**
 * @brief Ascend kernel implementation for batch matrix multiplication using aclnnBatchMatMul
 * This class provides the kernel execution interface for batch matmul operations on Ascend devices.
 */
class BatchMatMulCustomAscend : public AclnnCustomKernelMod {
 public:
  BatchMatMulCustomAscend() : AclnnCustomKernelMod("aclnnBatchMatMul") {}
  ~BatchMatMulCustomAscend() = default;

  /**
   * @brief Launch the batch matmul kernel execution
   * @param inputs Input tensors including x1, x2, transpose flags and cube math type
   * @param workspace Workspace memory for kernel execution
   * @param outputs Output tensors
   * @param stream_ptr Stream pointer for asynchronous execution
   * @return bool indicating success or failure
   */
  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs, void *stream_ptr) override {
    MS_EXCEPTION_IF_NULL(stream_ptr);
    int8_t cube_math_type = static_cast<int8_t>(
        inputs[static_cast<size_t>(BatchMatmulInputIndex::kBatchMatmulInputCubeMathTypeIndex)]
            ->GetValueWithCheck<int64_t>());
    RunOp(stream_ptr, workspace, inputs[static_cast<size_t>(BatchMatmulInputIndex::kBatchMatmulInputX1Index)],
          inputs[static_cast<size_t>(BatchMatmulInputIndex::kBatchMatmulInputX2Index)],
          outputs[static_cast<size_t>(BatchMatmulOutputIndex::kBatchMatmulOutputIndex)], cube_math_type);
    return true;
  }

  /**
   * @brief Get workspace information required for kernel execution
   * @param inputs Input tensors
   * @param outputs Output tensors
   */
  void GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                        const std::vector<KernelTensor *> &outputs) override {
    int8_t cube_math_type = static_cast<int8_t>(
        inputs[static_cast<size_t>(BatchMatmulInputIndex::kBatchMatmulInputCubeMathTypeIndex)]
            ->GetValueWithCheck<int64_t>());
    GetWorkspaceForResize(inputs[static_cast<size_t>(BatchMatmulInputIndex::kBatchMatmulInputX1Index)],
                          inputs[static_cast<size_t>(BatchMatmulInputIndex::kBatchMatmulInputX2Index)],
                          outputs[static_cast<size_t>(BatchMatmulOutputIndex::kBatchMatmulOutputIndex)],
                          cube_math_type);
  }

 private:
  DEFINE_GET_WORKSPACE_FOR_RESIZE();
};
}  // namespace ms_custom_ops

REG_GRAPH_MODE_OP(batch_matmul, ms_custom_ops::BatchMatMulCustomOpFuncImpl,
                  ms_custom_ops::BatchMatMulCustomAscend);

// =============================================================================
// PYBOOST MODE IMPLEMENTATION
// =============================================================================

#include "ops/framework/ms_kernels_internal/pyboost/internal_pyboost_runner.h"

namespace ms_custom_ops {
/**
 * @brief PyBoost mode implementation of batch matrix multiplication
 * @param x1 First input tensor with shape [..., M, K]
 * @param x2 Second input tensor with shape [..., K, N]
 * @param cube_math_type Cube math type for precision control
 * @return ms::Tensor Result tensor with shape [..., M, N]
 */
ms::Tensor batch_matmul_custom(const ms::Tensor &x1, const ms::Tensor &x2, int64_t cube_math_type) {
  // Check data types of both inputs
  TypeId x1_type = x1.data_type();
  TypeId x2_type = x2.data_type();

  // Check if data types are supported
  CheckBatchMatMulDataType(x1_type, "BatchMatMul");
  CheckBatchMatMulDataType(x2_type, "BatchMatMul");

  // Check if both inputs have the same type
  if (x1_type != x2_type) {
    MS_EXCEPTION(ValueError) << "For 'BatchMatMul', inputs x1 and x2 must have the same data type, "
                             << "but got x1 with " << TypeIdToString(x1_type)
                             << " and x2 with " << TypeIdToString(x2_type);
  }

  auto x1_shape = x1.shape();
  auto x2_shape = x2.shape();

  // Check that inputs must be exactly 3D tensors
  if (x1_shape.size() != kBatchMatmulDims || x2_shape.size() != kBatchMatmulDims) {
    MS_EXCEPTION(ValueError) << "For 'BatchMatMul', inputs must be exactly 3D tensors, "
                             << "but got x1 with " << x1_shape.size() << "D and x2 with " << x2_shape.size() << "D";
  }

  // aclnnBatchMatMul does not support transpose, so we pass false
  auto output_shape = BatchMatMulMakeShape(x1_shape, x2_shape, false, false, kBatchMatmulMatSize, "BatchMatMul");
  TypeId out_dtype = x1.data_type();
  auto out = ms::Tensor(out_dtype, output_shape);

  int8_t cube_math_type_int8 = static_cast<int8_t>(cube_math_type);
  auto runner = std::make_shared<ms::pynative::AclnnOpRunner>("BatchMatMul");
  runner->SetLaunchFunc(LAUNCH_ACLNN_FUNC(aclnnBatchMatMul, x1, x2, out, cube_math_type_int8));
  runner->Run({x1, x2}, {out});
  return out;
}
}  // namespace ms_custom_ops

auto pyboost_batch_matmul(const ms::Tensor &x1, const ms::Tensor &x2, int64_t cube_math_type) {
  return ms::pynative::PyboostRunner::Call<ms_custom_ops::kNumber1>(ms_custom_ops::batch_matmul_custom, x1, x2,
                                                                     cube_math_type);
}

MS_CUSTOM_OPS_EXTENSION_MODULE(m) {
  m.def("batch_matmul", &pyboost_batch_matmul, "BatchMatMul", pybind11::arg("x1"), pybind11::arg("x2"),
        pybind11::arg("cube_math_type") = 0);
}

