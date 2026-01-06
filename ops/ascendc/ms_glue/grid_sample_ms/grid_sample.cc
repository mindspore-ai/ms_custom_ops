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
#include <memory>
#include <utility>

#include "ops/framework/aclnn/graphmode/aclnn_kernel_mod.h"
#include "ops/framework/utils.h"
#include "mindspore/include/custom_op_api.h"

namespace ms_custom_ops {
constexpr size_t kCAligned = 8;
enum class GridSampleInputIndex : size_t {
  kGridSampleInputIndex = 0,
  kGridSampleGridIndex,
  kGridSampleModeIndex,
  kGridSamplePaddingModeIndex,
  kGridSampleAlignCornersIndex,
  kGridSampleInputsNum,
};

static void GridSampleCheckInputsShape(const std::string &op_name, const std::vector<int64_t> &input_shape,
                                      const std::vector<int64_t> &grid_shape, int64_t mode,
                                      int64_t padding_mode, bool align_corners) {
  auto any_dim = abstract::Shape::kShapeDimAny;
  MS_CHECK_VALUE(mode == 0,
                 CheckAndConvertUtils::FormatCommMsg(op_name, ", mode only supports 0, but got ", mode));
  MS_CHECK_VALUE(
    padding_mode == 1,
    CheckAndConvertUtils::FormatCommMsg(op_name, ", padding_mode only supports 1, but got ", padding_mode));
  MS_CHECK_VALUE(align_corners == false,
                 CheckAndConvertUtils::FormatCommMsg(op_name, ", align_corners only supports false, but got ",
                                                     align_corners));
  if (input_shape.size() != kDim4 || grid_shape.size() != kDim4) {
    MS_LOG(EXCEPTION) << op_name << ", the dim of inputs should be input.dim=grid.dim=4, "
                      << "but got input.dim=" << input_shape.size()
                      << ", grid.dim=" << grid_shape.size();
  }
  auto input_n = input_shape[kIndex0];
  auto grid_n = grid_shape[kIndex0];
  MS_CHECK_VALUE(input_n == grid_n || input_n == any_dim || grid_n == any_dim,
                 CheckAndConvertUtils::FormatCommMsg(
                   op_name, ", input.dim0 should be equal grid.dim0,",
                   " but got input.shape=", input_shape, ", grid.shape=", grid_shape));
  MS_CHECK_VALUE(
    grid_shape[kIndex3] == kDim2 || grid_shape[kIndex3] == any_dim,
    CheckAndConvertUtils::FormatCommMsg(
      op_name, ", grid.shape should be equals (N, H_OUT, W_OUT, 2), but got grid.shape=", grid_shape));
  auto c_in = input_shape[kIndex3];
  MS_CHECK_VALUE(
    c_in % kCAligned == 0,
    CheckAndConvertUtils::FormatCommMsg(
      op_name, ", c should be aligned with ", kCAligned, ", but got c=", c_in));
}

static void GridSampleCheckInputsType(const std::string &op_name, const TypeId &input_dtype,
                                               const TypeId &grid_dtype) {
  if (input_dtype != kNumberTypeFloat32) {
    MS_LOG(EXCEPTION) << op_name << ", the dtype of 'input' should be " << TypeIdToString(kNumberTypeFloat32)
                      << ", but got input.dtype=" << TypeIdToString(input_dtype);
  }
  if (grid_dtype != kNumberTypeFloat32) {
    MS_LOG(EXCEPTION) << op_name << ", the dtype of 'grid' should be " << TypeIdToString(kNumberTypeFloat32)
                      << ", but got grid.dtype=" << TypeIdToString(grid_dtype);
  }
}

class OPS_API GridSampleOpFuncImpl : public OpFuncImpl {
 public:
  ShapeArray InferShape(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const override {
    MS_EXCEPTION_IF_NULL(primitive);
    if (input_infos[static_cast<size_t>(GridSampleInputIndex::kGridSampleInputIndex)]
          ->IsDynamicRank() ||
        input_infos[static_cast<size_t>(GridSampleInputIndex::kGridSampleGridIndex)]
          ->IsDynamicRank()) {
      return {input_infos[static_cast<size_t>(GridSampleInputIndex::kGridSampleInputIndex)]->GetShape()};
    }
    auto op_name = primitive->name();
    auto input_shape =
      input_infos[static_cast<size_t>(GridSampleInputIndex::kGridSampleInputIndex)]->GetShape();
    auto grid_shape =
      input_infos[static_cast<size_t>(GridSampleInputIndex::kGridSampleGridIndex)]->GetShape();
    auto mode =
      input_infos[static_cast<size_t>(GridSampleInputIndex::kGridSampleModeIndex)]
        ->GetScalarValueWithCheck<int64_t>();
    auto padding_mode = input_infos[static_cast<size_t>(GridSampleInputIndex::kGridSamplePaddingModeIndex)]
                          ->GetScalarValueWithCheck<int64_t>();
    auto align_corners = input_infos[static_cast<size_t>(GridSampleInputIndex::kGridSampleAlignCornersIndex)]
      ->GetScalarValueWithCheck<bool>();
    GridSampleCheckInputsShape(op_name, input_shape, grid_shape, mode, padding_mode, align_corners);
    auto output_shape = grid_shape;
    output_shape[kIndex3] = input_shape[kIndex3];
    return {output_shape};
  }

  std::vector<TypeId> InferType(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const override {
    auto op_name = primitive->name();
    auto input_dtype =
      input_infos[static_cast<size_t>(GridSampleInputIndex::kGridSampleInputIndex)]->GetType();
    auto grid_dtype =
      input_infos[static_cast<size_t>(GridSampleInputIndex::kGridSampleGridIndex)]->GetType();
    GridSampleCheckInputsType(op_name, input_dtype, grid_dtype);
    return {input_dtype};
  }

  bool GeneralInferRegistered() const override { return true; }
};

class GridSample : public AclnnCustomKernelMod {
 public:
  GridSample() : AclnnCustomKernelMod(std::move("aclnnGridSampleMS")) {}
  ~GridSample() = default;

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs, void *stream_ptr) override {
    MS_EXCEPTION_IF_NULL(stream_ptr);
    RunOp(
      stream_ptr, workspace, inputs[static_cast<size_t>(GridSampleInputIndex::kGridSampleInputIndex)],
      inputs[static_cast<size_t>(GridSampleInputIndex::kGridSampleGridIndex)],
      outputs[0]);
    return true;
  }
  void GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                        const std::vector<KernelTensor *> &outputs) override {
    GetWorkspaceForResize(inputs[static_cast<size_t>(GridSampleInputIndex::kGridSampleInputIndex)],
                          inputs[static_cast<size_t>(GridSampleInputIndex::kGridSampleGridIndex)],
                          outputs[0]);
    return;
  }

 private:
  DEFINE_GET_WORKSPACE_FOR_RESIZE();
};
}  // namespace ms_custom_ops

REG_GRAPH_MODE_OP(grid_sample, ms_custom_ops::GridSampleOpFuncImpl,
                  ms_custom_ops::GridSample);

// =============================================================================
// PYBOOST MODE IMPLEMENTATION
// =============================================================================

namespace ms_custom_ops {
constexpr size_t kGridSampleOutputNum = 1;

std::vector<ms::Tensor> grid_sample_custom(const ms::Tensor &input, const ms::Tensor &grid,
                                           const int64_t mode, const int64_t padding_mode,
                                           const bool align_corners) {
  std::string op_name = "grid_sample";
  auto runner = std::make_shared<ms::pynative::AclnnOpRunner>(op_name);
  MS_EXCEPTION_IF_NULL(runner);
  auto input_shape = input.shape();
  auto grid_shape = grid.shape();
  auto output_shape = grid_shape;
  output_shape[kIndex3] = input_shape[kIndex3];
  GridSampleCheckInputsShape(op_name, input.shape(), grid.shape(), mode, padding_mode, align_corners);
  GridSampleCheckInputsType(op_name, input.data_type(), grid.data_type());
  auto out = ms::Tensor(input.data_type(), output_shape);
  runner->SetLaunchFunc(LAUNCH_ACLNN_FUNC(aclnnGridSampleMS, input, grid, out));
  runner->Run({input, grid}, {out});
  return {out};
}
}  // namespace ms_custom_ops

auto pyboost_grid_sample(const ms::Tensor &input, const ms::Tensor &grid, const int64_t mode,
                         const int64_t padding_mode, const bool align_corners) {
  return ms::pynative::PyboostRunner::Call<ms_custom_ops::kGridSampleOutputNum>(
    ms_custom_ops::grid_sample_custom, input, grid, mode, padding_mode, align_corners);
}

MS_CUSTOM_OPS_EXTENSION_MODULE(m) {
  m.def("grid_sample",
        &pyboost_grid_sample,
        "GridSample", pybind11::arg("input"), pybind11::arg("grid"), pybind11::arg("mode") = 0,
        pybind11::arg("padding_mode") = 1, pybind11::arg("align_corners") = false);
}
