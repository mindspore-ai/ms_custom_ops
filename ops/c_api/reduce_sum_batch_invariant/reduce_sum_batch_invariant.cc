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
enum class ReduceSumBatchInvariantInputIndex : size_t {
  kInputIndex = 0,
  kDimsIndex,
  kKeepDimsIndex,
  kOutputDtypeIndex,
  kInputsNum,
};

class OPS_API ReduceSumBatchInvariantCustomOpFuncImpl : public OpFuncImpl {
 public:
  ShapeArray InferShape(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const override {
    MS_EXCEPTION_IF_NULL(primitive);
    auto op_name = primitive->name();

    // Check for dynamic rank
    if (input_infos[static_cast<size_t>(ReduceSumBatchInvariantInputIndex::kInputIndex)]->IsDynamicRank()) {
      return {ShapeVector({abstract::Shape::kShapeRankAny})};
    }

    auto input_shape = input_infos[static_cast<size_t>(ReduceSumBatchInvariantInputIndex::kInputIndex)]->GetShape();

    // Get dims parameter
    auto dims_opt =
      input_infos[static_cast<size_t>(ReduceSumBatchInvariantInputIndex::kDimsIndex)]->GetArrayValue<int64_t>();
    if (!dims_opt.has_value()) {
      // If dims is dynamic, return dynamic shape
      return {ShapeVector({abstract::Shape::kShapeRankAny})};
    }
    auto dims_vec = dims_opt.value().ToVector();

    // Get keep_dims parameter
    auto keep_dims = input_infos[static_cast<size_t>(ReduceSumBatchInvariantInputIndex::kKeepDimsIndex)]
                       ->GetScalarValueWithCheck<bool>();

    // Handle empty dims case: output shape equals input shape
    if (dims_vec.empty()) {
      return {input_shape};
    }

    int64_t ndim = static_cast<int64_t>(input_shape.size());
    // Handle scalar input (0-dim tensor)
    int64_t effective_ndim = (ndim == 0) ? 1 : ndim;

    // Validate dims and convert to positive indices
    std::set<int64_t> reduce_dims_set;
    for (auto dim : dims_vec) {
      if (dim < -effective_ndim || dim >= effective_ndim) {
        MS_LOG(EXCEPTION) << "For '" << op_name << "', dim " << dim << " is out of range [" << -effective_ndim << ", "
                          << effective_ndim - 1 << "]";
      }
      int64_t pos_dim = dim < 0 ? dim + effective_ndim : dim;
      reduce_dims_set.insert(pos_dim);
    }

    // Build output shape
    ShapeVector out_shape;
    if (ndim == 0) {
      out_shape = input_shape;
    } else if (keep_dims) {
      // Keep dims: set reduced dimensions to 1
      out_shape = input_shape;
      for (auto dim : reduce_dims_set) {
        out_shape[dim] = 1;
      }
    } else {
      // Don't keep dims: remove reduced dimensions
      for (int64_t i = 0; i < ndim; ++i) {
        if (reduce_dims_set.find(i) == reduce_dims_set.end()) {
          out_shape.push_back(input_shape[i]);
        }
      }
    }

    return {out_shape};
  }

  std::vector<TypeId> InferType(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const override {
    MS_EXCEPTION_IF_NULL(primitive);

    // Get output_dtype parameter and use it as output type
    auto output_dtype = input_infos[static_cast<size_t>(ReduceSumBatchInvariantInputIndex::kOutputDtypeIndex)]
                          ->GetScalarValueWithCheck<int64_t>();
    auto out_type = static_cast<TypeId>(output_dtype);
    return {out_type};
  }

  bool GeneralInferRegistered() const override { return true; }
};

class ReduceSumBatchInvariantCustomAscend : public AclnnCustomKernelMod {
 public:
  ReduceSumBatchInvariantCustomAscend() : AclnnCustomKernelMod("aclnnReduceSumBatchInvariant") {}
  ~ReduceSumBatchInvariantCustomAscend() = default;

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs, void *stream_ptr) override {
    MS_EXCEPTION_IF_NULL(stream_ptr);
    auto input = inputs[static_cast<size_t>(ReduceSumBatchInvariantInputIndex::kInputIndex)];
    auto out = outputs[0];

    RunOp(stream_ptr, workspace, input, dims_, keep_dims_, output_dtype_, out);
    return true;
  }

  void GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                        const std::vector<KernelTensor *> &outputs) override {
    auto input = inputs[static_cast<size_t>(ReduceSumBatchInvariantInputIndex::kInputIndex)];

    const auto dims_opt = inputs[static_cast<size_t>(ReduceSumBatchInvariantInputIndex::kDimsIndex)]
                            ->GetOptionalValueWithCheck<std::vector<int64_t>>();

    dims_ = dims_opt.has_value() ? dims_opt.value() : std::vector<int64_t>{};

    keep_dims_ = device::ascend::ConvertKernelTensor<bool>(
      inputs[static_cast<size_t>(ReduceSumBatchInvariantInputIndex::kKeepDimsIndex)]);
    output_dtype_ = static_cast<TypeId>(device::ascend::ConvertKernelTensor<int64_t>(
      inputs[static_cast<size_t>(ReduceSumBatchInvariantInputIndex::kOutputDtypeIndex)]));
    auto out = outputs[0];

    GetWorkspaceForResize(input, dims_, keep_dims_, output_dtype_, out);
    return;
  }

 private:
  bool keep_dims_{false};
  TypeId output_dtype_{kTypeUnknown};
  std::vector<int64_t> dims_;
  DEFINE_GET_WORKSPACE_FOR_RESIZE();
};
}  // namespace ms_custom_ops

REG_GRAPH_MODE_OP(reduce_sum_batch_invariant, ms_custom_ops::ReduceSumBatchInvariantCustomOpFuncImpl,
                  ms_custom_ops::ReduceSumBatchInvariantCustomAscend);

// =============================================================================
// PYBOOST MODE IMPLEMENTATION
// =============================================================================

namespace ms_custom_ops {
ms::Tensor reduce_sum_batch_invariant_custom(const ms::Tensor &input, const std::vector<int64_t> &dims, bool keep_dims,
                                             int64_t output_dtype) {
  auto runner = std::make_shared<ms::pynative::AclnnOpRunner>("aclnnReduceSumBatchInvariant");

  // Infer output shape based on input shape, dims, and keep_dims
  auto input_shape = input.shape();
  std::vector<int64_t> out_shape;

  // Convert negative dims to positive and sort
  std::set<int64_t> reduce_dims_set;
  int64_t ndim = static_cast<int64_t>(input_shape.size());
  for (auto d : dims) {
    int64_t pos_d = d >= 0 ? d : d + ndim;
    if (pos_d >= 0 && pos_d < ndim) {
      reduce_dims_set.insert(pos_d);
    }
  }

  // Build output shape
  for (int64_t i = 0; i < ndim; ++i) {
    if (reduce_dims_set.find(i) != reduce_dims_set.end()) {
      if (keep_dims) {
        out_shape.push_back(1);
      }
    } else {
      out_shape.push_back(input_shape[i]);
    }
  }

  // Handle empty dims case (reduce all dimensions)
  if (dims.empty()) {
    if (keep_dims) {
      out_shape = std::vector<int64_t>(input_shape.size(), 1);
    } else {
      out_shape = {};
    }
  }

  // Create output tensor with specified output_dtype
  TypeId out_dtype = static_cast<TypeId>(output_dtype);
  auto out = ms::Tensor(out_dtype, out_shape);

  runner->SetLaunchFunc(LAUNCH_ACLNN_FUNC(aclnnReduceSumBatchInvariant, input, dims, keep_dims, out_dtype, out));
  runner->Run({input}, {out});

  return out;
}
}  // namespace ms_custom_ops

auto pyboost_reduce_sum_batch_invariant(const ms::Tensor &input, const std::vector<int64_t> &dims, bool keep_dims,
                                        int64_t output_dtype) {
  return ms::pynative::PyboostRunner::Call<ms_custom_ops::kNumber1>(ms_custom_ops::reduce_sum_batch_invariant_custom,
                                                                    input, dims, keep_dims, output_dtype);
}

MS_CUSTOM_OPS_EXTENSION_MODULE(m) {
  m.def("reduce_sum_batch_invariant", &pyboost_reduce_sum_batch_invariant, "Reduce Sum Batch Invariant",
        pybind11::arg("input"), pybind11::arg("dims"), pybind11::arg("keep_dims"), pybind11::arg("output_dtype"));
}
