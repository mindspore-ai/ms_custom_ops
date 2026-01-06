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
constexpr uint32_t COS_FORMAT_HALF_MODE = 2;
constexpr size_t kInputRankSize = 2;
constexpr int64_t kHiddenSizeLimit = 7168;
constexpr int64_t kHiddenSizeBase = 1024;
constexpr int64_t kExpertNumLimit = 256;
constexpr int64_t kOutputSize = 4;

inline void CheckVal(const std::string &op_name, const int64_t &value, const int64_t &expected_value,
                     const std::string &name) {
  auto any_dim = abstract::Shape::kShapeDimAny;
  if (value != any_dim && value != expected_value) {
    MS_EXCEPTION(ValueError) << "For op [" << op_name << "], " << name
                             << " must be " << expected_value << ", but got " << value;
  }
}

ShapeVector MoeInitRoutingV2MakeShape(const std::string &op_name, const ShapeVector &x_shape,
                                      const ShapeVector &expert_idx_shape, const int64_t &expert_num) {
  auto any_dim = abstract::Shape::kShapeDimAny;
  if (x_shape.size() != kInputRankSize) {
    MS_EXCEPTION(ShapeError) << "For op [" << op_name << "], the rank of x must be "
                             << kInputRankSize << ", but got " << x_shape.size();
  }
  if (expert_idx_shape.size() != kInputRankSize) {
    MS_EXCEPTION(ShapeError) << "For op [" << op_name << "], the rank of expert_idx must be "
                             << kInputRankSize << ", but got " << expert_idx_shape.size();
  }
  auto num_rows = x_shape[kIndex0];
  auto k = expert_idx_shape[kIndex1];
  auto expd_row_idx_dim = (num_rows == any_dim || k == any_dim) ? any_dim : num_rows * k;
  auto h = x_shape[kIndex1];
  if (h != any_dim && (h > kHiddenSizeLimit || h % kHiddenSizeBase != 0)) {
    MS_EXCEPTION(ValueError) << "For op [" << op_name << "], the hidden_size of x must be less than or "
                             << kHiddenSizeLimit << " and be multiple of " << kHiddenSizeBase
                             << ", but got " << h;
  }
  if (expert_num != any_dim && expert_num > kExpertNumLimit) {
    MS_EXCEPTION(ValueError) << "For op [" << op_name << "], the expert_num must be less than or equal to "
                             << kExpertNumLimit << ", but got " << expert_num;
  }
  return ShapeVector{expd_row_idx_dim, h};
}

// =============================================================================
// GRAPH MODE IMPLEMENTATION
// =============================================================================

class MoeInitRoutingV2FuncImpl : public OpFuncImpl {
 public:
  MoeInitRoutingV2FuncImpl() : OpFuncImpl() {}
  ~MoeInitRoutingV2FuncImpl() = default;

  ShapeArray InferShape(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const override {
    auto &x_info = input_infos[kIndex0];
    auto &expert_idx_info = input_infos[kIndex1];

    auto any_dim = abstract::Shape::kShapeDimAny;
    auto any_shape = abstract::TensorShape::kShapeRankAny;
    if (x_info->IsDynamicRank() || expert_idx_info->IsDynamicRank()) {
      return {{any_shape}, {any_dim}, {any_dim}, {any_dim}};
    }

    const auto &x_shp = x_info->GetShape();
    const auto &expert_idx_shp = expert_idx_info->GetShape();

    auto active_num = input_infos[kIndex2]->GetScalarValueWithCheck<int64_t>();
    auto expert_capacity = input_infos[kIndex3]->GetScalarValueWithCheck<int64_t>();
    auto expert_num = input_infos[kIndex4]->GetScalarValueWithCheck<int64_t>();
    auto drop_pad_mode = input_infos[kIndex5]->GetScalarValueWithCheck<int64_t>();

    CheckVal(primitive->name(), active_num, 0, "active_num");
    CheckVal(primitive->name(), expert_capacity, 0, "expert_capacity");
    CheckVal(primitive->name(), drop_pad_mode, 0, "drop_pad_mode");

    auto expanded_x_shape = MoeInitRoutingV2MakeShape(primitive->name(), x_shp, expert_idx_shp, expert_num);
    auto expanded_num = expanded_x_shape[kIndex0];

    ShapeArray out_shapes;
    (void)out_shapes.emplace_back(expanded_x_shape);
    (void)out_shapes.emplace_back(std::vector<int64_t>{expanded_num});
    (void)out_shapes.emplace_back(std::vector<int64_t>{expert_num});
    (void)out_shapes.emplace_back(std::vector<int64_t>{expert_num});
    return out_shapes;
  }

  std::vector<TypeId> InferType(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const override {
    auto x_type = input_infos[kIndex0]->GetType();
    auto idx_type = input_infos[kIndex1]->GetType();

    return {x_type, idx_type, idx_type, idx_type};
  }

  bool GeneralInferRegistered() const override { return true; }
};

class MoeInitRoutingV2 : public InternalKernelMod {
 public:
  MoeInitRoutingV2() : InternalKernelMod() {}
  ~MoeInitRoutingV2() = default;

  void InitKernelInputsOutputsIndex() override {
    kernel_inputs_index_ = {kIndex0, kIndex1};
    kernel_outputs_index_ = {kIndex0, kIndex1, kIndex2, kIndex3};
  }

  internal_v2::InternalOpPtr CreateKernel(const internal_v2::InputsImmutableInfoList &inputs,
                                          const internal_v2::OutputsImmutableInfoList &outputs,
                                          const std::vector<KernelTensor *> &ms_inputs,
                                          const std::vector<KernelTensor *> &ms_outputs) override {
    internal_v2::MoeInitRoutingParam param;
    param.active_num = ms_inputs[kIndex2]->GetValueWithCheck<int64_t>();
    param.expert_capacity = ms_inputs[kIndex3]->GetValueWithCheck<int64_t>();
    param.expert_num = ms_inputs[kIndex4]->GetValueWithCheck<int64_t>();
    param.drop_pad_mode = ms_inputs[kIndex5]->GetValueWithCheck<int64_t>();
    param.expert_tokens_count_or_cumsum_flag = ms_inputs[kIndex6]->GetValueWithCheck<int64_t>();
    param.expert_tokens_before_capacity_flag = ms_inputs[kIndex7]->GetValueWithCheck<bool>();
    return internal_v2::CreateMoeInitRoutingOp(inputs, outputs, param, internal_v2::kInternalMoeInitRoutingOpName);
  }
};
}  // namespace ms_custom_ops

REG_GRAPH_MODE_OP(moe_init_routing_v2, ms_custom_ops::MoeInitRoutingV2FuncImpl, ms_custom_ops::MoeInitRoutingV2);

// =============================================================================
// PYBOOST MODE IMPLEMENTATION
// =============================================================================

#include "ops/framework/ms_kernels_internal/pyboost/internal_pyboost_runner.h"

namespace ms_custom_ops {
class MoeInitRoutingV2Runner : public InternalPyboostRunner {
 public:
  using InternalPyboostRunner::InternalPyboostRunner;

  void SetParams(const int64_t &active_num, const int64_t &expert_capacity, const int64_t &expert_num,
                 const int64_t &drop_pad_mode, const int64_t &expert_tokens_count_or_cumsum_flag,
                 const bool &expert_tokens_before_capacity_flag) {
    param_.active_num = active_num;
    param_.expert_capacity = expert_capacity;
    param_.expert_num = expert_num;
    param_.drop_pad_mode = drop_pad_mode;
    param_.expert_tokens_count_or_cumsum_flag = expert_tokens_count_or_cumsum_flag;
    param_.expert_tokens_before_capacity_flag = expert_tokens_before_capacity_flag;
  }

 protected:
  internal_v2::InternalOpPtr CreateKernel(const internal_v2::InputsImmutableInfoList &inputs,
                                          const internal_v2::OutputsImmutableInfoList &outputs) override {
    return internal_v2::CreateMoeInitRoutingOp(inputs, outputs, param_, internal_v2::kInternalMoeInitRoutingOpName);
  }
 private:
  internal_v2::MoeInitRoutingParam param_;
};

std::vector<ms::Tensor> npu_moe_init_routing_v2(
  const ms::Tensor &x, const ms::Tensor &expert_idx, const int64_t &active_num,
  const int64_t &expert_capacity, const int64_t &expert_num,
  const int64_t &drop_pad_mode, const int64_t &expert_tokens_count_or_cumsum_flag,
  const bool &expert_tokens_before_capacity_flag) {
  auto op_name = "MoeInitRoutingV2";
  auto runner = std::make_shared<MoeInitRoutingV2Runner>(op_name);
  MS_EXCEPTION_IF_NULL(runner);

  runner->SetParams(active_num, expert_capacity, expert_num, drop_pad_mode, expert_tokens_count_or_cumsum_flag,
                    expert_tokens_before_capacity_flag);
  runner->Setup(op_name, x, expert_idx, active_num, expert_capacity, expert_num, drop_pad_mode,
                expert_tokens_count_or_cumsum_flag, expert_tokens_before_capacity_flag);

  auto expanded_x_shape = MoeInitRoutingV2MakeShape(op_name, x.shape(), expert_idx.shape(), expert_num);
  auto expanded_num = expanded_x_shape[kIndex0];

  std::vector<ms::Tensor> inputs = {x, expert_idx};
  std::vector<ms::Tensor> outputs = {ms::Tensor(x.data_type(), expanded_x_shape),
                                     ms::Tensor(expert_idx.data_type(), {expanded_num}),
                                     ms::Tensor(expert_idx.data_type(), {expert_num}),
                                     ms::Tensor(expert_idx.data_type(), {expert_num})};
  runner->GetOrCreateKernel(inputs, outputs);
  runner->Run(inputs, outputs);
  return outputs;
}
}  // namespace ms_custom_ops

auto pyboost_moe_init_routing_v2(
  const ms::Tensor &x, const ms::Tensor &expert_idx,
  const int64_t &active_num,
  const int64_t &expert_capacity, const int64_t &expert_num,
  const int64_t &drop_pad_mode, const int64_t &expert_tokens_count_or_cumsum_flag,
  const bool &expert_tokens_before_capacity_flag) {
  return ms::pynative::PyboostRunner::Call<ms_custom_ops::kOutputSize>(ms_custom_ops::npu_moe_init_routing_v2,
                                                                       x, expert_idx, active_num,
                                                                       expert_capacity, expert_num, drop_pad_mode,
                                                                       expert_tokens_count_or_cumsum_flag,
                                                                       expert_tokens_before_capacity_flag);
}

MS_CUSTOM_OPS_EXTENSION_MODULE(m) {
  m.def("moe_init_routing_v2", &pyboost_moe_init_routing_v2, "MoeInitRoutingV2", pybind11::arg("x"),
        pybind11::arg("expert_idx"), pybind11::arg("active_num"), pybind11::arg("expert_capacity"),
        pybind11::arg("expert_num"), pybind11::arg("drop_pad_mode"),
        pybind11::arg("expert_tokens_count_or_cumsum_flag"), pybind11::arg("expert_tokens_before_capacity_flag"));
}
