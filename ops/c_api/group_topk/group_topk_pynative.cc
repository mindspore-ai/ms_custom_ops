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
// PYBOOST MODE IMPLEMENTATION
// =============================================================================

#include <vector>
#include <memory>
#include "ops/framework/ms_kernels_internal/pyboost/internal_pyboost_runner.h"
#include "ops/framework/utils.h"

namespace ms_custom_ops {
class GroupTopkRunner : public InternalPyboostRunner {
 public:
  using InternalPyboostRunner::InternalPyboostRunner;

  void SetGroupNum(const int32_t &group_num) { this->group_num_ = group_num; }
  void SetK(const int32_t &k) { this->k_ = k; }
  void SetKInner(const int32_t &k_inner) { this->k_inner_ = k_inner; }

 protected:
  internal_v2::InternalOpPtr CreateKernel(const internal_v2::InputsImmutableInfoList &inputs,
                                       const internal_v2::OutputsImmutableInfoList &outputs) override {
    internal_v2::GroupTopkParam param;
    param.group_num = this->group_num_;
    param.k = this->k_;
    param.k_inner = this->k_inner_;

    return internal_v2::CreateGroupTopkOp(inputs, outputs, param, internal_v2::kInternalGroupTopkOpName);
  }

 private:
  int32_t group_num_{0};
  int32_t k_{0};
  int32_t k_inner_{0};
};

void npu_group_topk(const ms::Tensor &token, const ms::Tensor &idx_arr, int64_t group_num, int64_t k,
                    int64_t k_inner) {
  auto op_name = "GroupTopk";
  auto runner = std::make_shared<ms_custom_ops::GroupTopkRunner>(op_name);
  MS_EXCEPTION_IF_NULL(runner);

  runner->SetGroupNum(static_cast<int32_t>(group_num));
  runner->SetK(static_cast<int32_t>(k));
  runner->SetKInner(static_cast<int32_t>(k_inner));

  // Setup the runner with all parameters (including hash calculation)
  runner->Setup(op_name, token, idx_arr, group_num, k, k_inner);

  std::vector<ms::Tensor> inputs = {token, idx_arr};
  std::vector<ms::Tensor> outputs = {};
  runner->GetOrCreateKernel(inputs, outputs);
  runner->Run(inputs, outputs);
  return;
}
}  // namespace ms_custom_ops

auto pyboost_group_topk(const ms::Tensor &token, const ms::Tensor &idx_arr, int64_t group_num, int64_t k,
                        int64_t k_inner) {
  return ms::pynative::PyboostRunner::Call<0>(ms_custom_ops::npu_group_topk, token, idx_arr, group_num, k, k_inner);
}

MS_CUSTOM_OPS_EXTENSION_MODULE(m) {
  m.def("group_topk", &pyboost_group_topk, "Group Topk", pybind11::arg("token"), pybind11::arg("idx_arr"),
        pybind11::arg("group_num"), pybind11::arg("k"), pybind11::arg("k_inner") = 1);
}
