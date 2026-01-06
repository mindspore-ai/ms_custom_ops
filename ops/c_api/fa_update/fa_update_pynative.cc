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
#include <memory>
#include <vector>

#include "ops/framework/ms_kernels_internal/pyboost/internal_pyboost_runner.h"
#include "ops/c_api/fa_update/fa_update_common.h"
#include "ops/framework/utils.h"

namespace ms_custom_ops {
constexpr size_t FA_UPDATA_HEAD_SIZE_MIN = 8;
constexpr size_t FA_UPDATA_HEAD_SIZE_MAX = 512;
class FaUpdateRunner : public InternalPyboostRunner {
 public:
  using InternalPyboostRunner::InternalPyboostRunner;
  void SetFaUpdateType(const uint64_t &fa_update_type) { this->fa_update_type_ = fa_update_type; }
  void SetSp(const uint64_t &sp) { this->sp_ = sp; }
  internal_v2::FaUpdateParam param_;

 protected:
  internal_v2::InternalOpPtr CreateKernel(const internal_v2::InputsImmutableInfoList &inputs,
                                          const internal_v2::OutputsImmutableInfoList &outputs) override {
    return internal_v2::CreateFaUpdateOp(inputs, outputs, param_, internal_v2::kInternalFaUpdateOpName);
  }

 private:
  int32_t fa_update_type_{0};
  bool sp_{1};
};

std::vector<ms::Tensor> npu_fa_update(const ms::Tensor &lse, const ms::Tensor &local_out, const uint64_t fa_update_type,
                                      const uint64_t sp) {
  auto op_name = "FaUpdate";
  auto runner = std::make_shared<FaUpdateRunner>(op_name);
  MS_EXCEPTION_IF_NULL(runner);

  runner->SetFaUpdateType(fa_update_type);
  runner->SetSp(sp);

  runner->param_.fa_update_type = fa_update_type;
  runner->param_.sp = sp;

  // Setup the runner with all parameters (including hash calculation)
  runner->Setup(op_name, lse, local_out, fa_update_type, sp);
  std::vector<ms::Tensor> inputs = {lse, local_out};
  MS_CHECK_VALUE(
    local_out.shape().size() == kFaUpdateLocalOutShapeRank,
    CheckAndConvertUtils::FormatCommMsg("For FaUpdate, local_out dim must be 3, but got : ", local_out.shape().size()));
  auto head_size = local_out.shape()[kIndex2];
  MS_CHECK_VALUE(
    head_size >= FA_UPDATA_HEAD_SIZE_MIN && head_size <= FA_UPDATA_HEAD_SIZE_MAX && ALIGN_8(head_size),
    CheckAndConvertUtils::FormatCommMsg(
      "For FaUpdate, head_size must be in range [8, 512] and be the multiple of 8, but got : ", head_size));
  ShapeVector output_shape{local_out.shape()[kIndex1], local_out.shape()[kIndex2]};
  auto output_tensor = ms::Tensor(local_out.data_type(), output_shape);
  std::vector<ms::Tensor> outputs = {output_tensor};
  runner->GetOrCreateKernel(inputs, outputs);
  runner->Run(inputs, outputs);
  return outputs;
}

auto pyboost_fa_update(const ms::Tensor &lse, const ms::Tensor &local_out, const uint64_t fa_update_type,
                       const uint64_t sp) {
  return ms::pynative::PyboostRunner::Call<ms_custom_ops::kNumber1>(npu_fa_update, lse, local_out, fa_update_type, sp);
}
}  // namespace ms_custom_ops

MS_CUSTOM_OPS_EXTENSION_MODULE(m) {
  m.def("fa_update", &ms_custom_ops::pyboost_fa_update, "FaUpdate", pybind11::arg("lse"), pybind11::arg("local_out"),
        pybind11::arg("fa_update_type") = 0, pybind11::arg("sp") = 1);
}
