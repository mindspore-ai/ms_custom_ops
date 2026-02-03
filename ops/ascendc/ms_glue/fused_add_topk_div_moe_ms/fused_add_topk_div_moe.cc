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
#include <iostream>

#include "ops/framework/aclnn/graphmode/aclnn_kernel_mod.h"
#include "ops/framework/utils.h"
#include "mindspore/include/custom_op_api.h"

namespace ms_custom_ops {
constexpr int64_t kFusedAddTopkDivMoeExpertNum = 128;
constexpr int64_t kFusedAddTopkDivMoeTopK = 8;
constexpr int64_t kFusedAddTopkDivMoeNumGroups = 1;
constexpr int64_t kFusedAddTopkDivMoeGroupTopK = 1;
constexpr int64_t kFusedAddTopkDivMoeTopkDivGroupTopK = 8;
constexpr int64_t kFusedAddTopkDivMoeActivateType = 0;
constexpr bool kFusedAddTopkDivMoeIsNorm = true;
constexpr float kFusedAddTopkDivMoeScale = 1.0f;
enum class FusedAddTopkDivMoeInputIndex : size_t {
  kFusedAddTopkDivMoeLogitsIndex = 0,
  kFusedAddTopkDivMoeBiasIndex,
  kFusedAddTopkDivMoeNumTokensIndex,
};

static void FusedAddTopkDivMoeCheckInputsShape(const std::string &op_name, const std::vector<int64_t> &logits_shape,
                                               const std::vector<int64_t> &bias_shape) {
  if (logits_shape.size() != kDim2 || bias_shape.size() != kDim1) {
    MS_LOG(EXCEPTION) << op_name << ", the dim of inputs should be logits.dim=2, bias.dim=1 "
                      << "but got logits.dim=" << logits_shape.size() << ", bias.dim=" << bias_shape.size();
  }
  MS_CHECK_VALUE(
    logits_shape[kIndex1] == bias_shape[kIndex0],
    CheckAndConvertUtils::FormatCommMsg(op_name, ", logits.dim1 should be logits.dim1 = bias.dim0 = 128,",
                                        " but got logits.shape=", logits_shape, ", bias.shape=", bias_shape));
  MS_CHECK_VALUE(logits_shape[kIndex1] == kFusedAddTopkDivMoeExpertNum,
                 CheckAndConvertUtils::FormatCommMsg(
                   op_name, ", logits.shape should be equals (num_tokens,128), but got logits.shape=", logits_shape));
}

static void FusedAddTopkDivMoeCheckInputsType(const std::string &op_name, const TypeId &logits_dtype,
                                              const TypeId &bias_dtype) {
  if (logits_dtype != kNumberTypeFloat32) {
    MS_LOG(EXCEPTION) << op_name << ", the dtype of 'logits' should be " << TypeIdToString(kNumberTypeFloat32)
                      << ", but got logits.dtype=" << TypeIdToString(logits_dtype);
  }
  if (bias_dtype != kNumberTypeFloat32) {
    MS_LOG(EXCEPTION) << op_name << ", the dtype of 'bias' should be " << TypeIdToString(kNumberTypeFloat32)
                      << ", but got bias.dtype=" << TypeIdToString(bias_dtype);
  }
}

class OPS_API FusedAddTopkDivMoeOpFuncImpl : public OpFuncImpl {
 public:
  ShapeArray InferShape(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const override {
    MS_EXCEPTION_IF_NULL(primitive);
    auto op_name = primitive->name();
    if (input_infos[static_cast<size_t>(FusedAddTopkDivMoeInputIndex::kFusedAddTopkDivMoeLogitsIndex)]
          ->IsDynamicRank() ||
        input_infos[static_cast<size_t>(FusedAddTopkDivMoeInputIndex::kFusedAddTopkDivMoeBiasIndex)]->IsDynamicRank()) {
      MS_LOG(EXCEPTION) << op_name << "FusedAddTopkDivMoeMS don't support dynamic rank inputs" << '\n';
    }
    auto logits_shape =
      input_infos[static_cast<size_t>(FusedAddTopkDivMoeInputIndex::kFusedAddTopkDivMoeLogitsIndex)]->GetShape();
    auto bias_shape =
      input_infos[static_cast<size_t>(FusedAddTopkDivMoeInputIndex::kFusedAddTopkDivMoeBiasIndex)]->GetShape();
    FusedAddTopkDivMoeCheckInputsShape(op_name, logits_shape, bias_shape);
    auto expert_weight_shape = logits_shape;
    expert_weight_shape[kIndex1] = kFusedAddTopkDivMoeTopK;
    auto expert_index_shape = expert_weight_shape;
    return {expert_weight_shape, expert_index_shape};
  }

  std::vector<TypeId> InferType(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const override {
    auto op_name = primitive->name();
    auto logits_dtype =
      input_infos[static_cast<size_t>(FusedAddTopkDivMoeInputIndex::kFusedAddTopkDivMoeLogitsIndex)]->GetType();
    auto bias_dtype =
      input_infos[static_cast<size_t>(FusedAddTopkDivMoeInputIndex::kFusedAddTopkDivMoeBiasIndex)]->GetType();
    FusedAddTopkDivMoeCheckInputsType(op_name, logits_dtype, bias_dtype);
    return {logits_dtype, kNumberTypeInt32};
  }

  bool GeneralInferRegistered() const override { return true; }
};

class FusedAddTopkDivMoe : public AclnnCustomKernelMod {
 public:
  FusedAddTopkDivMoe() : AclnnCustomKernelMod(std::move("aclnnFusedAddTopkDivMoeMS")) {}
  ~FusedAddTopkDivMoe() = default;

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs, void *stream_ptr) override {
    MS_EXCEPTION_IF_NULL(stream_ptr);
    RunOp(
      stream_ptr, workspace, inputs[static_cast<size_t>(FusedAddTopkDivMoeInputIndex::kFusedAddTopkDivMoeLogitsIndex)],
      inputs[static_cast<size_t>(FusedAddTopkDivMoeInputIndex::kFusedAddTopkDivMoeBiasIndex)], outputs[0], outputs[1]);
    return true;
  }
  void GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                        const std::vector<KernelTensor *> &outputs) override {
    GetWorkspaceForResize(inputs[static_cast<size_t>(FusedAddTopkDivMoeInputIndex::kFusedAddTopkDivMoeLogitsIndex)],
                          inputs[static_cast<size_t>(FusedAddTopkDivMoeInputIndex::kFusedAddTopkDivMoeBiasIndex)],
                          outputs[0], outputs[1]);
    return;
  }

 private:
  DEFINE_GET_WORKSPACE_FOR_RESIZE();
};
}  // namespace ms_custom_ops

REG_GRAPH_MODE_OP(fused_add_topk_div_moe, ms_custom_ops::FusedAddTopkDivMoeOpFuncImpl,
                  ms_custom_ops::FusedAddTopkDivMoe);

// =============================================================================
// PYBOOST MODE IMPLEMENTATION
// =============================================================================

namespace ms_custom_ops {
constexpr size_t kFusedAddTopkDivMoeOutputNum = 2;

static void FusedAddTopkDivMoeCheckScalarParams(const std::string &op_name, int num_groups, int group_topk,
                                                int topk_div_group_topk, int topk, int activate_type, bool is_norm,
                                                float scale) {
  MS_CHECK_VALUE(num_groups == kFusedAddTopkDivMoeNumGroups,
                 CheckAndConvertUtils::FormatCommMsg(op_name, ", num_groups should be ", kFusedAddTopkDivMoeNumGroups,
                                                    ", but got num_groups=", num_groups));
  MS_CHECK_VALUE(group_topk == kFusedAddTopkDivMoeGroupTopK,
                 CheckAndConvertUtils::FormatCommMsg(op_name, ", group_topk should be ", kFusedAddTopkDivMoeGroupTopK,
                                                    ", but got group_topk=", group_topk));
  MS_CHECK_VALUE(topk_div_group_topk == kFusedAddTopkDivMoeTopkDivGroupTopK,
                 CheckAndConvertUtils::FormatCommMsg(op_name, ", topk_div_group_topk should be ",
                                                    kFusedAddTopkDivMoeTopkDivGroupTopK,
                                                    ", but got topk_div_group_topk=", topk_div_group_topk));
  MS_CHECK_VALUE(topk == kFusedAddTopkDivMoeTopK,
                 CheckAndConvertUtils::FormatCommMsg(op_name, ", topk should be ", kFusedAddTopkDivMoeTopK,
                                                    ", but got topk=", topk));
  MS_CHECK_VALUE(activate_type == kFusedAddTopkDivMoeActivateType,
                 CheckAndConvertUtils::FormatCommMsg(op_name, ", activate_type should be ",
                                                    kFusedAddTopkDivMoeActivateType,
                                                    ", but got activate_type=", activate_type));
  MS_CHECK_VALUE(is_norm == kFusedAddTopkDivMoeIsNorm,
                 CheckAndConvertUtils::FormatCommMsg(op_name, ", is_norm should be ", kFusedAddTopkDivMoeIsNorm,
                                                    ", but got is_norm=", is_norm));
  MS_CHECK_VALUE(scale == kFusedAddTopkDivMoeScale,
                 CheckAndConvertUtils::FormatCommMsg(op_name, ", scale should be ", kFusedAddTopkDivMoeScale,
                                                    ", but got scale=", scale));
}

std::vector<ms::Tensor> fused_add_topk_div_moe_custom(const ms::Tensor &logits, const ms::Tensor &bias, int num_groups,
                                                      int group_topk, int topk_div_group_topk, int topk,
                                                      int activate_type, bool is_norm, float scale) {
  std::string op_name = "fused_add_topk_div_moe";
  FusedAddTopkDivMoeCheckScalarParams(op_name, num_groups, group_topk, topk_div_group_topk, topk, activate_type,
                                      is_norm, scale);
  auto runner = std::make_shared<ms::pynative::AclnnOpRunner>(op_name);
  auto logits_shape = logits.shape();
  auto expert_weight_shape = logits_shape;
  expert_weight_shape[kIndex1] = kFusedAddTopkDivMoeTopK;
  auto expert_index_shape = expert_weight_shape;
  FusedAddTopkDivMoeCheckInputsShape(op_name, logits.shape(), bias.shape());
  FusedAddTopkDivMoeCheckInputsType(op_name, logits.data_type(), bias.data_type());
  auto expert_weight = ms::Tensor(logits.data_type(), expert_weight_shape);
  auto expert_index = ms::Tensor(kNumberTypeInt32, expert_index_shape);

  runner->SetLaunchFunc(LAUNCH_ACLNN_FUNC(aclnnFusedAddTopkDivMoeMS, logits, bias, expert_weight, expert_index));
  runner->Run({logits, bias}, {expert_weight, expert_index});
  return {expert_weight, expert_index};
}
}  // namespace ms_custom_ops

auto pyboost_fused_add_topk_div_moe(const ms::Tensor &logits, const ms::Tensor &bias, const int64_t num_groups,
                                    const int64_t group_topk, const int64_t topk_div_group_topk, const int64_t topk,
                                    const int64_t activate_type, const bool is_norm, const float scale) {
  return ms::pynative::PyboostRunner::Call<ms_custom_ops::kFusedAddTopkDivMoeOutputNum>(
    ms_custom_ops::fused_add_topk_div_moe_custom, logits, bias, num_groups, group_topk, topk_div_group_topk, topk,
    activate_type, is_norm, scale);
}

MS_CUSTOM_OPS_EXTENSION_MODULE(m) {
  m.def("fused_add_topk_div_moe", &pyboost_fused_add_topk_div_moe, "FusedAddTopkDivMoe", pybind11::arg("logits"),
        pybind11::arg("bias"), pybind11::arg("num_groups") = 1, pybind11::arg("group_topk") = 1,
        pybind11::arg("topk_div_group_topk") = 8, pybind11::arg("topk") = 8, pybind11::arg("activate_type") = 0,
        pybind11::arg("is_norm") = true, pybind11::arg("scale") = 1.0);
}
