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
#include <set>
#include <string>
#include <vector>

#include "ops/framework/aclnn/graphmode/aclnn_kernel_mod.h"
#include "ops/framework/utils.h"

namespace ms_custom_ops {
enum ApplyRotaryPosEmbCopsInputIndex : size_t {
  kApplyRotaryPosEmbCopsQueryIndex = 0,
  kApplyRotaryPosEmbCopsKeyIndex,
  kApplyRotaryPosEmbCopsCosIndex,
  kApplyRotaryPosEmbCopsSinIndex,
  kApplyRotaryPosEmbCopsLayoutIndex,
  kApplyRotaryPosEmbCopsRotaryModeIndex,
  kApplyRotaryPosEmbCopsInputsNum,
};

enum ApplyRotaryPosEmbCopsEnum : size_t {
  kApplyRotaryPosEmbCopsShapeSize = 4,
};

enum ApplyRotaryPosEmbCopsLayoutMode : size_t {
  LAYOUT_INVALID = 0,
  LAYOUT_BSND_BSH = 1,
  LAYOUT_BNSD = 2,
  LAYOUT_SBND = 3,
};

static std::set<std::string> apply_rotary_pos_emb_cops_rotary_mode_set = {
  "half",
  "quarter",
  "interleave",
};

static std::set<std::string> apply_rotary_pos_emb_layout_mode_set = {
  "BSND",
  "BSH",
  "BNSD",
  "SBND",
};

static size_t GetRopeLayout(const std::string &layout_str) {
  if (layout_str == "BSH" || layout_str == "BSND") {
    return static_cast<size_t>(ApplyRotaryPosEmbCopsLayoutMode::LAYOUT_BSND_BSH);
  } else if (layout_str == "BNSD") {
    return static_cast<size_t>(ApplyRotaryPosEmbCopsLayoutMode::LAYOUT_BNSD);
  } else if (layout_str == "SBND") {
    return static_cast<size_t>(ApplyRotaryPosEmbCopsLayoutMode::LAYOUT_SBND);
  }
  return static_cast<size_t>(ApplyRotaryPosEmbCopsLayoutMode::LAYOUT_INVALID);
}

ShapeArray ApplyRotaryPosEmbCopsMakeShape(const ShapeVector query_shape, const ShapeVector key_shape,
                                          const ShapeVector cos_shape, const ShapeVector sin_shape) {
  MS_CHECK_VALUE(
    query_shape.size() == kApplyRotaryPosEmbCopsShapeSize,
    "For ApplyRotaryPosEmbCops, Query must be a 4D tensor, but got shape " + ShapeVectorToStr(query_shape));
  MS_CHECK_VALUE(key_shape.size() == kApplyRotaryPosEmbCopsShapeSize,
                 "For ApplyRotaryPosEmbCops, key must be a 4D tensor, but got shape " + ShapeVectorToStr(key_shape));
  MS_CHECK_VALUE(cos_shape.size() == kApplyRotaryPosEmbCopsShapeSize,
                 "For ApplyRotaryPosEmbCops, cos must be a 4D tensor, but got shape " + ShapeVectorToStr(cos_shape));
  MS_CHECK_VALUE(sin_shape.size() == kApplyRotaryPosEmbCopsShapeSize,
                 "For ApplyRotaryPosEmbCops, sin must be a 4D tensor, but got shape " + ShapeVectorToStr(sin_shape));
  return kFakeOutTensorShapes;
}

class OPS_API ApplyRotaryPosEmbCopsCustomOpFuncImpl : public OpFuncImpl {
 public:
  ShapeArray InferShape(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const override {
    MS_EXCEPTION_IF_NULL(primitive);
    if (input_infos[kApplyRotaryPosEmbCopsQueryIndex]->IsDynamicRank() ||
        input_infos[kApplyRotaryPosEmbCopsKeyIndex]->IsDynamicRank() ||
        input_infos[kApplyRotaryPosEmbCopsCosIndex]->IsDynamicRank() ||
        input_infos[kApplyRotaryPosEmbCopsSinIndex]->IsDynamicRank()) {
      return kFakeOutTensorShapes;
    }

    return ApplyRotaryPosEmbCopsMakeShape(input_infos[kApplyRotaryPosEmbCopsQueryIndex]->GetShape(),
                                          input_infos[kApplyRotaryPosEmbCopsKeyIndex]->GetShape(),
                                          input_infos[kApplyRotaryPosEmbCopsCosIndex]->GetShape(),
                                          input_infos[kApplyRotaryPosEmbCopsSinIndex]->GetShape());
  }

  std::vector<TypeId> InferType(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const override {
    MS_EXCEPTION_IF_NULL(primitive);
    auto op_name = primitive->name();
    auto query_dtype = input_infos[kApplyRotaryPosEmbCopsQueryIndex]->GetType();
    auto key_dtype = input_infos[kApplyRotaryPosEmbCopsKeyIndex]->GetType();
    auto cos_dtype = input_infos[kApplyRotaryPosEmbCopsCosIndex]->GetType();
    auto sin_dtype = input_infos[kApplyRotaryPosEmbCopsSinIndex]->GetType();
    auto ms_context = MsContext::GetInstance();
    MS_EXCEPTION_IF_NULL(ms_context);
    const auto &soc_version = ms_context->ascend_soc_version();

    if (soc_version == kAscendVersion910_93 || soc_version == kAscendVersion910b) {
      const std::set<TypeId> valid_types = {kNumberTypeFloat16, kNumberTypeBFloat16, kNumberTypeFloat32};
      CheckAndConvertUtils::CheckTypeIdValid("query", query_dtype, valid_types, op_name);
      CheckAndConvertUtils::CheckTypeIdValid("key", key_dtype, valid_types, op_name);
      CheckAndConvertUtils::CheckTypeIdValid("cos", cos_dtype, valid_types, op_name);
      CheckAndConvertUtils::CheckTypeIdValid("sin", sin_dtype, valid_types, op_name);
    } else if (soc_version == kAscendVersion310p) {
      const std::set<TypeId> valid_types = {kNumberTypeFloat16, kNumberTypeFloat32};
      CheckAndConvertUtils::CheckTypeIdValid("query", query_dtype, valid_types, op_name);
      CheckAndConvertUtils::CheckTypeIdValid("key", key_dtype, valid_types, op_name);
      CheckAndConvertUtils::CheckTypeIdValid("cos", cos_dtype, valid_types, op_name);
      CheckAndConvertUtils::CheckTypeIdValid("sin", sin_dtype, valid_types, op_name);
    } else {
      MS_LOG(EXCEPTION) << "'ApplyRotaryPosEmbCops' only support [" << kAscendVersion910b << ", "
                        << kAscendVersion910_93 << ", " << kAscendVersion310p << "], but got " << soc_version;
    }
    return kFakeOutTensorTypes;
  }

  bool GeneralInferRegistered() const override { return true; }
};

class ApplyRotaryPosEmbCopsCustomAscend : public AclnnCustomKernelMod {
 public:
  ApplyRotaryPosEmbCopsCustomAscend() : AclnnCustomKernelMod("aclnnApplyRotaryPosEmbV2") {}
  ~ApplyRotaryPosEmbCopsCustomAscend() = default;

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs, void *stream_ptr) override {
    MS_EXCEPTION_IF_NULL(stream_ptr);
    RunOp(stream_ptr, workspace, inputs[kApplyRotaryPosEmbCopsQueryIndex], inputs[kApplyRotaryPosEmbCopsKeyIndex],
          inputs[kApplyRotaryPosEmbCopsCosIndex], inputs[kApplyRotaryPosEmbCopsSinIndex], layout_, rotary_mode_);
    return true;
  }

  void GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                        const std::vector<KernelTensor *> &outputs) override {
    auto layout_str = inputs[kApplyRotaryPosEmbCopsLayoutIndex]->GetValueWithCheck<std::string>();
    layout_ = GetRopeLayout(layout_str);
    rotary_mode_ = inputs[kApplyRotaryPosEmbCopsRotaryModeIndex]->GetValueWithCheck<std::string>();
    GetWorkspaceForResize(inputs[kApplyRotaryPosEmbCopsQueryIndex], inputs[kApplyRotaryPosEmbCopsKeyIndex],
                          inputs[kApplyRotaryPosEmbCopsCosIndex], inputs[kApplyRotaryPosEmbCopsSinIndex], layout_,
                          rotary_mode_);
    return;
  }

 private:
  DEFINE_GET_WORKSPACE_FOR_RESIZE();
  size_t layout_ = ApplyRotaryPosEmbCopsLayoutMode::LAYOUT_INVALID;
  std::string rotary_mode_ = "half";
  static constexpr int64_t bsnd_layout_ = 1;
};
}  // namespace ms_custom_ops

REG_GRAPH_MODE_OP(apply_rotary_pos_emb_cops, ms_custom_ops::ApplyRotaryPosEmbCopsCustomOpFuncImpl,
                  ms_custom_ops::ApplyRotaryPosEmbCopsCustomAscend);

// =============================================================================
// PYBOOST MODE IMPLEMENTATION
// =============================================================================

namespace ms_custom_ops {
void apply_rotary_pos_emb_cops_custom(const ms::Tensor &query, const ms::Tensor &key, const ms::Tensor &cos,
                                      const ms::Tensor &sin, const std::string &layout_str,
                                      const std::string &rotary_mode) {
  (void)ApplyRotaryPosEmbCopsMakeShape(query.shape(), key.shape(), cos.shape(), sin.shape());
  auto layout_mode = GetRopeLayout(layout_str);
  auto runner = std::make_shared<ms::pynative::AclnnOpRunner>("aclnnApplyRotaryPosEmbV2");
  runner->SetLaunchFunc(LAUNCH_ACLNN_FUNC(aclnnApplyRotaryPosEmbV2, query, key, cos, sin, layout_mode, rotary_mode));
  // only set tensor.
  runner->Run({query, key, cos, sin}, {});
  return;
}
}  // namespace ms_custom_ops

auto pyboost_apply_rotary_pos_emb_cops(const ms::Tensor &query, const ms::Tensor &key, const ms::Tensor &cos,
                                       const ms::Tensor &sin, const std::string &layout_str,
                                       const std::string &rotary_mode) {
  return ms::pynative::PyboostRunner::Call<ms_custom_ops::kNumber0>(ms_custom_ops::apply_rotary_pos_emb_cops_custom,
                                                                    query, key, cos, sin, layout_str, rotary_mode);
}

MS_CUSTOM_OPS_EXTENSION_MODULE(m) {
  m.def("apply_rotary_pos_emb_cops", &pyboost_apply_rotary_pos_emb_cops, "ApplyRotaryPosEmbCops",
        pybind11::arg("query"), pybind11::arg("key"), pybind11::arg("cos"), pybind11::arg("sin"),
        pybind11::arg("layout"), pybind11::arg("rotary_mode"));
}
