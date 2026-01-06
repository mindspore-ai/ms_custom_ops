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
#include <vector>
#include <unordered_set>
#include <utility>

#include "ops/framework/aclnn/graphmode/aclnn_kernel_mod.h"
#include "ops/framework/utils.h"
#include "mindspore/include/custom_op_api.h"

namespace ms_custom_ops {
// 约束条件: rotary_dim = 2 * cos_head_dim, query_head_dim >= rotary_dim
constexpr uint32_t ROTARY_DIM_FACTOR = 2;
constexpr int32_t LAYOUT_BSH = 1;
constexpr const char *LAYOUT_BSH_STR = "BSH";
constexpr const char *ROTARY_INTERLEAVE_STR = "interleave";
enum class ApplyRotaryPosEmbMSInputIndex : size_t {
  kApplyRotaryPosEmbMSQueryIndex = 0,
  kApplyRotaryPosEmbMSKeyIndex,
  kApplyRotaryPosEmbMSCosIndex,
  kApplyRotaryPosEmbMSSinIndex,
  kApplyRotaryPosEmbMSLayoutIndex,
  kApplyRotaryPosEmbMSRotaryModeIndex,
  kApplyRotaryPosEmbMSInputsNum,
};

static void ApplyRotaryPosEmbMSCheckInputsShape(const std::string &op_name, const std::vector<int64_t> &query_shape,
                                                const std::vector<int64_t> &key_shape,
                                                const std::vector<int64_t> &cos_shape,
                                                const std::vector<int64_t> &sin_shape) {
  if (query_shape.size() != kDim3 || key_shape.size() != kDim3 || cos_shape.size() != kDim2 ||
      sin_shape.size() != kDim2) {
    MS_LOG(EXCEPTION) << op_name << ", the dim of inputs should be query.dim=key.dim=3, "
                      << "cos.dim=sin.dim=2, but got query.dim=" << query_shape.size()
                      << ", key.dim=" << key_shape.size() << ", cos.dim=" << cos_shape.size()
                      << ", sin.dim=" << sin_shape.size();
  }
  MS_CHECK_VALUE(query_shape[kIndex2] == key_shape[kIndex2] && query_shape[kIndex0] == key_shape[kIndex0],
                 CheckAndConvertUtils::FormatCommMsg(
                   op_name, ", query.dim0 should be equal key.dim0, query.dim2 should be equal key.dim2,",
                   " but got query.shape=", query_shape, ", key.shape=", key_shape));
  MS_CHECK_VALUE(
    cos_shape == sin_shape,
    CheckAndConvertUtils::FormatCommMsg(
      op_name, ", cos.shape should be equals sin.shape, but got cos.shape=", cos_shape, ", sin.shape=", sin_shape));
  MS_CHECK_VALUE(
    query_shape[kIndex2] >= ROTARY_DIM_FACTOR * cos_shape[kIndex1],
    CheckAndConvertUtils::FormatCommMsg(
      op_name, ", the head_dim of query and key should be greater than or equal to twice head_dim of cos or sin,",
      " but got query.shape=", query_shape, ", cos.shape=", cos_shape));
  MS_CHECK_VALUE(query_shape[kIndex0] == cos_shape[kIndex0],
                 CheckAndConvertUtils::FormatCommMsg(
                   op_name, ", query/key's dim0 should be equal cos/sin's dim0, but got query's shape is ", query_shape,
                   ", cos's shape is ", cos_shape));
}
static void ApplyRotaryPosEmbMSCheckInputsType(const std::string &op_name, const TypeId &query_dtype,
                                               const TypeId &key_dtype, const TypeId &cos_dtype,
                                               const TypeId &sin_dtype) {
  const std::unordered_set<TypeId> valid_types = {kNumberTypeFloat16, kNumberTypeFloat32};
  std::unordered_set<TypeId> input_types = {query_dtype, key_dtype, cos_dtype, sin_dtype};
  if (input_types.size() > 1) {
    MS_LOG(EXCEPTION) << op_name << ", the dtype of 'query, key, cos, sin' should be same, but got '"
                      << TypeIdToString(query_dtype) << ", " << TypeIdToString(key_dtype) << ", "
                      << TypeIdToString(cos_dtype) << ", " << TypeIdToString(sin_dtype) << "'";
  }
  if (valid_types.find(query_dtype) == valid_types.end()) {
    MS_LOG(EXCEPTION) << op_name << ", the dtype of 'query, key, cos, sin' should be "
                      << TypeIdToString(kNumberTypeFloat16) << " or " << TypeIdToString(kNumberTypeFloat32)
                      << ", but got '" << TypeIdToString(query_dtype) << ", " << TypeIdToString(key_dtype) << ", "
                      << TypeIdToString(cos_dtype) << ", " << TypeIdToString(sin_dtype) << "'";
  }
}
class OPS_API ApplyRotaryPosEmbMSOpFuncImpl : public OpFuncImpl {
 public:
  ShapeArray InferShape(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const override {
    MS_EXCEPTION_IF_NULL(primitive);
    if (input_infos[static_cast<size_t>(ApplyRotaryPosEmbMSInputIndex::kApplyRotaryPosEmbMSQueryIndex)]
          ->IsDynamicRank() ||
        input_infos[static_cast<size_t>(ApplyRotaryPosEmbMSInputIndex::kApplyRotaryPosEmbMSKeyIndex)]
          ->IsDynamicRank() ||
        input_infos[static_cast<size_t>(ApplyRotaryPosEmbMSInputIndex::kApplyRotaryPosEmbMSCosIndex)]
          ->IsDynamicRank() ||
        input_infos[static_cast<size_t>(ApplyRotaryPosEmbMSInputIndex::kApplyRotaryPosEmbMSSinIndex)]
          ->IsDynamicRank()) {
      return kFakeOutTensorShapes;
    }
    auto op_name = primitive->name();
    auto query_shape =
      input_infos[static_cast<size_t>(ApplyRotaryPosEmbMSInputIndex::kApplyRotaryPosEmbMSQueryIndex)]->GetShape();
    auto key_shape =
      input_infos[static_cast<size_t>(ApplyRotaryPosEmbMSInputIndex::kApplyRotaryPosEmbMSKeyIndex)]->GetShape();
    auto cos_shape =
      input_infos[static_cast<size_t>(ApplyRotaryPosEmbMSInputIndex::kApplyRotaryPosEmbMSCosIndex)]->GetShape();
    auto sin_shape =
      input_infos[static_cast<size_t>(ApplyRotaryPosEmbMSInputIndex::kApplyRotaryPosEmbMSSinIndex)]->GetShape();
    auto rotary_mode =
      input_infos[static_cast<size_t>(ApplyRotaryPosEmbMSInputIndex::kApplyRotaryPosEmbMSRotaryModeIndex)]
        ->GetScalarValueWithCheck<string>();
    auto layout = input_infos[static_cast<size_t>(ApplyRotaryPosEmbMSInputIndex::kApplyRotaryPosEmbMSLayoutIndex)]
                    ->GetScalarValueWithCheck<string>();
    MS_CHECK_VALUE(layout == LAYOUT_BSH_STR,
                   CheckAndConvertUtils::FormatCommMsg(op_name, " layout should be 'BSH', but got ", layout));
    MS_CHECK_VALUE(
      rotary_mode == ROTARY_INTERLEAVE_STR,
      CheckAndConvertUtils::FormatCommMsg(op_name, " rotary_mode should be 'interleave', but got ", rotary_mode));
    ApplyRotaryPosEmbMSCheckInputsShape(op_name, query_shape, key_shape, cos_shape, sin_shape);
    // 复写算子, 无输出. 此处的输出与yaml中算子定义的return相对应, 无实际意义.
    return kFakeOutTensorShapes;
  }

  std::vector<TypeId> InferType(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const override {
    auto op_name = primitive->name();
    auto query_dtype =
      input_infos[static_cast<size_t>(ApplyRotaryPosEmbMSInputIndex::kApplyRotaryPosEmbMSQueryIndex)]->GetType();
    auto key_dtype =
      input_infos[static_cast<size_t>(ApplyRotaryPosEmbMSInputIndex::kApplyRotaryPosEmbMSKeyIndex)]->GetType();
    auto cos_dtype =
      input_infos[static_cast<size_t>(ApplyRotaryPosEmbMSInputIndex::kApplyRotaryPosEmbMSCosIndex)]->GetType();
    auto sin_dtype =
      input_infos[static_cast<size_t>(ApplyRotaryPosEmbMSInputIndex::kApplyRotaryPosEmbMSSinIndex)]->GetType();
    ApplyRotaryPosEmbMSCheckInputsType(op_name, query_dtype, key_dtype, cos_dtype, sin_dtype);
    // 复写算子, 无输出. 此处的输出与yaml中算子定义的return相对应, 无实际意义.
    return kFakeOutTensorTypes;
  }

  bool GeneralInferRegistered() const override { return true; }
};

class ApplyRotaryPosEmbMSAscend : public AclnnCustomKernelMod {
 public:
  ApplyRotaryPosEmbMSAscend() : AclnnCustomKernelMod(std::move("aclnnApplyRotaryPosEmbMS")) {}
  ~ApplyRotaryPosEmbMSAscend() = default;

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs, void *stream_ptr) override {
    MS_EXCEPTION_IF_NULL(stream_ptr);
    RunOp(
      stream_ptr, workspace, inputs[static_cast<size_t>(ApplyRotaryPosEmbMSInputIndex::kApplyRotaryPosEmbMSQueryIndex)],
      inputs[static_cast<size_t>(ApplyRotaryPosEmbMSInputIndex::kApplyRotaryPosEmbMSKeyIndex)],
      inputs[static_cast<size_t>(ApplyRotaryPosEmbMSInputIndex::kApplyRotaryPosEmbMSCosIndex)],
      inputs[static_cast<size_t>(ApplyRotaryPosEmbMSInputIndex::kApplyRotaryPosEmbMSSinIndex)], layout_, rotary_mode_);
    return true;
  }
  void GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                        const std::vector<KernelTensor *> &outputs) override {
    auto layout_str = inputs[static_cast<size_t>(ApplyRotaryPosEmbMSInputIndex::kApplyRotaryPosEmbMSLayoutIndex)]
                        ->GetValueWithCheck<std::string>();
    if (layout_str == LAYOUT_BSH_STR) {
      layout_ = LAYOUT_BSH;
    } else {
      MS_LOG(EXCEPTION) << "layout should be 'BSH', but got " << layout_str;
    }
    rotary_mode_ = inputs[static_cast<size_t>(ApplyRotaryPosEmbMSInputIndex::kApplyRotaryPosEmbMSRotaryModeIndex)]
                     ->GetValueWithCheck<std::string>();
    GetWorkspaceForResize(inputs[static_cast<size_t>(ApplyRotaryPosEmbMSInputIndex::kApplyRotaryPosEmbMSQueryIndex)],
                          inputs[static_cast<size_t>(ApplyRotaryPosEmbMSInputIndex::kApplyRotaryPosEmbMSKeyIndex)],
                          inputs[static_cast<size_t>(ApplyRotaryPosEmbMSInputIndex::kApplyRotaryPosEmbMSCosIndex)],
                          inputs[static_cast<size_t>(ApplyRotaryPosEmbMSInputIndex::kApplyRotaryPosEmbMSSinIndex)],
                          layout_, rotary_mode_);
    return;
  }

 private:
  DEFINE_GET_WORKSPACE_FOR_RESIZE();
  int32_t layout_ = LAYOUT_BSH;
  std::string rotary_mode_ = ROTARY_INTERLEAVE_STR;
};
}  // namespace ms_custom_ops

REG_GRAPH_MODE_OP(apply_rotary_pos_emb_ms, ms_custom_ops::ApplyRotaryPosEmbMSOpFuncImpl,
                  ms_custom_ops::ApplyRotaryPosEmbMSAscend);

// =============================================================================
// PYBOOST MODE IMPLEMENTATION
// =============================================================================

namespace ms_custom_ops {
void npu_apply_rotary_pos_emb_ms(const ms::Tensor &query, const ms::Tensor &key, const ms::Tensor &cos,
                                 const ms::Tensor &sin, const std::string &layout_str, const std::string &rotary_mode) {
  std::string op_name = "ApplyRotaryPosEmbMS";
  // 此处op_name是给人看的, 跟算子命名没有直接关联
  auto runner = std::make_shared<ms::pynative::AclnnOpRunner>(op_name);
  // 输入shape检查
  ApplyRotaryPosEmbMSCheckInputsShape(op_name, query.shape(), key.shape(), cos.shape(), sin.shape());
  // 输入dtype检查
  ApplyRotaryPosEmbMSCheckInputsType(op_name, query.data_type(), key.data_type(), cos.data_type(), sin.data_type());
  // 此处"aclnnApplyRotaryPosEmbMS", 是算字库函数表中名字前面加上aclnn
  // 可通过 nm -D ./build/xxx/xxx/ms_custom_ops.xxx.so | grep "ApplyRotaryPosEmbMS"来确认
  // 如果是复写算子(inplace), 不必添加输出参数
  runner->SetLaunchFunc(LAUNCH_ACLNN_FUNC(aclnnApplyRotaryPosEmbMS, query, key, cos, sin, layout_str, rotary_mode));
  // 如果是复写算子(inplace), 输出参数为空
  runner->Run({query, key, cos, sin}, {});
  // 无输出的算子返回值用void(不同于静态图)
  return;
}
}  // namespace ms_custom_ops

auto pyboost_apply_rotary_pos_emb_ms(const ms::Tensor &query, const ms::Tensor &key, const ms::Tensor &cos,
                                     const ms::Tensor &sin, const std::string &layout_str,
                                     const std::string &rotary_mode) {
  return ms::pynative::PyboostRunner::Call<ms_custom_ops::kNumber0>(ms_custom_ops::npu_apply_rotary_pos_emb_ms, query,
                                                                    key, cos, sin, layout_str, rotary_mode);
}

MS_CUSTOM_OPS_EXTENSION_MODULE(m) {
  m.def("apply_rotary_pos_emb_ms", &pyboost_apply_rotary_pos_emb_ms, "ApplyRotaryPosEmbMS", pybind11::arg("query"),
        pybind11::arg("key"), pybind11::arg("cos"), pybind11::arg("sin"), pybind11::arg("layout"),
        pybind11::arg("rotary_mode"));
}
