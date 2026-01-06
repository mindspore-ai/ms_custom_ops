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
#include "ops/framework/ms_kernels_internal/graphmode/internal_kernel_mod.h"
#include "ops/framework/utils.h"

namespace ms_custom_ops {
constexpr uint32_t APPLY_ROTARY_POS_EMB_ATB_QK_SUPPORT_DIM1 = 2;
constexpr uint32_t APPLY_ROTARY_POS_EMB_ATB_QK_SUPPORT_DIM2 = 4;
constexpr uint32_t APPLY_ROTARY_POS_EMB_ATB_COS_SIN_SUPPORT_DIM = 2;
constexpr uint32_t APPLY_ROTARY_POS_EMB_ATB_SEQLEN_SUPPORT_DIM = 1;
constexpr uint32_t APPLY_ROTARY_POS_EMB_ATB_ROTARY_COEFF_DEFAULT = 4;
enum class ApplyRotaryPosEmbATBInputIndex : size_t {
  kApplyRotaryPosEmbATBQueryIndex = 0,
  kApplyRotaryPosEmbATBKeyIndex,
  kApplyRotaryPosEmbATBCosIndex,
  kApplyRotaryPosEmbATBSinIndex,
  kApplyRotaryPosEmbATBSeqLenIndex,
  kApplyRotaryPosEmbATBRotaryCoeffIndex,
  kApplyRotaryPosEmbATBCosFormatIndex,
  kApplyRotaryPosEmbATBInputsNum,
};

enum class ApplyRotaryPosEmbATBOutputIndex : size_t {
  kApplyRotaryPosEmbATBQueryIndex = 0,
  kApplyRotaryPosEmbATBKeyIndex,
  kApplyRotaryPosEmbATBOutputsNum,
};

class OPS_API ApplyRotaryPosEmbATBOpFuncImpl : public OpFuncImpl {
 public:
  ShapeArray InferShape(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const override {
    if (input_infos[static_cast<size_t>(ApplyRotaryPosEmbATBInputIndex::kApplyRotaryPosEmbATBQueryIndex)]
          ->IsDynamicRank() ||
        input_infos[static_cast<size_t>(ApplyRotaryPosEmbATBInputIndex::kApplyRotaryPosEmbATBKeyIndex)]
          ->IsDynamicRank() ||
        input_infos[static_cast<size_t>(ApplyRotaryPosEmbATBInputIndex::kApplyRotaryPosEmbATBCosIndex)]
          ->IsDynamicRank() ||
        input_infos[static_cast<size_t>(ApplyRotaryPosEmbATBInputIndex::kApplyRotaryPosEmbATBSinIndex)]
          ->IsDynamicRank() ||
        input_infos[static_cast<size_t>(ApplyRotaryPosEmbATBInputIndex::kApplyRotaryPosEmbATBSeqLenIndex)]
          ->IsDynamicRank()) {
      return {
        input_infos[static_cast<size_t>(ApplyRotaryPosEmbATBInputIndex::kApplyRotaryPosEmbATBQueryIndex)]->GetShape(),
        input_infos[static_cast<size_t>(ApplyRotaryPosEmbATBInputIndex::kApplyRotaryPosEmbATBKeyIndex)]->GetShape()};
    }
    MS_EXCEPTION_IF_NULL(primitive);
    auto op_name = primitive->name();
    auto query_shape =
      input_infos[static_cast<size_t>(ApplyRotaryPosEmbATBInputIndex::kApplyRotaryPosEmbATBQueryIndex)]->GetShape();
    auto key_shape =
      input_infos[static_cast<size_t>(ApplyRotaryPosEmbATBInputIndex::kApplyRotaryPosEmbATBKeyIndex)]->GetShape();
    auto cos_shape =
      input_infos[static_cast<size_t>(ApplyRotaryPosEmbATBInputIndex::kApplyRotaryPosEmbATBCosIndex)]->GetShape();
    auto sin_shape =
      input_infos[static_cast<size_t>(ApplyRotaryPosEmbATBInputIndex::kApplyRotaryPosEmbATBSinIndex)]->GetShape();
    auto seqlen_shape =
      input_infos[static_cast<size_t>(ApplyRotaryPosEmbATBInputIndex::kApplyRotaryPosEmbATBSeqLenIndex)]->GetShape();
    if (query_shape.size() != key_shape.size() || cos_shape.size() != sin_shape.size()) {
      MS_LOG(EXCEPTION) << op_name
                        << " query's dim should be equal key's dim, and cos's dim should be equal sin's dim, but got "
                        << "query.dim=" << query_shape.size() << ", key.dim=" << key_shape.size()
                        << ", cos.dim=" << cos_shape.size() << ", sin.dim=" << sin_shape.size();
    }
    if (query_shape.size() != APPLY_ROTARY_POS_EMB_ATB_QK_SUPPORT_DIM1 &&
          query_shape.size() != APPLY_ROTARY_POS_EMB_ATB_QK_SUPPORT_DIM2 ||
        cos_shape.size() != APPLY_ROTARY_POS_EMB_ATB_COS_SIN_SUPPORT_DIM ||
        seqlen_shape.size() != APPLY_ROTARY_POS_EMB_ATB_SEQLEN_SUPPORT_DIM) {
      MS_LOG(EXCEPTION) << op_name
                        << " query/key's dim should be 2 or 4, cos/sin's dim should be 2, seqlen's dim should be 1"
                        << ", but got query/key's dim is " << query_shape.size() << ", cos/sin's dim is "
                        << cos_shape.size() << ", seqlen's dim is " << seqlen_shape.size();
    }
    auto query_tokens = query_shape.size() == 2 ? query_shape[0] : query_shape[0] * query_shape[1];
    auto key_tokens = key_shape.size() == 2 ? key_shape[0] : key_shape[0] * key_shape[1];
    if (query_tokens != key_tokens) {
      MS_LOG(EXCEPTION) << op_name << " tokens of query/key should be same, but got query's shape is " << query_shape
                        << ", key's shape is " << key_shape;
    }
    return {query_shape, key_shape};
  }

  std::vector<TypeId> InferType(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const override {
    MS_EXCEPTION_IF_NULL(primitive);
    auto op_name = primitive->name();
    auto query_type =
      input_infos[static_cast<size_t>(ApplyRotaryPosEmbATBInputIndex::kApplyRotaryPosEmbATBQueryIndex)]->GetType();
    auto key_type =
      input_infos[static_cast<size_t>(ApplyRotaryPosEmbATBInputIndex::kApplyRotaryPosEmbATBKeyIndex)]->GetType();
    auto cos_type =
      input_infos[static_cast<size_t>(ApplyRotaryPosEmbATBInputIndex::kApplyRotaryPosEmbATBCosIndex)]->GetType();
    auto sin_type =
      input_infos[static_cast<size_t>(ApplyRotaryPosEmbATBInputIndex::kApplyRotaryPosEmbATBSinIndex)]->GetType();
    if (query_type != key_type || cos_type != sin_type) {
      MS_LOG(EXCEPTION) << op_name
                        << ", the type should be query.dtype==key.dtype and cos.dtype==sin.dtype, but got query.dtype="
                        << TypeIdToString(query_type) << ", key.dtype=" << TypeIdToString(key_type)
                        << ", cos.dtype=" << TypeIdToString(cos_type) << "sin.dtype=" << TypeIdToString(sin_type);
    }
    if (query_type != kNumberTypeFloat16 && query_type != kNumberTypeBFloat16) {
      MS_LOG(EXCEPTION) << op_name << ", The type of query/key should be float16/bfloat16, but got "
                        << TypeIdToString(query_type);
    }
    if (query_type == kNumberTypeFloat16 && !(cos_type == kNumberTypeFloat16 || cos_type == kNumberTypeFloat32)) {
      MS_LOG(EXCEPTION) << op_name << ", The type of query/key is " << TypeIdToString(query_type)
                        << ", the type of cos/sin should be float16/float32, but got " << TypeIdToString(cos_type);
    }
    if (query_type == kNumberTypeBFloat16 && !(cos_type == kNumberTypeBFloat16 || cos_type == kNumberTypeFloat32)) {
      MS_LOG(EXCEPTION) << op_name << ", The type of query/key is " << TypeIdToString(query_type)
                        << ", the type of cos/sin should be bfloat16/float32, but got " << TypeIdToString(cos_type);
    }
    return {query_type, key_type};
  }

  bool GeneralInferRegistered() const override { return true; }
};

class ApplyRotaryPosEmbATB : public InternalKernelMod {
 public:
  ApplyRotaryPosEmbATB() : InternalKernelMod() {}
  ~ApplyRotaryPosEmbATB() = default;

  void InitKernelInputsOutputsIndex() override {
    kernel_inputs_index_ = {static_cast<size_t>(ApplyRotaryPosEmbATBInputIndex::kApplyRotaryPosEmbATBQueryIndex),
                            static_cast<size_t>(ApplyRotaryPosEmbATBInputIndex::kApplyRotaryPosEmbATBKeyIndex),
                            static_cast<size_t>(ApplyRotaryPosEmbATBInputIndex::kApplyRotaryPosEmbATBCosIndex),
                            static_cast<size_t>(ApplyRotaryPosEmbATBInputIndex::kApplyRotaryPosEmbATBSinIndex),
                            static_cast<size_t>(ApplyRotaryPosEmbATBInputIndex::kApplyRotaryPosEmbATBSeqLenIndex)};
    kernel_outputs_index_ = {static_cast<size_t>(ApplyRotaryPosEmbATBOutputIndex::kApplyRotaryPosEmbATBQueryIndex),
                             static_cast<size_t>(ApplyRotaryPosEmbATBOutputIndex::kApplyRotaryPosEmbATBKeyIndex)};
  }

 protected:
  internal_v2::InternalOpPtr CreateKernel(const internal_v2::InputsImmutableInfoList &inputs,
                                          const internal_v2::OutputsImmutableInfoList &outputs,
                                          const std::vector<KernelTensor *> &ms_inputs,
                                          const std::vector<KernelTensor *> &ms_outputs) override {
    internal_v2::RopeParam param;
    param.rotary_coeff = static_cast<int32_t>(
      ms_inputs.at(static_cast<size_t>(ApplyRotaryPosEmbATBInputIndex::kApplyRotaryPosEmbATBRotaryCoeffIndex))
        ->GetValue<int64_t>()
        .value());
    param.cos_format = static_cast<int32_t>(
      ms_inputs.at(static_cast<size_t>(ApplyRotaryPosEmbATBInputIndex::kApplyRotaryPosEmbATBCosFormatIndex))
        ->GetValue<int64_t>()
        .value());
    return internal_v2::CreateRopeOp(inputs, outputs, param, internal_v2::kInternalRopeOpName);
  }
};
}  // namespace ms_custom_ops

REG_GRAPH_MODE_OP(apply_rotary_pos_emb_atb, ms_custom_ops::ApplyRotaryPosEmbATBOpFuncImpl,
                  ms_custom_ops::ApplyRotaryPosEmbATB);

// =============================================================================
// PYBOOST MODE IMPLEMENTATION
// =============================================================================

#include "ops/framework/ms_kernels_internal/pyboost/internal_pyboost_runner.h"

namespace ms_custom_ops {
class ApplyRotaryPosEmbATBRunner : public InternalPyboostRunner {
 public:
  using InternalPyboostRunner::InternalPyboostRunner;

  void SetRotaryCoeff(const int32_t &rotary_coeff, const int32_t &cos_format) {
    this->rotary_coeff_ = rotary_coeff;
    this->cos_format_ = cos_format;
  }

 protected:
  internal_v2::InternalOpPtr CreateKernel(const internal_v2::InputsImmutableInfoList &inputs,
                                          const internal_v2::OutputsImmutableInfoList &outputs) override {
    internal_v2::RopeParam param;
    param.rotary_coeff = this->rotary_coeff_;
    param.cos_format = this->cos_format_;
    return internal_v2::CreateRopeOp(inputs, outputs, param, internal_v2::kInternalRopeOpName);
  }

 private:
  int32_t rotary_coeff_{APPLY_ROTARY_POS_EMB_ATB_ROTARY_COEFF_DEFAULT};
  int32_t cos_format_{0};
};

std::vector<ms::Tensor> npu_rope(const ms::Tensor &query, const ms::Tensor &key, const ms::Tensor &cos,
                                 const ms::Tensor &sin, const ms::Tensor &seq_len, int64_t rotary_coeff,
                                 int64_t cos_format) {
  auto op_name = "ApplyRotaryPosEmbATB";
  auto runner = std::make_shared<ApplyRotaryPosEmbATBRunner>(op_name);
  MS_EXCEPTION_IF_NULL(runner);

  // Set cos_format if provided
  runner->SetRotaryCoeff(static_cast<int32_t>(rotary_coeff), static_cast<int32_t>(cos_format));

  // Setup the runner with all parameters (including hash calculation)
  runner->Setup(op_name, query, key, cos, sin, seq_len, rotary_coeff, cos_format);

  // if you need infer shape and type, you can use this
  std::vector<ms::Tensor> inputs = {query, key, cos, sin, seq_len};
  std::vector<ms::Tensor> outputs = {ms::Tensor(query.data_type(), query.shape()),
                                     ms::Tensor(key.data_type(), key.shape())};
  runner->GetOrCreateKernel(inputs, outputs);
  runner->Run(inputs, outputs);
  return outputs;
}
}  // namespace ms_custom_ops

auto pyboost_apply_rotary_pos_emb_atb(const ms::Tensor &query, const ms::Tensor &key, const ms::Tensor &cos,
                                      const ms::Tensor &sin, const ms::Tensor &seq_len, int64_t rotary_coeff,
                                      int64_t cos_format) {
  return ms::pynative::PyboostRunner::Call<ms_custom_ops::kNumber2>(ms_custom_ops::npu_rope, query, key, cos, sin,
                                                                    seq_len, rotary_coeff, cos_format);
}

MS_CUSTOM_OPS_EXTENSION_MODULE(m) {
  m.def("apply_rotary_pos_emb_atb", &pyboost_apply_rotary_pos_emb_atb, "ApplyRotaryPosEmbATB", pybind11::arg("query"),
        pybind11::arg("key"), pybind11::arg("cos"), pybind11::arg("sin"), pybind11::arg("seq_len"),
        pybind11::arg("rotary_coeff") = 4, pybind11::arg("cos_format") = 0);
}
