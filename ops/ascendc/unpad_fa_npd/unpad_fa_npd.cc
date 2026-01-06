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

#include <algorithm>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <unordered_set>
#include <vector>
#include "op_host/unpad_fa_npd.h"
// #include "kernel_common/op_host/npd/npd_common.h"
#include "ops/framework/aclnn/graphmode/aclnn_kernel_mod.h"
#include "ops/framework/utils.h"
#include "mindspore/include/custom_op_api.h"

using optiling::UnpadFaNpdInputIndex;
using optiling::UnpadFaNpdOutputIndex;

namespace ms_custom_ops {

template <typename T1, typename T2>
static inline std::vector<T2> GetSeqLenFromTensor(const mindspore::tensor::TensorPtr &seq_length_tensor) {
  if (seq_length_tensor != nullptr) {
    auto seq_length_values = static_cast<T1 *>(seq_length_tensor->data_c());
    auto seq_length_values_num = seq_length_tensor->DataSize();
    std::vector<T2> seq_len;
    seq_len.reserve(seq_length_values_num);
    std::transform(seq_length_values, seq_length_values + seq_length_values_num, std::back_inserter(seq_len),
                   [](T1 val) { return static_cast<T2>(val); });
    return seq_len;
  }
  return {};
}

template <typename T1, typename T2>
static inline std::vector<T2> CastVector(const std::vector<T1> &src) {
  auto elem_num = src.size();
  if (elem_num > 0) {
    std::vector<T2> dst;
    auto src_data = src.data();
    dst.reserve(elem_num);
    std::transform(src_data, src_data + elem_num, std::back_inserter(dst), [](T1 val) { return static_cast<T2>(val); });
    return dst;
  }
  return {};
}

static void UnpadFaNpdCheckInputsShape(const std::string &op_name, const std::vector<int64_t> &q_shape,
                                       const std::vector<int64_t> &k_shape, const std::vector<int64_t> &v_shape,
                                       const std::vector<int64_t> &act_q_seq_shape,
                                       const std::vector<int64_t> &act_kv_seq_shape) {
  if (q_shape.size() != kDim2 && q_shape.size() != kDim3) {
    MS_LOG(EXCEPTION) << op_name << ", q should be in TH or BSH layout";
  }

  if (k_shape.size() != v_shape.size()) {
    MS_LOG(EXCEPTION) << op_name << ", key dim number should be equal to value dim number, "
                      << "key.dim=" << k_shape.size() << "value.dim=" << v_shape.size();
  }

  if (k_shape.size() != kDim4 && k_shape.size() != kDim2) {
    MS_LOG(EXCEPTION) << op_name << ", key dim number  should be 4 or 2, but got "
                      << "key.dim=" << k_shape.size();
  }

  if (v_shape.size() != kDim4 && v_shape.size() != kDim2) {
    MS_LOG(EXCEPTION) << op_name << ", value dim number  should be 4 or 2, but got "
                      << "value.dim=" << k_shape.size();
  }

  if (act_q_seq_shape.size() != kDim1 && act_kv_seq_shape.size() != kDim1) {
    MS_LOG(EXCEPTION) << op_name
                      << ", the dim of inputs should be act_q_seq_shape.dim==act_kv_seq_shape.dim=1, but got "
                      << "act_q_seq_shape.dim=" << act_q_seq_shape.size()
                      << "act_kv_seq_shape.dim=" << act_kv_seq_shape.size();
  }
}

static void UnpadFaNpdCheckInputsType(const std::string &op_name, const TypeId &query_dtype, const TypeId &key_dtype,
                                      const TypeId &value_dtype, const TypeId &act_q_seq_dtype,
                                      const TypeId &act_kv_seq_dtype) {
  const std::unordered_set<TypeId> valid_types = {kNumberTypeFloat16, kNumberTypeBFloat16};
  std::unordered_set<TypeId> input_types = {query_dtype, key_dtype, value_dtype};
  if (input_types.size() > 1) {
    MS_LOG(EXCEPTION) << op_name << ", the dtype of 'query_dtype, key_dtype, value_dtype' should be same, but got '"
                      << TypeIdToString(query_dtype) << ", " << TypeIdToString(key_dtype) << ", "
                      << TypeIdToString(value_dtype) << "'";
  }
  if (valid_types.find(key_dtype) == valid_types.end()) {
    MS_LOG(EXCEPTION) << op_name << ", the dtype of 'query_dtype, key_dtype, value_dtype' should be "
                      << TypeIdToString(kNumberTypeFloat16) << " or " << TypeIdToString(kNumberTypeBFloat16)
                      << ", but got '" << TypeIdToString(query_dtype) << ", " << TypeIdToString(key_dtype) << ", "
                      << TypeIdToString(value_dtype) << ", " << "'";
  }
  const std::unordered_set<TypeId> valid_int_types = {kNumberTypeInt32, kNumberTypeInt};
  std::unordered_set<TypeId> input_int_types = {act_q_seq_dtype, act_kv_seq_dtype};
  if (input_int_types.size() > 1) {
    MS_LOG(EXCEPTION) << op_name << ", the dtype of 'act_q_seq_dtype, act_kv_seq_dtype' should be same, but got '"
                      << TypeIdToString(act_q_seq_dtype) << ", " << TypeIdToString(act_kv_seq_dtype) << "'";
  }
  if (valid_int_types.find(act_q_seq_dtype) == valid_int_types.end()) {
    MS_LOG(EXCEPTION) << op_name << ", the dtype of 'act_q_seq_dtype, act_kv_seq_dtype' should be "
                      << TypeIdToString(kNumberTypeInt32) << " or " << TypeIdToString(kNumberTypeInt) << ", but got '"
                      << TypeIdToString(act_q_seq_dtype) << ", " << TypeIdToString(act_kv_seq_dtype);
  }
}

static mindspore::ShapeVector UnpadFaNpdDoInferShape(mindspore::ShapeVector shape) {
  ShapeVector out_shape = shape;
  return out_shape;
}

class OPS_API UnpadFaNpdOpFuncImpl : public OpFuncImpl {
 public:
  ShapeArray InferShape(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const override {
    auto q_shape = input_infos[static_cast<size_t>(UnpadFaNpdInputIndex::kInputQIndex)]->GetShape();
    auto q_layout = input_infos[static_cast<size_t>(UnpadFaNpdInputIndex::kInputInputQLayoutIndex)]
                      ->GetScalarValueWithCheck<std::string>();
    auto kv_layout = input_infos[static_cast<size_t>(UnpadFaNpdInputIndex::kInputInputKVLayoutIndex)]
                       ->GetScalarValueWithCheck<std::string>();
    auto op_name = primitive->name();
    MS_CHECK_VALUE(
      q_layout == LAYOUT_BSH || q_layout == LAYOUT_TH,
      CheckAndConvertUtils::FormatCommMsg(op_name, " q_layout should be 'BSH' or 'TH', but got ", q_layout));

    MS_CHECK_VALUE(
      kv_layout == LAYOUT_BSH || kv_layout == LAYOUT_TH || kv_layout == LAYOUT_NPD,
      CheckAndConvertUtils::FormatCommMsg(op_name, " kv_layout should be 'BSH' or 'TH' or 'NPD', but got ", kv_layout));

    ShapeVector out_shape = UnpadFaNpdDoInferShape(q_shape);
    return {out_shape};
  }
  std::vector<TypeId> InferType(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const override {
    auto out_type = input_infos[static_cast<size_t>(UnpadFaNpdInputIndex::kInputQIndex)]->GetType();
    return {out_type};
  }
  std::set<int64_t> GetValueDependArgIndices() const override {
    return {static_cast<size_t>(UnpadFaNpdInputIndex::kInputActualQSeqIndex),
            static_cast<size_t>(UnpadFaNpdInputIndex::kInputActualKVSeqIndex)};
  }

  bool GeneralInferRegistered() const override { return true; }
};

class UnpadFaNpdAscend : public AclnnCustomKernelMod {
 public:
  UnpadFaNpdAscend() : AclnnCustomKernelMod(std::move("aclnnUnpadFaNpd")) {}
  ~UnpadFaNpdAscend() = default;

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs, void *stream_ptr) override {
    MS_EXCEPTION_IF_NULL(stream_ptr);
    RunOp(stream_ptr, workspace, inputs[static_cast<size_t>(UnpadFaNpdInputIndex::kInputQIndex)],
          inputs[static_cast<size_t>(UnpadFaNpdInputIndex::kInputKIndex)],
          inputs[static_cast<size_t>(UnpadFaNpdInputIndex::kInputVIndex)],
          inputs[static_cast<size_t>(UnpadFaNpdInputIndex::kInputAttnMaskIndex)],
          inputs[static_cast<size_t>(UnpadFaNpdInputIndex::kInputActualQSeqIndex)],
          inputs[static_cast<size_t>(UnpadFaNpdInputIndex::kInputActualKVSeqIndex)], head_num_, scale_, q_layout_,
          kv_layout_, block_size_, outputs[static_cast<size_t>(UnpadFaNpdOutputIndex::kOutputAttnOutIndex)]);
    return true;
  }

  void GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                        const std::vector<KernelTensor *> &outputs) override {
    head_num_ = inputs[static_cast<size_t>(UnpadFaNpdInputIndex::kInputHeadNumIndex)]->GetValueWithCheck<int64_t>();
    scale_ = static_cast<double>(
      inputs[static_cast<size_t>(UnpadFaNpdInputIndex::kInputScaleValueIndex)]->GetValueWithCheck<float>());
    q_layout_ =
      inputs[static_cast<size_t>(UnpadFaNpdInputIndex::kInputInputQLayoutIndex)]->GetValueWithCheck<std::string>();
    kv_layout_ =
      inputs[static_cast<size_t>(UnpadFaNpdInputIndex::kInputInputKVLayoutIndex)]->GetValueWithCheck<std::string>();
    block_size_ = inputs[static_cast<size_t>(UnpadFaNpdInputIndex::kInputBlockSizeIndex)]->GetValueWithCheck<int64_t>();

    auto q_cpu = CastVector<int32_t, int64_t>(inputs[static_cast<size_t>(UnpadFaNpdInputIndex::kInputActualQSeqIndex)]
                                                ->GetValueWithCheck<std::vector<int32_t>>());
    auto kv_cpu = CastVector<int32_t, int64_t>(inputs[static_cast<size_t>(UnpadFaNpdInputIndex::kInputActualKVSeqIndex)]
                                                 ->GetValueWithCheck<std::vector<int32_t>>());

    GetWorkspaceForResize(inputs[static_cast<size_t>(UnpadFaNpdInputIndex::kInputQIndex)],
                          inputs[static_cast<size_t>(UnpadFaNpdInputIndex::kInputKIndex)],
                          inputs[static_cast<size_t>(UnpadFaNpdInputIndex::kInputVIndex)],
                          inputs[static_cast<size_t>(UnpadFaNpdInputIndex::kInputAttnMaskIndex)], q_cpu, kv_cpu,
                          head_num_, scale_, q_layout_, kv_layout_, block_size_,
                          outputs[static_cast<size_t>(UnpadFaNpdOutputIndex::kOutputAttnOutIndex)]);
    return;
  }

 private:
  DEFINE_GET_WORKSPACE_FOR_RESIZE();
  std::string q_layout_{""};
  std::string kv_layout_{""};
  int64_t head_num_{0};
  int64_t block_size_{0};
  double scale_{1.0f};
};
}  // namespace ms_custom_ops

REG_GRAPH_MODE_OP(unpad_fa_npd, ms_custom_ops::UnpadFaNpdOpFuncImpl, ms_custom_ops::UnpadFaNpdAscend);

// =============================================================================
// PYBOOST MODE IMPLEMENTATION
// =============================================================================

namespace ms_custom_ops {
constexpr size_t kUnpadFaNpdOutputNum = 1;

ms::Tensor unpad_fa_npd_custom(const ms::Tensor &query, const ms::Tensor &key, const ms::Tensor &value,
                               const std::optional<ms::Tensor> &attn_mask, const ms::Tensor &q_seq,
                               const ms::Tensor &kv_seq, const int32_t &head_num, const float &scale,
                               const std::string &q_layout, const std::string &kv_layout, const int32_t &block_size) {
  std::string op_name = "UnpadFaNpd";
  auto q_cpu = GetSeqLenFromTensor<int32_t, int64_t>(q_seq.tensor()->cpu());
  if (q_cpu.empty()) {
    MS_LOG(EXCEPTION) << "Get q_cpu seq len failed ";
  }
  auto kv_cpu = GetSeqLenFromTensor<int32_t, int64_t>(kv_seq.tensor()->cpu());
  if (kv_cpu.empty()) {
    MS_LOG(EXCEPTION) << "Get kv_cpu seq len failed ";
  }
  auto runner = std::make_shared<ms::pynative::AclnnOpRunner>(op_name);
  UnpadFaNpdCheckInputsShape(op_name, query.shape(), key.shape(), value.shape(), q_seq.shape(), kv_seq.shape());
  UnpadFaNpdCheckInputsType(op_name, query.data_type(), key.data_type(), value.data_type(), q_seq.data_type(),
                            kv_seq.data_type());
  ShapeVector out_shape = UnpadFaNpdDoInferShape(query.shape());
  auto attn_out = ms::Tensor(query.data_type(), out_shape);

  runner->SetLaunchFunc(LAUNCH_ACLNN_FUNC(aclnnUnpadFaNpd, query, key, value, attn_mask, q_cpu, kv_cpu, head_num,
                                          static_cast<double>(scale), q_layout, kv_layout, block_size, attn_out));
  runner->Run({query, key, value, GetTensorOrEmpty(attn_mask), q_seq, kv_seq}, {attn_out});
  return attn_out;
}
}  // namespace ms_custom_ops

MS_CUSTOM_OPS_EXTENSION_MODULE(m) {
  m.def("unpad_fa_npd", &ms_custom_ops::unpad_fa_npd_custom, pybind11::arg("query"), pybind11::arg("key"),
        pybind11::arg("value"), pybind11::arg("attn_mask") = nullptr, pybind11::arg("actual_seq_qlen"),
        pybind11::arg("actual_seq_kvlen"), pybind11::arg("head_num") = 32, pybind11::arg("scale") = 1.0f,
        pybind11::arg("q_layout") = LAYOUT_TH, pybind11::arg("kv_layout") = LAYOUT_NPD,
        pybind11::arg("block_size") = 32);
}
