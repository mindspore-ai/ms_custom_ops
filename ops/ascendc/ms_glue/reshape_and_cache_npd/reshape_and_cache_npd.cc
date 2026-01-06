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
#include <vector>
#include <unordered_set>
#include <utility>
#include "ops/ascendc/aclnn_src/reshape_and_cache_npd/op_host/reshape_and_cache_npd_comm.h"
#include "ops/framework/aclnn/graphmode/aclnn_kernel_mod.h"
#include "ops/framework/utils.h"
#include "mindspore/include/custom_op_api.h"

namespace ms_custom_ops {

const char *op_name = "ReshapeAndCacheNpd";

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

static void ReshapeAndCacheNpdCheckInputsShape(
  const std::string &op_name, const std::vector<int64_t> &key_shape, const std::vector<int64_t> &value_shape,
  const std::vector<int64_t> &key_cache_shape, const std::vector<int64_t> &value_cache_shape,
  const std::vector<int64_t> &slot_mapping_shape, const std::vector<int64_t> &q_seq_shape,
  const std::vector<int64_t> &kv_seq_shape, const std::vector<int64_t> &block_tbl_shape) {
  if (key_cache_shape.size() != kDim4 || key_shape.size() != kDim2 || value_cache_shape.size() != kDim4 ||
      value_shape.size() != kDim2) {
    MS_LOG(EXCEPTION) << op_name << ", the dim of inputs should be value.dim=key.dim=2, "
                      << "key_cache_shape.dim=value_shape.dim=4, but got value.dim=" << value_shape.size()
                      << ", key.dim=" << key_shape.size();
  }
  if (slot_mapping_shape.size() != kDim1) {
    MS_LOG(EXCEPTION) << op_name << ", the dim of slot mapping is illegal, "
                      << "slot_mapping_shape.dim=" << slot_mapping_shape.size();
  }
  if (q_seq_shape.size() != kDim1 || kv_seq_shape.size() != kDim1) {
    MS_LOG(EXCEPTION) << op_name << ", the dim of inputs should be q_seq.dim=kv_seq.dim=1, but got q_seq.dim="
                      << "q_seq_shape.dim=" << q_seq_shape.size() << "kv_seq_shape.dim=" << kv_seq_shape.size();
  }
  if (block_tbl_shape.size() != kDim2) {
    MS_LOG(EXCEPTION) << op_name << ", the dim of block_tbl_shape is illegal, "
                      << "block_tbl_shape.dim=" << block_tbl_shape.size();
  }
  MS_CHECK_VALUE(
    key_shape == value_shape && key_cache_shape == value_cache_shape,
    CheckAndConvertUtils::FormatCommMsg(
      op_name, ", key_shape should be equal value_shape, key_cache_shape should be equal value_cache_shape,",
      " but got, key.shape=", key_shape, ", value_shape.shape=", value_shape, ", key_cache_shape=", key_cache_shape,
      ", value_cache_shape=", value_cache_shape));
}
static void ReshapeAndCacheNpdCheckInputsType(const std::string &op_name, const TypeId &key_dtype,
                                              const TypeId &value_dtype, const TypeId &key_cache_dtype,
                                              const TypeId &value_cache_dtype, const TypeId &slot_mapping_dtype,
                                              const TypeId &q_seq_dtype, const TypeId &kv_seq_dtype,
                                              const TypeId &block_tbl_dtype) {
  const std::unordered_set<TypeId> valid_types = {kNumberTypeFloat16, kNumberTypeBFloat16, kNumberTypeInt8};
  std::unordered_set<TypeId> input_types = {key_dtype, value_dtype, key_cache_dtype, value_cache_dtype};
  if (input_types.size() > 1) {
    MS_LOG(EXCEPTION) << op_name << ", the dtype of 'key, value, key_cache, value_cache' should be same, but got '"
                      << TypeIdToString(key_dtype) << ", " << TypeIdToString(value_dtype) << ", "
                      << TypeIdToString(key_cache_dtype) << ", " << TypeIdToString(value_cache_dtype) << "'";
  }
  if (valid_types.find(key_dtype) == valid_types.end()) {
    MS_LOG(EXCEPTION) << op_name << ", the dtype of 'key, value, key_cache, value_cache' should be "
                      << TypeIdToString(kNumberTypeFloat16) << " or " << TypeIdToString(kNumberTypeBFloat16) << " or "
                      << TypeIdToString(kNumberTypeInt8) << ", but got '" << TypeIdToString(key_dtype) << ", "
                      << TypeIdToString(value_dtype) << ", " << TypeIdToString(key_cache_dtype) << ", "
                      << TypeIdToString(value_cache_dtype) << "'";
  }
  const std::unordered_set<TypeId> valid_int_types = {kNumberTypeInt32, kNumberTypeInt};
  std::unordered_set<TypeId> input_int_types = {slot_mapping_dtype, q_seq_dtype, kv_seq_dtype, block_tbl_dtype};
  if (input_int_types.size() > 1) {
    MS_LOG(EXCEPTION) << op_name << ", the dtype of 'slot_mapping, q_seq, kv_seq, block_tbl' should be same, but got '"
                      << TypeIdToString(slot_mapping_dtype) << ", " << TypeIdToString(q_seq_dtype) << ", "
                      << TypeIdToString(kv_seq_dtype) << ", " << TypeIdToString(block_tbl_dtype) << "'";
  }
  if (valid_int_types.find(slot_mapping_dtype) == valid_int_types.end()) {
    MS_LOG(EXCEPTION) << op_name << ", the dtype of 'slot_mapping, q_seq, kv_seq, block_tbl' should be "
                      << TypeIdToString(kNumberTypeInt32) << " or " << TypeIdToString(kNumberTypeInt) << ", but got '"
                      << TypeIdToString(slot_mapping_dtype) << ", " << TypeIdToString(q_seq_dtype) << ", "
                      << TypeIdToString(kv_seq_dtype) << ", " << TypeIdToString(block_tbl_dtype) << "'";
  }
}
static mindspore::ShapeVector ReshapeAndCacheNpdDoInferShape(const mindspore::ShapeVector &shape,
                                                             const mindspore::ShapeVector &cache_shape,
                                                             const uint32_t &cache_key_value_layout,
                                                             const uint32_t &key_value_layout) {
  ShapeVector out_shape = shape;
  if (cache_shape.size() != kDim4) {
    MS_LOG(EXCEPTION) << "The dim cache should be 4 but got," << cache_shape.size();
  }
  auto block_num = cache_shape.at(0);
  auto block_size = cache_key_value_layout ? cache_shape.at(kDim2) : cache_shape.at(kDim1);
  auto max_seq = block_num * block_size;
  if (out_shape.at(0) >= 0) {
    out_shape.at(0) = max_seq;
  }
  return out_shape;
}

static void ReshapeAndCacheNpdCheckLayout(const std::string &op_name, const std::string &out_kv_cache_layout,
                                          const std::string &out_kv_layout) {
  if (out_kv_cache_layout != BSND_LAYOUT && out_kv_cache_layout != BNSD_LAYOUT) {
    MS_LOG(EXCEPTION) << op_name << ", out kv cache layout must be BSND or BNSD, but got" << out_kv_cache_layout;
  }
  if (out_kv_layout != TH_LAYOUT && out_kv_layout != NPD_LAYOUT) {
    MS_LOG(EXCEPTION) << op_name << ", out kv layout must be TH or NPD, but got" << out_kv_cache_layout;
  }
}
class OPS_API ReshapeAndCacheNpdOpFuncImpl : public OpFuncImpl {
 public:
  ShapeArray InferShape(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const override {
    auto key_shape = input_infos[static_cast<size_t>(ReshapeAndCacheNpdInputIndex::kInputKeyIndex)]->GetShape();
    auto k_cache_shape =
      input_infos[static_cast<size_t>(ReshapeAndCacheNpdInputIndex::kInputKeyCacheIndex)]->GetShape();

    auto cache_kv_layout = input_infos[static_cast<size_t>(ReshapeAndCacheNpdInputIndex::kInputKVCacheModeIndex)]
                             ->GetScalarValueWithCheck<std::string>() == std::string(BNSD_LAYOUT);

    auto kv_layout = input_infos[static_cast<size_t>(ReshapeAndCacheNpdInputIndex::kInputKeyValueModeIndex)]
                       ->GetScalarValueWithCheck<std::string>() == std::string(NPD_LAYOUT);

    ShapeVector out_shape = ReshapeAndCacheNpdDoInferShape(key_shape, k_cache_shape, cache_kv_layout, kv_layout);
    return {out_shape, out_shape};
  }
  std::vector<TypeId> InferType(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const override {
    auto out_type = input_infos[static_cast<size_t>(ReshapeAndCacheNpdInputIndex::kInputKeyIndex)]->GetType();
    return {out_type, out_type};
  }
  std::set<int64_t> GetValueDependArgIndices() const override {
    return {static_cast<size_t>(ReshapeAndCacheNpdInputIndex::kInputQSeqIndex),
            static_cast<size_t>(ReshapeAndCacheNpdInputIndex::kInputKVSeqIndex)};
  }

  bool GeneralInferRegistered() const override { return true; }
};

class ReshapeAndCacheNpdAscend : public AclnnCustomKernelMod {
 public:
  ReshapeAndCacheNpdAscend() : AclnnCustomKernelMod(std::move("aclnnReshapeAndCacheNpd")) {}
  ~ReshapeAndCacheNpdAscend() = default;

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs, void *stream_ptr) override {
    MS_EXCEPTION_IF_NULL(stream_ptr);
    RunOp(stream_ptr, workspace, inputs[static_cast<size_t>(ReshapeAndCacheNpdInputIndex::kInputKeyIndex)],
          inputs[static_cast<size_t>(ReshapeAndCacheNpdInputIndex::kInputValueIndex)],
          inputs[static_cast<size_t>(ReshapeAndCacheNpdInputIndex::kInputKeyCacheIndex)],
          inputs[static_cast<size_t>(ReshapeAndCacheNpdInputIndex::kInputValueCacheIndex)],
          inputs[static_cast<size_t>(ReshapeAndCacheNpdInputIndex::kInputSlotMappingIndex)],
          inputs[static_cast<size_t>(ReshapeAndCacheNpdInputIndex::kInputQSeqIndex)],
          inputs[static_cast<size_t>(ReshapeAndCacheNpdInputIndex::kInputKVSeqIndex)],
          inputs[static_cast<size_t>(ReshapeAndCacheNpdInputIndex::kInputBlockTblIndex)], kv_cache_mode_, kv_mode_,
          outputs[static_cast<size_t>(ReshapeAndCacheNpdOutputIndex::kOutputKeyIndex)],
          outputs[static_cast<size_t>(ReshapeAndCacheNpdOutputIndex::kOutputValueIndex)]);
    return true;
  }
  void GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                        const std::vector<KernelTensor *> &outputs) override {
    kv_cache_mode_ = inputs[static_cast<size_t>(ReshapeAndCacheNpdInputIndex::kInputKVCacheModeIndex)]
                       ->GetValueWithCheck<std::string>();
    kv_mode_ = inputs[static_cast<size_t>(ReshapeAndCacheNpdInputIndex::kInputKeyValueModeIndex)]
                 ->GetValueWithCheck<std::string>();

    ReshapeAndCacheNpdCheckLayout(op_name, kv_cache_mode_, kv_mode_);
    std::vector<int64_t> q_cpu =
      CastVector<int32_t, int64_t>(inputs[static_cast<size_t>(ReshapeAndCacheNpdInputIndex::kInputQSeqIndex)]
                                     ->GetValueWithCheck<std::vector<int32_t>>());
    std::vector<int64_t> kv_cpu =
      CastVector<int32_t, int64_t>(inputs[static_cast<size_t>(ReshapeAndCacheNpdInputIndex::kInputKVSeqIndex)]
                                     ->GetValueWithCheck<std::vector<int32_t>>());

    GetWorkspaceForResize(inputs[static_cast<size_t>(ReshapeAndCacheNpdInputIndex::kInputKeyIndex)],
                          inputs[static_cast<size_t>(ReshapeAndCacheNpdInputIndex::kInputValueIndex)],
                          inputs[static_cast<size_t>(ReshapeAndCacheNpdInputIndex::kInputKeyCacheIndex)],
                          inputs[static_cast<size_t>(ReshapeAndCacheNpdInputIndex::kInputValueCacheIndex)],
                          inputs[static_cast<size_t>(ReshapeAndCacheNpdInputIndex::kInputSlotMappingIndex)], q_cpu,
                          kv_cpu, inputs[static_cast<size_t>(ReshapeAndCacheNpdInputIndex::kInputBlockTblIndex)],
                          kv_cache_mode_, kv_mode_,
                          outputs[static_cast<size_t>(ReshapeAndCacheNpdOutputIndex::kOutputKeyIndex)],
                          outputs[static_cast<size_t>(ReshapeAndCacheNpdOutputIndex::kOutputValueIndex)]);
    return;
  }

 private:
  DEFINE_GET_WORKSPACE_FOR_RESIZE();
  std::string kv_cache_mode_ = BNSD_LAYOUT;
  std::string kv_mode_ = NPD_LAYOUT;
};
}  // namespace ms_custom_ops

REG_GRAPH_MODE_OP(reshape_and_cache_npd, ms_custom_ops::ReshapeAndCacheNpdOpFuncImpl,
                  ms_custom_ops::ReshapeAndCacheNpdAscend);

// =============================================================================
// PYBOOST MODE IMPLEMENTATION
// =============================================================================

namespace ms_custom_ops {
std::vector<ms::Tensor> reshape_and_cache_npd_custom(const ms::Tensor &key, const ms::Tensor &value,
                                                     const ms::Tensor &key_cache, const ms::Tensor &value_cache,
                                                     const ms::Tensor &slot_mapping, const ms::Tensor &q_seq,
                                                     const ms::Tensor &kv_seq, const ms::Tensor &block_tbl,
                                                     const std::string &kv_cache_layout,
                                                     const std::string &key_value_layout) {
  auto q_cpu = GetSeqLenFromTensor<int32_t, int64_t>(q_seq.tensor()->cpu());
  if (q_cpu.empty()) {
    MS_LOG(EXCEPTION) << "Get q_cpu seq len failed ";
  }
  auto kv_cpu = GetSeqLenFromTensor<int32_t, int64_t>(kv_seq.tensor()->cpu());
  if (kv_cpu.empty()) {
    MS_LOG(EXCEPTION) << "Get kv_cpu seq len failed ";
  }

  auto runner = std::make_shared<ms::pynative::AclnnOpRunner>(op_name);
  ReshapeAndCacheNpdCheckInputsShape(op_name, key.shape(), value.shape(), key_cache.shape(), value_cache.shape(),
                                     slot_mapping.shape(), q_seq.shape(), kv_seq.shape(), block_tbl.shape());
  ReshapeAndCacheNpdCheckInputsType(op_name, key.data_type(), value.data_type(), key_cache.data_type(),
                                    value_cache.data_type(), slot_mapping.data_type(), q_seq.data_type(),
                                    kv_seq.data_type(), block_tbl.data_type());
  ReshapeAndCacheNpdCheckLayout(op_name, kv_cache_layout, key_value_layout);

  auto cache_layout = kv_cache_layout == std::string(BNSD_LAYOUT);
  auto kv_layout = key_value_layout == std::string(NPD_LAYOUT);

  ShapeVector out_shape = ReshapeAndCacheNpdDoInferShape(key.shape(), key_cache.shape(), cache_layout, kv_layout);
  auto out_k = ms::Tensor(key.data_type(), out_shape);
  auto out_v = ms::Tensor(value.data_type(), out_shape);
  runner->SetLaunchFunc(LAUNCH_ACLNN_FUNC(aclnnReshapeAndCacheNpd, key, value, key_cache, value_cache, slot_mapping,
                                          q_cpu, kv_cpu, block_tbl, kv_cache_layout, key_value_layout, out_k, out_v));
  runner->Run({key, value, key_cache, value_cache, slot_mapping, q_seq, kv_seq, block_tbl}, {out_k, out_v});
  return {out_k, out_v};
}
}  // namespace ms_custom_ops

MS_CUSTOM_OPS_EXTENSION_MODULE(m) {
  m.def("reshape_and_cache_npd", &ms_custom_ops::reshape_and_cache_npd_custom, pybind11::arg("key"),
        pybind11::arg("value"), pybind11::arg("key_cache"), pybind11::arg("value_cache"), pybind11::arg("slot_mapping"),
        pybind11::arg("actual_seq_qlen"), pybind11::arg("actual_seq_kvlen"), pybind11::arg("block_tbl"),
        pybind11::arg("kv_cache_layout") = BNSD_LAYOUT, pybind11::arg("key_value_layout") = NPD_LAYOUT);
}
