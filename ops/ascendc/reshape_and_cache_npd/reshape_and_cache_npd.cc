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
#include "ops/framework/aclnn/graphmode/aclnn_kernel_mod.h"
#include "ops/framework/utils.h"
#include "mindspore/include/custom_op_api.h"

namespace ms_custom_ops {

enum class CacheMode : int32_t {
  ND = 0,
  NPD = 1,
};

enum class ReshapeAndCacheNpdInputIndex : size_t {
  kInputKeyIndex = 0,
  kInputValueIndex = 1,
  kInputKeyCacheIndex = 2,
  kInputValueCacheIndex = 3,
  kInputSlotMappingIndex = 4,
  kInputQSeqIndex = 5,
  kInputKVSeqIndex = 6,
  kInputBlockTblIndex = 7,
  kInputCacheModeIndex = 8
};
enum class ReshapeAndCacheNpdOutputIndex : size_t { kOutputKeyIndex = 0, kOutputValueIndex = 1};

static void ReshapeAndCacheNpdCheckInputsShape(const std::string &op_name, const std::vector<int64_t> &key_shape,
                                                const std::vector<int64_t> &value_shape,
                                                const std::vector<int64_t> &key_cache_shape,
                                                const std::vector<int64_t> &value_cache_shape,
                                                const std::vector<int64_t> &slot_mapping_shape,
                                                const std::vector<int64_t> &q_seq_shape,
                                                const std::vector<int64_t> &kv_seq_shape,
                                                const std::vector<int64_t> &block_tbl_shape) {
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
                      << "q_seq_shape.dim=" << q_seq_shape.size()
                      << "kv_seq_shape.dim=" << kv_seq_shape.size();
  }
  if (block_tbl_shape.size() != kDim2) {
    MS_LOG(EXCEPTION) << op_name << ", the dim of block_tbl_shape is illegal, "
                      << "block_tbl_shape.dim=" << block_tbl_shape.size();
  }
  MS_CHECK_VALUE(key_shape == value_shape && key_cache_shape == value_cache_shape,
                 CheckAndConvertUtils::FormatCommMsg(
                 op_name, ", key_shape should be equal value_shape, key_cache_shape should be equal value_cache_shape,",
                 " but got, key.shape=", key_shape, ", value_shape.shape=", value_shape, ", key_cache_shape=",
                 key_cache_shape, ", value_cache_shape=", value_cache_shape));
}
static void ReshapeAndCacheNpdCheckInputsType(const std::string &op_name, const TypeId &key_dtype,
                                               const TypeId &value_dtype,
                                               const TypeId &key_cache_dtype,
                                               const TypeId &value_cache_dtype,
                                               const TypeId &slot_mapping_dtype,
                                               const TypeId &q_seq_dtype,
                                               const TypeId &kv_seq_dtype,
                                               const TypeId &block_tbl_dtype) {
  const std::unordered_set<TypeId> valid_types = {kNumberTypeFloat16, kNumberTypeBFloat16};
  std::unordered_set<TypeId> input_types = {key_dtype, value_dtype, key_cache_dtype, value_cache_dtype};
  if (input_types.size() > 1) {
    MS_LOG(EXCEPTION) << op_name << ", the dtype of 'key, value, key_cache, value_cache' should be same, but got '"
                      << TypeIdToString(key_dtype) << ", " << TypeIdToString(value_dtype) << ", "
                      << TypeIdToString(key_cache_dtype) << ", " << TypeIdToString(value_cache_dtype) << "'";
  }
  if (valid_types.find(key_dtype) == valid_types.end()) {
    MS_LOG(EXCEPTION) << op_name << ", the dtype of 'key, value, key_cache, value_cache' should be "
                      << TypeIdToString(kNumberTypeFloat16) << " or " << TypeIdToString(kNumberTypeFloat32)
                      << ", but got '" << TypeIdToString(key_dtype) << ", " << TypeIdToString(value_dtype) << ", "
                      << TypeIdToString(key_cache_dtype) << ", " << TypeIdToString(value_cache_dtype) << "'";
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
                      << TypeIdToString(kNumberTypeInt32) << " or " << TypeIdToString(kNumberTypeInt)
                      << ", but got '" << TypeIdToString(slot_mapping_dtype) << ", " << TypeIdToString(q_seq_dtype)
                      << ", " << TypeIdToString(kv_seq_dtype) << ", " << TypeIdToString(block_tbl_dtype) << "'";
  }
}
static mindspore::ShapeVector ReshapeAndCacheNpdDoInferShape(const mindspore::ShapeVector &shape,
                                                             const uint32_t &block_size, const uint32_t &batch) {
  int max_seq = shape.at(0) + (block_size - 1) * batch;
  ShapeVector out_shape = shape;
  out_shape.at(0) = max_seq;
  return out_shape;
}
class OPS_API ReshapeAndCacheNpdOpFuncImpl : public OpFuncImpl {
 public:
  ShapeArray InferShape(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const override {
    auto key_shape = input_infos[static_cast<size_t>(ReshapeAndCacheNpdInputIndex::kInputKeyIndex)]->GetShape();
    auto k_cache_shape =
           input_infos[static_cast<size_t>(ReshapeAndCacheNpdInputIndex::kInputKeyCacheIndex)]->GetShape();
    int32_t batch =
              input_infos[static_cast<size_t>(ReshapeAndCacheNpdInputIndex::kInputBlockTblIndex)]->GetShape().at(0);
    int32_t block_size = k_cache_shape.at(kDim2);
    ShapeVector out_shape = ReshapeAndCacheNpdDoInferShape(key_shape, block_size, batch);
    return {out_shape, out_shape};
  }
  std::vector<TypeId> InferType(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const override {
    auto out_type = input_infos[static_cast<size_t>(ReshapeAndCacheNpdInputIndex::kInputKeyIndex)]->GetType();
    return {out_type, out_type};
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
    RunOp(
      stream_ptr, workspace, inputs[static_cast<size_t>(ReshapeAndCacheNpdInputIndex::kInputKeyIndex)],
      inputs[static_cast<size_t>(ReshapeAndCacheNpdInputIndex::kInputValueIndex)],
      inputs[static_cast<size_t>(ReshapeAndCacheNpdInputIndex::kInputKeyCacheIndex)],
      inputs[static_cast<size_t>(ReshapeAndCacheNpdInputIndex::kInputValueCacheIndex)],
      inputs[static_cast<size_t>(ReshapeAndCacheNpdInputIndex::kInputSlotMappingIndex)],
      inputs[static_cast<size_t>(ReshapeAndCacheNpdInputIndex::kInputQSeqIndex)],
      inputs[static_cast<size_t>(ReshapeAndCacheNpdInputIndex::kInputKVSeqIndex)],
      inputs[static_cast<size_t>(ReshapeAndCacheNpdInputIndex::kInputBlockTblIndex)],
      cache_mode_,
      outputs[static_cast<size_t>(ReshapeAndCacheNpdOutputIndex::kOutputKeyIndex)],
      outputs[static_cast<size_t>(ReshapeAndCacheNpdOutputIndex::kOutputValueIndex)]);
    return true;
  }
  void GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                        const std::vector<KernelTensor *> &outputs) override {
    cache_mode_ = inputs[static_cast<size_t>(ReshapeAndCacheNpdInputIndex::kInputCacheModeIndex)]
                     ->GetValueWithCheck<int64_t>();
    GetWorkspaceForResize(inputs[static_cast<size_t>(ReshapeAndCacheNpdInputIndex::kInputKeyIndex)],
                          inputs[static_cast<size_t>(ReshapeAndCacheNpdInputIndex::kInputValueIndex)],
                          inputs[static_cast<size_t>(ReshapeAndCacheNpdInputIndex::kInputKeyCacheIndex)],
                          inputs[static_cast<size_t>(ReshapeAndCacheNpdInputIndex::kInputValueCacheIndex)],
                          inputs[static_cast<size_t>(ReshapeAndCacheNpdInputIndex::kInputSlotMappingIndex)],
                          inputs[static_cast<size_t>(ReshapeAndCacheNpdInputIndex::kInputQSeqIndex)],
                          inputs[static_cast<size_t>(ReshapeAndCacheNpdInputIndex::kInputKVSeqIndex)],
                          inputs[static_cast<size_t>(ReshapeAndCacheNpdInputIndex::kInputBlockTblIndex)],
                          cache_mode_,
                          outputs[static_cast<size_t>(ReshapeAndCacheNpdOutputIndex::kOutputKeyIndex)],
                          outputs[static_cast<size_t>(ReshapeAndCacheNpdOutputIndex::kOutputValueIndex)]);
    return;
  }

 private:
  DEFINE_GET_WORKSPACE_FOR_RESIZE();
  int64_t cache_mode_{1};
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
                                                     const int32_t &cache_mode) {
  std::string op_name = "ReshapeAndCacheNpd";
  auto runner = std::make_shared<ms::pynative::AclnnOpRunner>(op_name);
  ReshapeAndCacheNpdCheckInputsShape(op_name, key.shape(), value.shape(), key_cache.shape(), value_cache.shape(),
                                     slot_mapping.shape(), q_seq.shape(), kv_seq.shape(), block_tbl.shape());
  ReshapeAndCacheNpdCheckInputsType(op_name, key.data_type(), value.data_type(), key_cache.data_type(),
                                    value_cache.data_type(), slot_mapping.data_type(), q_seq.data_type(),
                                    kv_seq.data_type(), block_tbl.data_type());
  ShapeVector out_shape = ReshapeAndCacheNpdDoInferShape(key.shape(), key_cache.shape().at(kDim2),
                                                         block_tbl.shape().at(0));
  auto out_k = ms::Tensor(key.data_type(), out_shape);
  auto out_v = ms::Tensor(value.data_type(), out_shape);
  runner->SetLaunchFunc(LAUNCH_ACLNN_FUNC(aclnnReshapeAndCacheNpd, key, value, key_cache, value_cache, slot_mapping,
                                          q_seq, kv_seq, block_tbl, cache_mode, out_k, out_v));
  runner->Run({key, value, key_cache, value_cache, slot_mapping, q_seq, kv_seq, block_tbl}, {out_k, out_v});
  return {out_k, out_v};
}
}  // namespace ms_custom_ops

MS_CUSTOM_OPS_EXTENSION_MODULE(m) {
  m.def("reshape_and_cache_npd", &ms_custom_ops::reshape_and_cache_npd_custom, pybind11::arg("key"),
        pybind11::arg("value"), pybind11::arg("key_cache"), pybind11::arg("value_cache"), pybind11::arg("slot_mapping"),
        pybind11::arg("q_seq"), pybind11::arg("kv_seq"), pybind11::arg("block_tbl"), pybind11::arg("cache_mode") = 1);
}

