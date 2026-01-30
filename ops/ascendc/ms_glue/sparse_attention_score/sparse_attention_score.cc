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
#include <cstring>

#include "ops/framework/aclnn/graphmode/aclnn_kernel_mod.h"
#include "ops/framework/utils.h"
#include "mindspore/include/custom_op_api.h"

namespace ms_custom_ops {

constexpr inline const char *LAYOUT_BNSD_STR = "BNSD";
constexpr inline const char *LAYOUT_BSND_STR = "BSND";

constexpr static size_t INPUT_LAYOUT_BUFFER_SIZE = 128;

enum class SparseAttentionScoreInputIndex : size_t {
  kQuery = 0,
  kKey,
  kValue,
  kRealShiftOptional,
  kDropMaskOptional,
  kPaddingMaskOptional,
  kAttenMaskOptional,
  kPrefixOptional,
  kSelectBlockIdxOptional,
  kScaleValueOptional,
  kKeepProbOptional,
  kPreTokensOptional,
  kNextTokensOptional,
  kHeadNum,
  kInputLayout,
  kInnerPreciseOptional,
  kSparseModeOptional,
  kSelectBlockLenOptional,
  kSoftmaxMaxOut,
  kSoftmaxSumOut,
  kSoftmaxOutOut,
  kAttentionOutOut,
  kInputsNumber,
};

class OPS_API SparseAttentionScoreOpFuncImpl : public OpFuncImpl {
 public:
  ShapeArray InferShape([[maybe_unused]] const PrimitivePtr &primitive,
                        [[maybe_unused]] const InferInfoPtrList &input_infos) const override {
    MS_EXCEPTION_IF_NULL(primitive);
    return kFakeOutTensorShapes;
  }

  std::vector<TypeId> InferType([[maybe_unused]] const PrimitivePtr &primitive,
                                [[maybe_unused]] const InferInfoPtrList &input_infos) const override {
    return kFakeOutTensorTypes;
  }

  bool GeneralInferRegistered() const override { return true; }
};

class SparseAttentionScoreAscend : public AclnnCustomKernelMod {
 public:
  SparseAttentionScoreAscend() : AclnnCustomKernelMod(std::move("aclnnSparseAttentionScoreMS")) {}
  ~SparseAttentionScoreAscend() = default;

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              [[maybe_unused]] const std::vector<KernelTensor *> &outputs, void *stream_ptr) override {
    MS_EXCEPTION_IF_NULL(stream_ptr);

    const std::string input_layout_str =
      inputs[static_cast<size_t>(SparseAttentionScoreInputIndex::kInputLayout)]->GetValueWithCheck<std::string>();
    char input_layout[INPUT_LAYOUT_BUFFER_SIZE] = {'\0'};
    std::strncpy(input_layout, input_layout_str.c_str(), sizeof input_layout / sizeof *input_layout - 1);

    RunOp(stream_ptr, workspace, inputs[static_cast<size_t>(SparseAttentionScoreInputIndex::kQuery)],
          inputs[static_cast<size_t>(SparseAttentionScoreInputIndex::kKey)],
          inputs[static_cast<size_t>(SparseAttentionScoreInputIndex::kValue)],
          nullptr,  // inputs[static_cast<size_t>(SparseAttentionScoreInputIndex::kRealShiftOptional)],
          nullptr,  // inputs[static_cast<size_t>(SparseAttentionScoreInputIndex::kDropMaskOptional)],
          nullptr,  // inputs[static_cast<size_t>(SparseAttentionScoreInputIndex::kPaddingMaskOptional)],
          inputs[static_cast<size_t>(SparseAttentionScoreInputIndex::kAttenMaskOptional)],
          nullptr,  // inputs[static_cast<size_t>(SparseAttentionScoreInputIndex::kPrefixOptional)],
          inputs[static_cast<size_t>(SparseAttentionScoreInputIndex::kSelectBlockIdxOptional)],
          inputs[static_cast<size_t>(SparseAttentionScoreInputIndex::kScaleValueOptional)],
          inputs[static_cast<size_t>(SparseAttentionScoreInputIndex::kKeepProbOptional)],
          inputs[static_cast<size_t>(SparseAttentionScoreInputIndex::kPreTokensOptional)],
          inputs[static_cast<size_t>(SparseAttentionScoreInputIndex::kNextTokensOptional)],
          inputs[static_cast<size_t>(SparseAttentionScoreInputIndex::kHeadNum)], input_layout,
          inputs[static_cast<size_t>(SparseAttentionScoreInputIndex::kInnerPreciseOptional)],
          inputs[static_cast<size_t>(SparseAttentionScoreInputIndex::kSparseModeOptional)],
          inputs[static_cast<size_t>(SparseAttentionScoreInputIndex::kSelectBlockLenOptional)],
          inputs[static_cast<size_t>(SparseAttentionScoreInputIndex::kSoftmaxMaxOut)],
          inputs[static_cast<size_t>(SparseAttentionScoreInputIndex::kSoftmaxSumOut)],
          nullptr,  // inputs[static_cast<size_t>(SparseAttentionScoreInputIndex::kSoftmaxOutOut)],
          inputs[static_cast<size_t>(SparseAttentionScoreInputIndex::kAttentionOutOut)]);
    return true;
  }

  void GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                        [[maybe_unused]] const std::vector<KernelTensor *> &outputs) override {
    const std::string input_layout_str =
      inputs[static_cast<size_t>(SparseAttentionScoreInputIndex::kInputLayout)]->GetValueWithCheck<std::string>();
    char input_layout[INPUT_LAYOUT_BUFFER_SIZE] = {'\0'};
    std::strncpy(input_layout, input_layout_str.c_str(), sizeof input_layout / sizeof *input_layout - 1);

    GetWorkspaceForResize(
      inputs[static_cast<size_t>(SparseAttentionScoreInputIndex::kQuery)],
      inputs[static_cast<size_t>(SparseAttentionScoreInputIndex::kKey)],
      inputs[static_cast<size_t>(SparseAttentionScoreInputIndex::kValue)],
      nullptr,  // inputs[static_cast<size_t>(SparseAttentionScoreInputIndex::kRealShiftOptional)],
      nullptr,  // inputs[static_cast<size_t>(SparseAttentionScoreInputIndex::kDropMaskOptional)],
      nullptr,  // inputs[static_cast<size_t>(SparseAttentionScoreInputIndex::kPaddingMaskOptional)],
      inputs[static_cast<size_t>(SparseAttentionScoreInputIndex::kAttenMaskOptional)],
      nullptr,  // inputs[static_cast<size_t>(SparseAttentionScoreInputIndex::kPrefixOptional)],
      inputs[static_cast<size_t>(SparseAttentionScoreInputIndex::kSelectBlockIdxOptional)],
      inputs[static_cast<size_t>(SparseAttentionScoreInputIndex::kScaleValueOptional)],
      inputs[static_cast<size_t>(SparseAttentionScoreInputIndex::kKeepProbOptional)],
      inputs[static_cast<size_t>(SparseAttentionScoreInputIndex::kPreTokensOptional)],
      inputs[static_cast<size_t>(SparseAttentionScoreInputIndex::kNextTokensOptional)],
      inputs[static_cast<size_t>(SparseAttentionScoreInputIndex::kHeadNum)], input_layout,
      inputs[static_cast<size_t>(SparseAttentionScoreInputIndex::kInnerPreciseOptional)],
      inputs[static_cast<size_t>(SparseAttentionScoreInputIndex::kSparseModeOptional)],
      inputs[static_cast<size_t>(SparseAttentionScoreInputIndex::kSoftmaxMaxOut)],
      inputs[static_cast<size_t>(SparseAttentionScoreInputIndex::kSoftmaxSumOut)],
      nullptr,  // inputs[static_cast<size_t>(SparseAttentionScoreInputIndex::kSoftmaxOutOut)],
      inputs[static_cast<size_t>(SparseAttentionScoreInputIndex::kAttentionOutOut)]);
    return;
  }

 private:
  DEFINE_GET_WORKSPACE_FOR_RESIZE();
};
}  // namespace ms_custom_ops

REG_GRAPH_MODE_OP(sparse_attention_score, ms_custom_ops::SparseAttentionScoreOpFuncImpl,
                  ms_custom_ops::SparseAttentionScoreAscend);

// =============================================================================
// PYBOOST MODE IMPLEMENTATION
// =============================================================================

namespace ms_custom_ops {
void npu_sparse_attention_score(
  const ms::Tensor &query, const ms::Tensor &key, const ms::Tensor &value,
  [[maybe_unused]] const ms::Tensor &real_shift_optional, [[maybe_unused]] const ms::Tensor &drop_mask_optional,
  [[maybe_unused]] const ms::Tensor &padding_mask_optional, const ms::Tensor &atten_mask_optional,
  [[maybe_unused]] const ms::Tensor &prefix_optional, const ms::Tensor &select_block_idx_optional,
  const double &scale_value_optional, const double keep_prob_optional, const std::int64_t pre_tokens_optional,
  const std::int64_t next_tokens_optional, const std::int64_t head_num, const std::string &input_layout,
  const std::int64_t inner_precise_optional, const std::int64_t sparse_mode_optional,
  const std::int64_t select_block_len_optional, const ms::Tensor &softmax_max_out, const ms::Tensor &softmax_sum_out,
  [[maybe_unused]] const ms::Tensor &softmax_out_out, const ms::Tensor &attention_out_out) {
  std::string op_name = "SparseAttentionScore";
  // 此处op_name是给人看的, 跟算子命名没有直接关联
  auto runner = std::make_shared<ms::pynative::AclnnOpRunner>(op_name);
  // 输入shape检查
  // ApplyRotaryPosEmbV3CheckInputsShape(op_name, query.shape(), key.shape(), cos.shape(), sin.shape());
  // 输入dtype检查
  // ApplyRotaryPosEmbV3CheckInputsType(op_name, query.data_type(), key.data_type(), cos.data_type(), sin.data_type());
  // 此处"aclnnSparseAttentionScoreMS", 是算字库函数表中名字前面加上aclnn
  // 可通过 nm -D ./build/xxx/xxx/ms_custom_ops.xxx.so | grep "SparseAttentionScoreMS"来确认
  // 如果是复写算子(inplace), 不必添加输出参数
  runner->SetLaunchFunc(LAUNCH_ACLNN_FUNC(aclnnSparseAttentionScoreMS, query, key, value,
                                          nullptr,  // real_shift_optional,
                                          nullptr,  // drop_mask_optional,
                                          nullptr,  // padding_mask_optional,
                                          atten_mask_optional,
                                          nullptr,  // prefix_optional,
                                          select_block_idx_optional, scale_value_optional, keep_prob_optional,
                                          pre_tokens_optional, next_tokens_optional, head_num, input_layout,
                                          inner_precise_optional, sparse_mode_optional, select_block_len_optional,
                                          softmax_max_out, softmax_sum_out,
                                          nullptr,  // softmax_out_out,
                                          attention_out_out));
  // 如果是复写算子(inplace), 输出参数为空
  runner->Run({query, key, value, real_shift_optional, drop_mask_optional, padding_mask_optional, atten_mask_optional,
               prefix_optional, select_block_idx_optional,
               // scale_value_optional,
               // keep_prob_optional,
               // pre_tokens_optional,
               // next_tokens_optional,
               // head_num,
               // input_layout,
               // inner_precise_optional,
               // sparse_mode_optional,
               softmax_max_out, softmax_sum_out, softmax_out_out, attention_out_out},
              {});
  // 无输出的算子返回值用void(不同于静态图)
  return;
}
}  // namespace ms_custom_ops

auto pyboost_sparse_attention_score(const ms::Tensor &query, const ms::Tensor &key, const ms::Tensor &value,
                                    const ms::Tensor &real_shift_optional, const ms::Tensor &drop_mask_optional,
                                    const ms::Tensor &padding_mask_optional, const ms::Tensor &atten_mask_optional,
                                    const ms::Tensor &prefix_optional, const ms::Tensor &select_block_idx_optional,
                                    const double scale_value, const double keep_prob, const std::int64_t pre_tokens,
                                    const std::int64_t next_tokens, const std::int64_t head_num,
                                    const std::string &input_layout, const std::int64_t inner_precise,
                                    const std::int64_t sparse_mode, const std::int64_t select_block_len_optional,
                                    const ms::Tensor &softmax_max_out, const ms::Tensor &softmax_sum_out,
                                    const ms::Tensor &softmax_out_out, const ms::Tensor &attention_out_out) {
  return ms::pynative::PyboostRunner::Call<ms_custom_ops::kNumber0>(
    ms_custom_ops::npu_sparse_attention_score, query, key, value, real_shift_optional, drop_mask_optional,
    padding_mask_optional, atten_mask_optional, prefix_optional, select_block_idx_optional, scale_value, keep_prob,
    pre_tokens, next_tokens, head_num, input_layout, inner_precise, sparse_mode, select_block_len_optional,
    softmax_max_out, softmax_sum_out, softmax_out_out, attention_out_out);
}

MS_CUSTOM_OPS_EXTENSION_MODULE(m) {
  m.def("sparse_attention_score", &pyboost_sparse_attention_score, "SparseAttentionScore",
        pybind11::arg("query"), pybind11::arg("key"), pybind11::arg("value"),
        pybind11::arg("real_shift_optional"),
        pybind11::arg("drop_mask_optional"), pybind11::arg("padding_mask_optional"),
        pybind11::arg("atten_mask_optional"), pybind11::arg("prefix_optional"),
        pybind11::arg("select_block_idx_optional"), pybind11::arg("scale_value"),
        pybind11::arg("keep_prob") = 1.0,
        pybind11::arg("pre_tokens") = 2147483647,
        pybind11::arg("next_tokens") = 2147483647,
        pybind11::arg("head_num") = 1,
        pybind11::arg("input_layout") = "BNSD",
        pybind11::arg("inner_precise")= 0,
        pybind11::arg("sparse_mode") = 2,
        pybind11::arg("select_block_len_optional") = 16,
        pybind11::arg("softmax_max_out") = py::none(),
        pybind11::arg("softmax_sum_out") = py::none(),
        pybind11::arg("softmax_out_out") = py::none(),
        pybind11::arg("attention_out_out") = py::none());
}
