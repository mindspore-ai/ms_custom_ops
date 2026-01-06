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

#include "ops/c_api/flash_attention_encoder/flash_attention_encoder_runner.h"
#include <memory>
#include <vector>
#include <optional>
#include <cstdint>
#include "ops/framework/utils.h"
namespace ms_custom_ops {
void FlashAttentionEncoderRunner::SetParam(int64_t head_num, float scale_value, int64_t kv_head_num, int64_t mask_type,
                                           int64_t kernel_type, int64_t window_size, int64_t cache_type,
                                           int64_t inner_precise, const std::vector<int32_t> &q_seq_len,
                                           const std::vector<int32_t> &kv_seq_len) {
  param_.head_num = static_cast<int32_t>(head_num);
  param_.qk_scale = static_cast<float>(scale_value);
  param_.kv_head_num = static_cast<int32_t>(kv_head_num);

  param_.calc_type = internal_v2::SelfAttentionParam::CalcType::PA_ENCODER;
  param_.mask_type = static_cast<internal_v2::SelfAttentionParam::MaskType>(mask_type);
  param_.kernel_type = static_cast<internal_v2::SelfAttentionParam::KernelType>(kernel_type);
  param_.window_size = static_cast<uint32_t>(window_size);
  param_.cache_type = static_cast<internal_v2::SelfAttentionParam::CacheType>(cache_type);
  param_.inner_precise = static_cast<uint32_t>(inner_precise);

  param_.q_seq_len = q_seq_len;
  param_.kv_seq_len = kv_seq_len;
  param_.input_format = 0;
}

bool FlashAttentionEncoderRunner::UpdateParam() {
  if (created_flag_) {
    created_flag_ = false;
    return true;
  }

  auto ret = internal_op_->UpdateParam(&param_);
  if (ret != internal_v2::kInternalOk) {
    MS_LOG(ERROR) << "Internal FlashAttentionEncoder UpdateParam failed.";
    return false;
  }
  return true;
}

internal_v2::InternalOpPtr FlashAttentionEncoderRunner::CreateKernel(
  const internal_v2::InputsImmutableInfoList &inputs, const internal_v2::OutputsImmutableInfoList &outputs) {
  created_flag_ = true;
  // NZ format routing in PyBoost mode when input_format == 1
  if (param_.input_format == 1) {
    auto inputs_clone = inputs;
    auto outputs_clone = outputs;
    inputs_clone[static_cast<size_t>(kQueryIdx)].SetFormat(internal_v2::kFormatFRACTAL_NZ);
    inputs_clone[static_cast<size_t>(kKeyIdx)].SetFormat(internal_v2::kFormatFRACTAL_NZ);
    inputs_clone[static_cast<size_t>(kValueIdx)].SetFormat(internal_v2::kFormatFRACTAL_NZ);
    inputs_clone[static_cast<size_t>(kMaskIdx)].SetFormat(internal_v2::kFormatFRACTAL_NZ);
    outputs_clone[static_cast<size_t>(kAttentionOutIdx)].SetFormat(internal_v2::kFormatFRACTAL_NZ);
    return internal_v2::CreateFlashAttentionEncoderOp(inputs_clone, outputs_clone, param_,
                                                      internal_v2::kInternalFlashAttentionEncoderOpName);
  }
  return internal_v2::CreateFlashAttentionEncoderOp(inputs, outputs, param_,
                                                    internal_v2::kInternalFlashAttentionEncoderOpName);
}

static std::vector<ms::Tensor> npu_flash_attention_encoder(
  const ms::Tensor &query, const ms::Tensor &key, const ms::Tensor &value, const std::optional<ms::Tensor> &layer_id,
  const std::optional<ms::Tensor> &mask, const std::optional<ms::Tensor> &alibi_coeff,
  const std::optional<ms::Tensor> &deq_scale_qk, const std::optional<ms::Tensor> &deq_offset_qk,
  const std::optional<ms::Tensor> &deq_scale_pv, const std::optional<ms::Tensor> &deq_offset_pv,
  const std::optional<ms::Tensor> &quant_p, const std::optional<ms::Tensor> &logN,
  const std::optional<ms::Tensor> &q_seq_len, const std::optional<ms::Tensor> &kv_seq_len, int64_t head_num,
  float scale_value, int64_t kv_head_num, int64_t mask_type, int64_t kernel_type, int64_t window_size,
  int64_t cache_type, int64_t inner_precise, int64_t input_format) {
  static auto op_name = "FlashAttentionEncoder";
  auto runner = std::make_shared<FlashAttentionEncoderRunner>(op_name);
  MS_EXCEPTION_IF_NULL(runner);

  // TH/TND 必须提供 q_seq_len/kv_seq_len
  // q_seq_len/kv_seq_len 必须为 int32
  if (!q_seq_len.has_value() || !kv_seq_len.has_value()) {
    MS_LOG(EXCEPTION) << "For " << op_name
                      << ", the q_seq_len and kv_seq_len can not be None, but got q_seq_len.has_value(): "
                      << q_seq_len.has_value() << ", kv_seq_len.has_value(): " << kv_seq_len.has_value();
  }
  auto q_seq = GetValueFromTensor<std::vector<int32_t>>(q_seq_len.value(), op_name, "q_seq_len");
  auto kv_seq = GetValueFromTensor<std::vector<int32_t>>(kv_seq_len.value(), op_name, "kv_seq_len");

  runner->SetParam(head_num, scale_value, kv_head_num, mask_type, kernel_type, window_size, cache_type, inner_precise,
                   q_seq, kv_seq);
  runner->SetInputFormat(input_format);

  // Setup the runner with all parameters to form cache key
  runner->Setup(op_name, query, key, value, layer_id, mask, alibi_coeff, deq_scale_qk, deq_offset_qk, deq_scale_pv,
                deq_offset_pv, quant_p, logN, q_seq_len, kv_seq_len, head_num, scale_value, kv_head_num, mask_type,
                kernel_type, window_size, cache_type, inner_precise, input_format);

  // outputs
  auto attn_out = ms::Tensor(query.data_type(), query.shape());
  std::vector<ms::Tensor> inputs = {query,
                                    key,
                                    value,
                                    GetTensorOrEmpty(layer_id),
                                    GetTensorOrEmpty(mask),
                                    GetTensorOrEmpty(alibi_coeff),
                                    GetTensorOrEmpty(deq_scale_qk),
                                    GetTensorOrEmpty(deq_offset_qk),
                                    GetTensorOrEmpty(deq_scale_pv),
                                    GetTensorOrEmpty(deq_offset_pv),
                                    GetTensorOrEmpty(quant_p),
                                    GetTensorOrEmpty(logN)};
  std::vector<ms::Tensor> outputs = {attn_out};
  runner->GetOrCreateKernel(inputs, outputs);
  runner->Run(inputs, outputs);
  return outputs;
}

static auto pyboost_flash_attention_encoder(
  const ms::Tensor &query, const ms::Tensor &key, const ms::Tensor &value, const std::optional<ms::Tensor> &layer_id,
  const std::optional<ms::Tensor> &mask, const std::optional<ms::Tensor> &alibi_coeff,
  const std::optional<ms::Tensor> &deq_scale_qk, const std::optional<ms::Tensor> &deq_offset_qk,
  const std::optional<ms::Tensor> &deq_scale_pv, const std::optional<ms::Tensor> &deq_offset_pv,
  const std::optional<ms::Tensor> &quant_p, const std::optional<ms::Tensor> &logN,
  const std::optional<ms::Tensor> &q_seq_len, const std::optional<ms::Tensor> &kv_seq_len, int64_t head_num,
  float scale_value, int64_t kv_head_num, int64_t mask_type, int64_t kernel_type, int64_t window_size,
  int64_t cache_type, int64_t inner_precise, int64_t input_format) {
  return ms::pynative::PyboostRunner::Call<1>(npu_flash_attention_encoder, query, key, value, layer_id, mask,
                                              alibi_coeff, deq_scale_qk, deq_offset_qk, deq_scale_pv, deq_offset_pv,
                                              quant_p, logN, q_seq_len, kv_seq_len, head_num, scale_value, kv_head_num,
                                              mask_type, kernel_type, window_size, cache_type, inner_precise,
                                              input_format);
}
}  // namespace ms_custom_ops

MS_CUSTOM_OPS_EXTENSION_MODULE(m) {
  m.def("flash_attention_encoder", &ms_custom_ops::pyboost_flash_attention_encoder, "Flash Attention Encoder",
        pybind11::arg("query"), pybind11::arg("key"), pybind11::arg("value"), pybind11::arg("layer_id") = std::nullopt,
        pybind11::arg("mask") = std::nullopt, pybind11::arg("alibi_coeff") = std::nullopt,
        pybind11::arg("deq_scale_qk") = std::nullopt, pybind11::arg("deq_offset_qk") = std::nullopt,
        pybind11::arg("deq_scale_pv") = std::nullopt, pybind11::arg("deq_offset_pv") = std::nullopt,
        pybind11::arg("quant_p") = std::nullopt, pybind11::arg("logN") = std::nullopt,
        pybind11::arg("q_seq_len") = std::nullopt, pybind11::arg("kv_seq_len") = std::nullopt,
        pybind11::arg("head_num") = 0, pybind11::arg("scale_value") = 1.0, pybind11::arg("kv_head_num") = 0,
        pybind11::arg("mask_type") = 0, pybind11::arg("kernel_type") = 0, pybind11::arg("window_size") = 0,
        pybind11::arg("cache_type") = 0, pybind11::arg("inner_precise") = 0, pybind11::arg("input_format") = 0);
}
