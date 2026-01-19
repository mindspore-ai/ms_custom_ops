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

#ifndef CCSRC_OPS_MS_KERNELS_INTERNAL_FLASH_ATTENTION_ENCODER_FLASH_ATTENTION_ENCODER_RUNNER_H_
#define CCSRC_OPS_MS_KERNELS_INTERNAL_FLASH_ATTENTION_ENCODER_FLASH_ATTENTION_ENCODER_RUNNER_H_

#include <vector>
#include <optional>
#include <cstdint>
#include <string>
#include <memory>

#include "ops/framework/ms_kernels_internal/pyboost/internal_pyboost_runner.h"
#include "ops/c_api/utils/attention_utils.h"
#include "ops/c_api/flash_attention_encoder/flash_attention_encoder.h"

namespace ms_custom_ops {
class FlashAttentionEncoderRunner : public InternalPyboostRunner {
 public:
  using InternalPyboostRunner::InternalPyboostRunner;
  void SetParam(int64_t head_num, float scale_value, int64_t kv_head_num, int64_t mask_type, int64_t kernel_type,
                int64_t window_size, int64_t cache_type, int64_t inner_precise,
                const std::vector<int32_t> &q_seq_len,
                const std::vector<int32_t> &kv_seq_len);
  void SetInputFormat(int64_t input_format) { param_.input_format = static_cast<int32_t>(input_format); }

 protected:
  bool UpdateParam() override;
  internal_v2::InternalOpPtr CreateKernel(const internal_v2::InputsImmutableInfoList &inputs,
                                          const internal_v2::OutputsImmutableInfoList &outputs) override;

 private:
  internal_v2::SelfAttentionParam param_{};
  bool created_flag_{false};
};
}  // namespace ms_custom_ops

#endif  // CCSRC_OPS_MS_KERNELS_INTERNAL_FLASH_ATTENTION_ENCODER_FLASH_ATTENTION_ENCODER_RUNNER_H_
