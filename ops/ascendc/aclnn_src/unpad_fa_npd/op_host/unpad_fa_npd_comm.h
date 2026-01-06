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

#ifndef MS_CUSTOM_OPS_ASCENDC_UNPAD_FA_NPD_H
#define MS_CUSTOM_OPS_ASCENDC_UNPAD_FA_NPD_H

namespace {
static constexpr auto kDim1 = 1;
static constexpr auto kDim2 = 2;
static constexpr auto kDim3 = 3;
static constexpr auto kDim4 = 4;
static constexpr auto kDim9 = 9;
static constexpr auto kHigh32Bit = 32;
static constexpr int32_t kMaxEmbedSize = 256;

enum class InputLayout { BSH = 0, TH = 1, NPD = 2 };

enum class UnpadFaNpdInputIndex {
  kInputQIndex = 0,
  kInputKIndex = 1,
  kInputVIndex = 2,
  kInputAttnMaskIndex = 3,
  kInputActualQSeqIndex = 4,
  kInputActualKVSeqIndex = 5,
  kInputHeadNumIndex = 6,
  kInputScaleValueIndex = 7,
  kInputInputQLayoutIndex = 8,
  kInputInputKVLayoutIndex = 9,
  kInputBlockSizeIndex = 10
};

enum class UnpadFaNpdOutputIndex { kOutputAttnOutIndex = 0 };

#define GRAPH_INPUT_KV_SEQ_NAME "actual_seq_kvlen", "batch_valid_length"
#define GRAPH_INPUT_Q_SEQ_NAME "actual_seq_qlen", "q_seq_lens"
#define LAYOUT_BSH "BSH"
#define LAYOUT_TH "TH"
#define LAYOUT_NPD "NPD"
}  // namespace
#endif  // MS_CUSTOM_OPS_ASCENDC_UNPAD_FA_NPD_H
