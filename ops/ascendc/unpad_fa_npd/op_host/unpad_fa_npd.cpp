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

#include "unpad_fa_npd.h"
#include <algorithm>
#include <sstream>
#include <string>
#include <vector>
#include "unpad_fa_npd_tiling.h"
#include "register/op_def_registry.h"
#include "graph/utils/type_utils.h"
#include "tiling/platform/platform_ascendc.h"
#include "utils/log/asc_cpu_log.h"

namespace optiling {
static constexpr auto kDim1 = 1;
static constexpr auto kDim2 = 2;
static constexpr auto kDim3 = 3;
static constexpr auto kDim4 = 4;
static constexpr auto kDim9 = 9;
static constexpr auto kHigh32Bit = 32;

class tiling_info {
 public:
  int32_t batch;
  int32_t head_num = 0;
  int32_t d = 0;
  int32_t kv_head_num = 0;
  int32_t head_stride = 0;
  std::vector<int64_t> q_seq_len;
  std::vector<int64_t> kv_seq_len;
  int32_t max_q_seq = 0;
  int32_t max_kv_seq = 0;
  bool is_triu = false;
  bool is_long_seq = false;
  int32_t mask_type = 0;
  int32_t mask_stride = 0;
  int32_t mask_seq_len = 0;
  int32_t splitm = 0;
  // Friend function to overload the << operator
  friend std::ostream &operator<<(std::ostream &os, const tiling_info &obj) {
    os << "\nbatch " << obj.batch << "\n"
       << "head_num " << obj.head_num << "\n"
       << "d " << obj.d << "\n"
       << "kv_head_num " << obj.kv_head_num << "\n"
       << "head_stride " << obj.head_stride << "\n";
    os << "q_seq_len ";
    for (auto &s : obj.q_seq_len) os << " " << s;
    os << "\n";
    os << "kv_seq_len ";
    for (auto &s : obj.kv_seq_len) os << " " << s;
    os << "\n";
    os << "max_q_seq " << obj.max_q_seq << "\n"
       << "is_triu " << obj.is_triu << "\n"
       << "is_long_seq " << obj.is_long_seq << "\n"
       << "mask_type " << obj.mask_type << "\n"
       << "mask_stride " << obj.mask_stride << "\n"
       << "mask_seq_len " << obj.mask_seq_len << "\n"
       << "splitm " << obj.splitm;
    return os;
  }
  void print_to_log() {
    std::ostringstream oss;
    oss << this;
    ASC_CPU_LOG_INFO("tiling info: %s\n", oss.str().c_str());
  }
};

constexpr int32_t triu_dim = 128;
constexpr int32_t block_size = 16;
constexpr int32_t double_ping_pong_size = 32768 * 8;
constexpr int32_t splitm_double_ping_pong_size = 32768 * 16;

#define FA_ASSERT_RET(context, condition, logMessage, ret)                                         \
  do {                                                                                             \
    if (!(condition)) {                                                                            \
      const char *name = ((context)->GetNodeName() == nullptr) ? "nil" : (context)->GetNodeName(); \
      ASC_CPU_LOG_ERROR("%s: %s", name, logMessage);                                               \
      return ret;                                                                                  \
    }                                                                                              \
  } while (0)

inline static int32_t UpDiv(int32_t val, int32_t divider = block_size) { return (val + divider - 1) / divider; }

inline static int32_t UpRound(int32_t val, int32_t round = block_size) { return UpDiv(val, round) * round; }

inline static uint32_t FloatToBits(float x) {
  uint32_t u;
  std::memcpy(&u, &x, sizeof(u));
  return u;
}

static ge::graphStatus GetD(gert::TilingContext *context, int32_t head_num, int32_t *d) {
  auto input_idx = static_cast<size_t>(UnpadFaNpdInputIndex::kInputQIndex);
  auto q_shape = context->GetInputShape(input_idx);
  FA_ASSERT_RET(context, q_shape != nullptr, "q sahpe is null", ge::GRAPH_FAILED);
  auto q_dims_num = q_shape->GetOriginShape().GetDimNum();
  if (q_dims_num == kDim2) {  // TH
    *d = static_cast<int32_t>(q_shape->GetOriginShape().GetDim(kDim1)) / head_num;
  } else if (q_dims_num == kDim3) {  // BSH
    *d = static_cast<int32_t>(q_shape->GetOriginShape().GetDim(kDim2)) / head_num;
  } else {
    FA_ASSERT_RET(context, 0, "query format not supported", ge::GRAPH_FAILED);
  }
  return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetKvHeadNum(gert::TilingContext *context, int32_t d, int32_t *kv_head_num) {
  auto input_idx = static_cast<size_t>(UnpadFaNpdInputIndex::kInputKIndex);
  auto k_shape = context->GetInputShape(input_idx);
  FA_ASSERT_RET(context, k_shape != nullptr, "k sahpe is null", ge::GRAPH_FAILED);
  auto k_dims_num = k_shape->GetOriginShape().GetDimNum();
  if (k_dims_num == kDim2) {  // TH
    *kv_head_num = static_cast<int32_t>(k_shape->GetOriginShape().GetDim(kDim1)) / d;
  } else if (k_dims_num == kDim3) {  // BSH
    *kv_head_num = static_cast<int32_t>(k_shape->GetOriginShape().GetDim(kDim2)) / d;
  } else {
    FA_ASSERT_RET(context, 0, "kv format not supported", ge::GRAPH_FAILED);
  }
  return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetBatch(gert::TilingContext *context, int32_t *b) {
  auto input_idx = static_cast<size_t>(UnpadFaNpdInputIndex::kInputActualQSeqIndex);
  auto q_seq_len = context->GetInputShape(input_idx);
  FA_ASSERT_RET(context, q_seq_len != nullptr, "q sequence length sahpe is null", ge::GRAPH_FAILED);
  auto q_seq_dims_num = q_seq_len->GetOriginShape().GetDimNum();
  if (q_seq_dims_num == 1) {
    *b = q_seq_len->GetOriginShape().GetDim(0);
  } else if (q_seq_dims_num == kDim2) {
    *b = q_seq_len->GetOriginShape().GetDim(kDim1);
  } else {
    FA_ASSERT_RET(context, 0, "query sequence length format not supported", ge::GRAPH_FAILED);
  }
  return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetQSeqLens(gert::TilingContext *context, int batch, std::vector<int64_t> *q_seq_len) {
  auto input_idx = static_cast<size_t>(UnpadFaNpdInputIndex::kInputActualQSeqIndex);
  auto t = context->GetInputTensor(input_idx);
  FA_ASSERT_RET(context, t != nullptr, "q sequence length is null", ge::GRAPH_FAILED);
  FA_ASSERT_RET(context, t->GetShapeSize() == batch, "q sequence length size is not valid", ge::GRAPH_FAILED);
  auto p = t->GetData<int64_t>();

  q_seq_len->assign(p, p + batch);
  return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetKvSeqLens(gert::TilingContext *context, int batch, std::vector<int64_t> *kv_seq_len) {
  auto input_idx = static_cast<size_t>(UnpadFaNpdInputIndex::kInputActualKVSeqIndex);
  auto t = context->GetInputTensor(input_idx);
  FA_ASSERT_RET(context, t != nullptr, "kv sequence length is null", ge::GRAPH_FAILED);
  FA_ASSERT_RET(context, t->GetShapeSize() == batch, "kv sequence length size is not valid", ge::GRAPH_FAILED);
  auto p = t->GetData<int64_t>();
  kv_seq_len->assign(p, p + batch);
  return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetTriuMask(gert::TilingContext *context, bool *is_triu) {
  auto input_idx = static_cast<size_t>(UnpadFaNpdInputIndex::kInputAttnMaskIndex);
  *is_triu = false;
  if (context->GetOptionalInputTensor(input_idx) == nullptr) {
    return ge::GRAPH_SUCCESS;
  }
  auto mask_shape = context->GetInputShape(input_idx);
  if ((mask_shape && mask_shape->GetOriginShape().GetDimNum() == kDim2) &&
      (mask_shape->GetOriginShape().GetDim(0) == triu_dim) && (mask_shape->GetOriginShape().GetDim(1) == triu_dim)) {
    *is_triu = true;
  }
  return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetLongMask(gert::TilingContext *context, bool is_triu, bool *is_long_seq) {
  auto input_idx = static_cast<size_t>(UnpadFaNpdInputIndex::kInputAttnMaskIndex);
  *is_long_seq = false;
  if (context->GetOptionalInputTensor(input_idx) == nullptr) {
    return ge::GRAPH_SUCCESS;
  }
  auto mask_shape = context->GetInputShape(input_idx);
  if (mask_shape) {
    auto dim_num = mask_shape->GetOriginShape().GetDimNum();
    auto max_seq = mask_shape->GetOriginShape().GetDim(dim_num - 1);
    if ((max_seq == triu_dim) && is_triu) {
      *is_long_seq = true;
    }
  }
  return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetMaskType(gert::TilingContext *context, int32_t *mask_type) {
  auto input_idx = static_cast<size_t>(UnpadFaNpdInputIndex::kInputAttnMaskIndex);
  *mask_type = 0;
  if (context->GetOptionalInputTensor(input_idx) == nullptr) {
    return ge::GRAPH_SUCCESS;
  }
  auto mask_shape = context->GetInputShape(input_idx);

  if (mask_shape) {
    *mask_type = 1;
  }
  return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetMaskStride(gert::TilingContext *context, int32_t *mask_stride) {
  auto input_idx = static_cast<size_t>(UnpadFaNpdInputIndex::kInputAttnMaskIndex);
  *mask_stride = 0;
  if (context->GetOptionalInputTensor(input_idx) == nullptr) {
    return ge::GRAPH_SUCCESS;
  }
  auto mask_shape = context->GetInputShape(input_idx);

  if (mask_shape) {
    auto dim_num = mask_shape->GetOriginShape().GetDimNum();
    if (dim_num == kDim2) {
      *mask_stride = 0;
    } else if (dim_num == kDim3) {
      *mask_stride = mask_shape->GetOriginShape().GetDim(kDim1);
    } else {
      FA_ASSERT_RET(context, 0, "mask format not supported", ge::GRAPH_FAILED);
    }
  }
  return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetMaskSeq(gert::TilingContext *context, int32_t *mask_seq_len) {
  auto input_idx = static_cast<size_t>(UnpadFaNpdInputIndex::kInputAttnMaskIndex);
  *mask_seq_len = 0;
  if (context->GetOptionalInputTensor(input_idx) == nullptr) {
    return ge::GRAPH_SUCCESS;
  }
  auto mask_shape = context->GetInputShape(input_idx);

  if (mask_shape) {
    auto dim_num = mask_shape->GetOriginShape().GetDimNum();
    *mask_seq_len = mask_shape->GetOriginShape().GetDim(dim_num - 1);
  }
  return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetTilingKey(gert::TilingContext *context, tiling_info *ti, int32_t *tiling_key) {
  auto input_idx = static_cast<size_t>(UnpadFaNpdInputIndex::kInputQIndex);
  auto t = context->GetInputTensor(input_idx);
  FA_ASSERT_RET(context, t != nullptr, "q is null", ge::GRAPH_FAILED);
  *tiling_key = 0;
  bool is_bf_16 = t->GetDataType() == ge::DataType::DT_BF16;
  *tiling_key = is_bf_16 ? 1 : 0;
  if (is_bf_16 && ti->d <= 128 && ti->mask_type == 0) {
    *tiling_key |= (1 << 1);
  }
  return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InitUnpadFaNpdTilingInfo(gert::TilingContext *context, tiling_info &ti, float &tor, bool &npd,
                                                int64_t &page_size) {
  ti.head_num = *context->GetAttrs()->GetAttrPointer<int64_t>(0);

  FA_ASSERT_RET(context, (GetD(context, ti.head_num, &ti.d) == ge::GRAPH_SUCCESS), "fail to get embed size",
                ge::GRAPH_FAILED);
  FA_ASSERT_RET(context, (GetKvHeadNum(context, ti.d, &ti.kv_head_num) == ge::GRAPH_SUCCESS), "fail to get embed size",
                ge::GRAPH_FAILED);
  FA_ASSERT_RET(context, (GetBatch(context, &ti.batch) == ge::GRAPH_SUCCESS), "fail to get batch size",
                ge::GRAPH_FAILED);
  FA_ASSERT_RET(context, (GetQSeqLens(context, ti.batch, &ti.q_seq_len) == ge::GRAPH_SUCCESS),
                "fail to get query sequnch length", ge::GRAPH_FAILED);
  FA_ASSERT_RET(context, (GetKvSeqLens(context, ti.batch, &ti.kv_seq_len) == ge::GRAPH_SUCCESS),
                "fail to key&value sequnch length", ge::GRAPH_FAILED);
  ti.head_stride = 0;
  ti.max_q_seq = *std::max_element(ti.q_seq_len.begin(), ti.q_seq_len.end());
  ti.max_kv_seq = *std::max_element(ti.kv_seq_len.begin(), ti.kv_seq_len.end());
  FA_ASSERT_RET(context, (GetTriuMask(context, &ti.is_triu) == ge::GRAPH_SUCCESS), "fail to setup is triu mask",
                ge::GRAPH_FAILED);
  FA_ASSERT_RET(context, (GetLongMask(context, ti.is_triu, &ti.is_long_seq) == ge::GRAPH_SUCCESS),
                "fail to setup is long mask", ge::GRAPH_FAILED);
  FA_ASSERT_RET(context, (GetMaskStride(context, &ti.mask_stride) == ge::GRAPH_SUCCESS), "fail to setup mask stride",
                ge::GRAPH_FAILED);
  FA_ASSERT_RET(context, (GetMaskType(context, &ti.mask_type) == ge::GRAPH_SUCCESS), "fail to setup mask type",
                ge::GRAPH_FAILED);
  FA_ASSERT_RET(context, (GetMaskSeq(context, &ti.mask_seq_len) == ge::GRAPH_SUCCESS), "fail to setup mask type",
                ge::GRAPH_FAILED);
  return ge::GRAPH_SUCCESS;
}

static int32_t ComputeBatchTilingParams(gert::TilingContext *context, tiling_info &ti, UnpadFaNpdTilingData &tiling,
                                        bool npd, int64_t page_size, int d_round, int &q_blocks_total,
                                        uint64_t &in_out_offset, uint64_t &kv_offset) {
  constexpr int pp_buff_size = 128 * 128;
  constexpr int pp_mm[] = {16, 32, 48, 64, 80, 96, 112, 128};
  constexpr int pp_mm_len = sizeof(pp_mm) / sizeof(pp_mm[0]);
  constexpr int pp_mm_max = pp_mm[pp_mm_len - 1];
  constexpr int pp_nn[] = {16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 256};
  constexpr int pp_nn_len = sizeof(pp_nn) / sizeof(pp_nn[0]);
  constexpr int pp_nn_max = pp_nn[pp_nn_len - 1];

  for (int i = 0; i < ti.batch; i++) {
    auto tiling_param = &(tiling.get_buf()[i * KParaLastIdx + KHeadLastIdx]);
    auto cur_q_seq = ti.q_seq_len.at(i);
    auto cur_q_seq_round = UpRound(cur_q_seq);
    auto cur_kv_seq = ti.kv_seq_len.at(i);
    auto cur_kv_seq_round = UpRound(cur_kv_seq);
    auto tiling_k = d_round < triu_dim ? triu_dim : d_round;

    uint32_t nUbd = std::min((pp_buff_size / tiling_k / block_size) * block_size, cur_kv_seq_round);
    uint32_t nIbd = (nUbd > pp_nn_max) ? pp_nn_len - 1 : (nUbd / block_size - 1);
    uint32_t mUbd =
      std::min((pp_buff_size / std::max(d_round, pp_nn[nIbd]) / block_size) * block_size, cur_q_seq_round);
    uint32_t mIbd = (mUbd > pp_mm_max) ? pp_mm_len - 1 : (mUbd / block_size - 1);
    mUbd = (ti.splitm) ? pp_nn_max : pp_mm[mIbd];

    int32_t q_blocks_num = (cur_q_seq != 0 && cur_kv_seq != 0) ? (UpDiv(cur_q_seq, mUbd)) : 0;
    q_blocks_total += q_blocks_num;

    // sets M,N and K
    tiling_param[KQSeqLenIdx] = cur_q_seq;
    tiling_param[KKvSeqLenIdx] = cur_kv_seq;
    tiling_param[KMMIdx] = mUbd;
    tiling_param[KNNIdx] = pp_nn[nIbd];
    // sets IO offsets
    tiling_param[KAddrQSeqOffsetHighIdx] = (in_out_offset >> kHigh32Bit);
    tiling_param[KAddrQSeqOffsetLowIdx] = (static_cast<uint32_t>(in_out_offset));
    tiling_param[KAddrKSeqOffsetHighIdx] = (kv_offset >> kHigh32Bit);
    tiling_param[KAddrKSeqOffsetLowIdx] = (static_cast<uint32_t>(kv_offset));
    tiling_param[KAddrVSeqOffsetHighIdx] = (kv_offset >> kHigh32Bit);
    tiling_param[KAddrVSeqOffsetLowIdx] = (static_cast<uint32_t>(kv_offset));
    tiling_param[KAddrOSeqOffsetHighIdx] = (in_out_offset >> kHigh32Bit);
    tiling_param[KAddrOSeqOffsetLowIdx] = (static_cast<uint32_t>(in_out_offset));
    tiling_param[KTotalQBlockIdx] = q_blocks_total;
    tiling_param[KSplitIdx] = ti.splitm;
    tiling_param[KStateIdx] = (static_cast<int32_t>(cur_q_seq) > 0 && static_cast<int32_t>(cur_kv_seq) > 0);
    in_out_offset += static_cast<uint64_t>(cur_q_seq * ti.head_num * ti.d);
    if (npd) {
      kv_offset += static_cast<uint64_t>(UpRound(cur_kv_seq, page_size) * ti.kv_head_num * ti.d);
    } else {
      kv_offset += static_cast<uint64_t>(cur_kv_seq * ti.kv_head_num * ti.d);
    }
  }
  return q_blocks_total;
}

static ge::graphStatus UnpadFaNpdTiling(gert::TilingContext *context) {
  tiling_info ti;
  UnpadFaNpdTilingData tiling;

  auto tor = static_cast<float>(*context->GetAttrs()->GetAttrPointer<float>(1));
  bool npd = std::string(context->GetAttrs()->GetAttrPointer<char>(3)) == std::string("NPD");
  auto page_size = *context->GetAttrs()->GetAttrPointer<int64_t>(4);
  auto status = InitUnpadFaNpdTilingInfo(context, ti, tor, npd, page_size);
  if (status != ge::GRAPH_SUCCESS) {
    return status;
  }

  auto d_round = UpRound(ti.d);
  int32_t q_blocks_total = 0;
  uint64_t in_out_offset = 0;
  uint64_t kv_offset = 0;
  int tiling_key = 0;
  ti.splitm = false;

  auto tiling_size = (KHeadLastIdx + KParaLastIdx * ti.batch) * sizeof(uint32_t);
  FA_ASSERT_RET(context, (KMaxBatch >= ti.batch), "tiling size is too big", ge::GRAPH_FAILED);
  ti.print_to_log();

  // Call helper
  q_blocks_total =
    ComputeBatchTilingParams(context, ti, tiling, npd, page_size, d_round, q_blocks_total, in_out_offset, kv_offset);

  auto require_cores = static_cast<uint32_t>(ti.head_num * q_blocks_total);

  auto tiling_head = tiling.get_buf();
  tiling_head[KBatchIdx] = ti.batch;
  tiling_head[KMaxSeqIdx] = ti.mask_seq_len;
  tiling_head[KInnerBatchSizeIdx] = ti.head_num;
  tiling_head[KEmbeddingSizeIdx] = ti.d;
  tiling_head[KBvHeadNumIdx] = ti.kv_head_num;
  tiling_head[KTorIdx] = FloatToBits(tor);
  tiling_head[KHeadStrideIdx] = ti.head_stride;
  tiling_head[KMaskStrideIdx] = ti.mask_stride;
  tiling_head[KIsTriuMaskIdx] = (static_cast<uint32_t>(ti.is_triu));
  tiling_head[KTotalQBlkNumIdx] = q_blocks_total;
  tiling_head[KIsClampIdx] = 0;
  float clamp = 0.0f;
  auto clampInt = FloatToBits(clamp);
  tiling_head[KClampMinUptrIdx] = clampInt;
  tiling_head[KClampMaxUptrIdx] = clampInt;
  tiling_head[KNoneIdx] = 0;
  tiling_head[KTilingHeadSizeIdx] = KHeadLastIdx;
  tiling_head[KTilingParaSizeIdx] = KParaLastIdx;
  FA_ASSERT_RET(context, (GetTilingKey(context, &ti, &tiling_key) == ge::GRAPH_SUCCESS), "fail to compute tiling key",
                ge::GRAPH_FAILED);
  tiling_head[KKeyIdx] = tiling_key;  // TBD
  tiling_head[KIsLongSeqIdx] = static_cast<uint32_t>(ti.is_long_seq);
  tiling_head[KMaxKVSeqIdx] = ti.max_kv_seq;
  tiling_head[KIsAlibiMaskIdx] = 0;
  tiling_head[KMaskTypeIdx] = static_cast<uint32_t>(ti.mask_type);
  tiling_head[KAlibiCompresOffsetIdx] = 0;
  tiling_head[KAlibiLeftAlignIdx] = 0;
  tiling_head[KEmbeddingSizeVIdx] = ti.d;
  tiling_head[KQuantTypeIdx] = 0;
  tiling_head[KDataShapeTypeIdx] = 0;
  tiling_head[KScaleTypeIdx] = 0;
  tiling_head[KWindowSizeIdx] = 0;
  tiling_head[KMaxNumBlocksIdx] = 0;
  tiling_head[KQMaxSeqLenIdx] = ti.max_q_seq;
  tiling_head[KPreTokensIdx] = 0;
  tiling_head[KNextTokensIdx] = 0;
  tiling_head[KRazorLenIdx] = 0;
  tiling_head[KTileQIdx] = 0;
  tiling_head[KTileKvIdx] = 0;
  tiling_head[KTextQLenIdx] = 0;
  tiling_head[KTextKvLenIdx] = 0;
  tiling_head[KKvLayoutIdx] = npd ? kDim2 : 0;
  tiling_head[KPageSizeIdx] = page_size;

  auto ascendc_platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
  auto core_num = ascendc_platform.GetCoreNumAic();
  core_num = (core_num > require_cores) ? require_cores : core_num;
  uint64_t work_size = static_cast<uint64_t>(core_num) * static_cast<uint64_t>(double_ping_pong_size) * sizeof(float);
  if (ti.splitm) {
    work_size = static_cast<uint64_t>(core_num) * static_cast<uint64_t>(splitm_double_ping_pong_size) * sizeof(float);
  }
  tiling_head[KWsLowIdx] = (static_cast<uint32_t>(work_size));

  size_t *currentWorkspace = context->GetWorkspaceSizes(kDim1);
  currentWorkspace[0] = work_size * kDim9;

  context->SetBlockDim(core_num);
  tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
  context->GetRawTilingData()->SetDataSize(tiling_size);
  context->SetTilingKey(tiling_key);

  return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge {
static graphStatus UnpadFaNpdInferShape(gert::InferShapeContext *context) {
  auto q_shape = context->GetInputShape(0);
  gert::Shape *attn_out_shape = context->GetOutputShape(0);
  *attn_out_shape = *q_shape;
  return ge::GRAPH_SUCCESS;
}

static graphStatus UnpadFaNpdInferDataType(gert::InferDataTypeContext *context) {
  const auto inputDataType = context->GetInputDataType(0);
  context->SetOutputDataType(0, inputDataType);
  return ge::GRAPH_SUCCESS;
}
}  // namespace ge

namespace ops {
class UnpadFaNpd : public OpDef {
 public:
  explicit UnpadFaNpd(const char *name) : OpDef(name) {
    this->Input("query")
      .ParamType(REQUIRED)
      .DataType({ge::DT_FLOAT16, ge::DT_BF16})
      .Format({ge::FORMAT_ND, ge::FORMAT_ND})
      .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND})
      .AutoContiguous();
    this->Input("key")
      .ParamType(REQUIRED)
      .DataType({ge::DT_FLOAT16, ge::DT_BF16})
      .Format({ge::FORMAT_ND, ge::FORMAT_ND})
      .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND})
      .AutoContiguous();
    this->Input("value")
      .ParamType(REQUIRED)
      .DataType({ge::DT_FLOAT16, ge::DT_BF16})
      .Format({ge::FORMAT_ND, ge::FORMAT_ND})
      .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND})
      .AutoContiguous();
    this->Input("attn_mask")
      .ParamType(OPTIONAL)
      .DataType({ge::DT_FLOAT16, ge::DT_BF16})
      .Format({ge::FORMAT_ND, ge::FORMAT_ND})
      .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND})
      .AutoContiguous();
    this->Input("actual_seq_qlen")
      .ParamType(REQUIRED)
      .DataType({ge::DT_INT64, ge::DT_INT64})
      .Format({ge::FORMAT_ND, ge::FORMAT_ND})
      .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND})
      .ValueDepend(Option::OPTIONAL, DependScope::TILING)
      .AutoContiguous();
    this->Input("actual_seq_kvlen")
      .ParamType(REQUIRED)
      .DataType({ge::DT_INT64, ge::DT_INT64})
      .Format({ge::FORMAT_ND, ge::FORMAT_ND})
      .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND})
      .ValueDepend(Option::OPTIONAL, DependScope::TILING)
      .AutoContiguous();

    this->Attr("head_num").AttrType(REQUIRED).Int(1);
    this->Attr("scale_value").AttrType(OPTIONAL).Float(1.0f);
    this->Attr("q_input_layout").AttrType(OPTIONAL).String(LAYOUT_TH);
    this->Attr("kv_input_layout").AttrType(OPTIONAL).String(LAYOUT_NPD);
    this->Attr("block_size").AttrType(OPTIONAL).Int(16);
    this->Output("attention_out")
      .ParamType(REQUIRED)
      .DataType({ge::DT_FLOAT16, ge::DT_BF16})
      .Format({ge::FORMAT_ND, ge::FORMAT_ND})
      .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
    this->SetInferShape(ge::UnpadFaNpdInferShape).SetInferDataType(ge::UnpadFaNpdInferDataType);
    this->AICore().SetTiling(optiling::UnpadFaNpdTiling).AddConfig("ascend910b");
  }
};
OP_ADD(UnpadFaNpd);
}  // namespace ops
