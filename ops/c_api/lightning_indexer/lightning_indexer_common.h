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

#ifndef __MS_CUSTOM_OPS_OPS_C_API_LIGHTNING_INDEXER_LIGHTNING_INDEXER_COMMON_H__
#define __MS_CUSTOM_OPS_OPS_C_API_LIGHTNING_INDEXER_LIGHTNING_INDEXER_COMMON_H__

#include <cstdint>
#include "mindspore/include/custom_op_api.h"
#include "ops/c_api/utils/check_utils.h"

namespace ms_custom_ops {
static constexpr auto kQueryIndex = 0;
static constexpr auto kKeyIndex = 1;
static constexpr auto kWeightsIndex = 2;
static constexpr auto kQSeqLenIndex = 3;
static constexpr auto kKSeqLenIndex = 4;
static constexpr auto kBlockTable = 5;
static constexpr auto kLayoutQueryIndex = 6;
static constexpr auto kLayoutKeyIndex = 7;
static constexpr auto kSparseCountIndex = 8;
static constexpr auto kSparseModeIndex = 9;

static constexpr auto kTNDStr = "TND";
static constexpr auto kBSNDStr = "BSND";
static constexpr auto kPA_BSNDStr = "PA_BSND";

static constexpr auto kQueryTndRank = 3;
static constexpr auto kQueryBsndRank = 4;
static constexpr auto kKeyPaBsndRank = 4;

static constexpr auto kKeyHeadDim = 2;
static constexpr auto kOutTndHeadDim = 1;
static constexpr auto kOutBsndHeadDim = 2;
static constexpr auto kOutTndCountDim = 2;
static constexpr auto kOutBsndCountDim = 3;

static const std::vector<std::string> kQueryLayouts = {kBSNDStr, kTNDStr};
static const std::vector<std::string> kKeyLayouts = {kPA_BSNDStr};
static const std::vector<size_t> kQueryRanks = {kQueryBsndRank, kQueryTndRank};
static const std::vector<size_t> kKeyRanks = {kKeyPaBsndRank};
static constexpr auto kQueryLayoutMaxInt = 1;
static constexpr auto kKeyLayoutMaxInt = 0;

inline void CheckShape(const ShapeVector &query_shape, const ShapeVector &key_shape, const int64_t &layout_query,
                       const int64_t &layout_key) {
  const auto q_layout_index = static_cast<size_t>(layout_query);
  const auto key_layout_index = static_cast<size_t>(layout_query);
  if (q_layout_index > kQueryLayoutMaxInt) {
    MS_LOG(EXCEPTION) << "The layout of query for lightning indexer should be in range [0, " << kQueryLayoutMaxInt
                      << "], but got: " << layout_query;
  }

  if (key_layout_index > kKeyLayoutMaxInt) {
    MS_LOG(EXCEPTION) << "The layout of key for lightning indexer should be in range [0, " << kKeyLayoutMaxInt
                      << "], but got: " << layout_key;
  }

  MS_CUSTOM_OPS_CHECK_VALUE(query_shape.size() == kQueryRanks[q_layout_index],
                            mindspore::CheckAndConvertUtils::FormatCommMsg(
                              "The rank of query should be ", kQueryRanks[q_layout_index], "when layout is ",
                              kQueryLayouts[q_layout_index], ", but got shape ", query_shape));

  MS_CUSTOM_OPS_CHECK_VALUE(key_shape.size() == kKeyRanks[key_layout_index],
                            mindspore::CheckAndConvertUtils::FormatCommMsg(
                              "The rank of key should be ", kKeyRanks[key_layout_index], "when layout is ",
                              kKeyLayouts[key_layout_index], ", but got shape ", key_shape));
}

inline ShapeVector InferShapeForLightningIndexer(const ShapeVector &query_shape, const ShapeVector &key_shape,
                                                 int64_t layout_query, int64_t layout_key, int64_t sparse_count) {
  CheckShape(query_shape, key_shape, layout_query, layout_key);

  auto out_shape = query_shape;
  if (kQueryLayouts[static_cast<size_t>(layout_query)] == kBSNDStr) {
    out_shape[kOutBsndHeadDim] = key_shape[kKeyHeadDim];
    out_shape[kOutBsndCountDim] = sparse_count;
  } else {
    out_shape[kOutTndHeadDim] = key_shape[kKeyHeadDim];
    out_shape[kOutTndCountDim] = sparse_count;
  }
  return out_shape;
}
}  // namespace ms_custom_ops
#endif  //  __MS_CUSTOM_OPS_OPS_C_API_LIGHTNING_INDEXER_LIGHTNING_INDEXER_COMMON_H__
