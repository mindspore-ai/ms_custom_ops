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

#include <string>
#include <vector>
#include "ops/c_api/lightning_indexer/lightning_indexer_common.h"
#include "ops/framework/utils.h"
#include "mindspore/include/custom_op_api.h"
#include "ops/framework/module.h"

namespace ms_custom_ops {
ms::Tensor lightning_indexer_ascendc(const ms::Tensor &query, const ms::Tensor &key, const ms::Tensor &weights,
                                     const std::optional<ms::Tensor> &query_seq_len,
                                     const std::optional<ms::Tensor> &k_seq_len,
                                     const std::optional<ms::Tensor> &block_table, const int64_t &layout_query,
                                     const int64_t &layout_key, const int64_t sparse_count, const int64_t sparse_mode) {
  static constexpr auto op_name = "lightning_indexer_ascendc";
  auto runner = std::make_shared<ms::pynative::AclnnOpRunner>(op_name);
  MS_EXCEPTION_IF_NULL(runner);

  auto query_shape = query.shape();
  auto key_shape = key.shape();
  auto out_shape = InferShapeForLightningIndexer(query_shape, key_shape, layout_query, layout_key, sparse_count);

  auto out_sparse_indices = ms::Tensor(k_seq_len.value().data_type(), out_shape);
  runner->SetLaunchFunc(LAUNCH_ACLNN_FUNC(aclnnLightningIndexer, query, key, weights, query_seq_len, k_seq_len,
                                          block_table, layout_query, layout_key, sparse_count, sparse_mode,
                                          out_sparse_indices));
  std::vector<ms::Tensor> inputs = {
    query, key, weights, GetTensorOrEmpty(query_seq_len), GetTensorOrEmpty(k_seq_len), GetTensorOrEmpty(block_table)};
  runner->Run(inputs, {out_sparse_indices});
  return out_sparse_indices;
}

auto pyboost_li(const ms::Tensor &query, const ms::Tensor &key, const ms::Tensor &weights,
                const std::optional<ms::Tensor> &query_seq_len, const std::optional<ms::Tensor> &k_seq_len,
                const std::optional<ms::Tensor> &block_table, const int64_t &layout_query, const int64_t &layout_key,
                const int64_t sparse_count, const int64_t sparse_mode) {
  static constexpr auto kLIOutNum = 1;
  return ms::pynative::PyboostRunner::Call<kLIOutNum>(lightning_indexer_ascendc, query, key, weights, query_seq_len,
                                                      k_seq_len, block_table, layout_query, layout_key, sparse_count,
                                                      sparse_mode);
}
}  // namespace ms_custom_ops

MS_CUSTOM_OPS_EXTENSION_MODULE(m) {
  m.def("lightning_indexer", &ms_custom_ops::pyboost_li, "Lightning indexer", pybind11::arg("query"),
        pybind11::arg("key"), pybind11::arg("weights"), pybind11::arg("actual_seq_lengths_query") = std::nullopt,
        pybind11::arg("actual_seq_lengths_key") = std::nullopt, pybind11::arg("block_table") = std::nullopt,
        pybind11::arg("layout_query") = 0, pybind11::arg("layout_key") = 0, pybind11::arg("sparse_count") = 2048,
        pybind11::arg("sparse_mode") = 3);
}
