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

#include <string>
#include <vector>
#include "ops/c_api/sparse_flash_attention/sparse_flash_attention_common.h"
#include "ops/framework/utils.h"
#include "mindspore/include/custom_op_api.h"
#include "ops/framework/module.h"

namespace ms_custom_ops {
ms::Tensor sparse_flash_attention_ascendc(
  const ms::Tensor &query, const ms::Tensor &key, const ms::Tensor &value, const ms::Tensor &sparse_indices,
  const std::optional<ms::Tensor> &block_table, const std::optional<ms::Tensor> &query_seq_len,
  const std::optional<ms::Tensor> &k_seq_len, const std::optional<ms::Tensor> &q_rope,
  const std::optional<ms::Tensor> &k_rope, const float scale_value, const int64_t sparse_block_size,
  const int64_t &layout_query, const int64_t &layout_kv, const int64_t sparse_mode) {
  static constexpr auto op_name = "sparse_flash_attention_ascendc";
  auto runner = std::make_shared<ms::pynative::AclnnOpRunner>(op_name);
  MS_EXCEPTION_IF_NULL(runner);

  CheckLayout(layout_query, layout_kv);
  const auto &layout_q_str = kQueryLayouts[static_cast<size_t>(layout_query)];
  const auto &layout_kv_str = kKVLayouts[static_cast<size_t>(layout_kv)];

  auto query_shape = query.shape();
  auto out_attention = ms::Tensor(query.data_type(), query_shape);
  runner->SetLaunchFunc(LAUNCH_ACLNN_FUNC(aclnnSparseFlashAttention, query, key, value, sparse_indices, block_table,
                                          query_seq_len, k_seq_len, q_rope, k_rope, static_cast<double>(scale_value),
                                          sparse_block_size, layout_q_str, layout_kv_str, sparse_mode, out_attention));
  std::vector<ms::Tensor> inputs = {query,
                                    key,
                                    value,
                                    sparse_indices,
                                    GetTensorOrEmpty(block_table),
                                    GetTensorOrEmpty(query_seq_len),
                                    GetTensorOrEmpty(k_seq_len),
                                    GetTensorOrEmpty(q_rope),
                                    GetTensorOrEmpty(k_rope)};
  runner->Run(inputs, {out_attention});
  return out_attention;
}

auto pyboost_sfa(const ms::Tensor &query, const ms::Tensor &key, const ms::Tensor &value,
                 const ms::Tensor &sparse_indices, const std::optional<ms::Tensor> &block_table,
                 const std::optional<ms::Tensor> &query_seq_len, const std::optional<ms::Tensor> &k_seq_len,
                 const std::optional<ms::Tensor> &q_rope, const std::optional<ms::Tensor> &k_rope,
                 const float scale_value, const int64_t sparse_block_size, const int64_t &layout_query,
                 const int64_t &layout_kv, const int64_t sparse_mode) {
  static constexpr auto kSFAOutNum = 1;
  return ms::pynative::PyboostRunner::Call<kSFAOutNum>(
    sparse_flash_attention_ascendc, query, key, value, sparse_indices, block_table, query_seq_len, k_seq_len, q_rope,
    k_rope, scale_value, sparse_block_size, layout_query, layout_kv, sparse_mode);
}
}  // namespace ms_custom_ops

MS_CUSTOM_OPS_EXTENSION_MODULE(m) {
  m.def("sparse_flash_attention", &ms_custom_ops::pyboost_sfa, "Sparse flash attention", pybind11::arg("query"),
        pybind11::arg("key"), pybind11::arg("value"), pybind11::arg("sparse_indices"),
        pybind11::arg("block_table") = std::nullopt, pybind11::arg("actual_seq_lengths_query") = std::nullopt,
        pybind11::arg("actual_seq_lenghts_kv") = std::nullopt, pybind11::arg("query_rope") = std::nullopt,
        pybind11::arg("key_rope") = std::nullopt, pybind11::arg("scale_value") = 1.0,
        pybind11::arg("sparse_block_size") = 1, pybind11::arg("layout_query") = 0, pybind11::arg("layout_kv") = 0,
        pybind11::arg("sparse_mode") = 3);
}
