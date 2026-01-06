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

#ifndef __MS_CUSTOM_OPS_C_API_UTILS_CHECK_UTILS_H_
#define __MS_CUSTOM_OPS_C_API_UTILS_CHECK_UTILS_H_

#include <cstdint>
#include <string>
#include <vector>
#include "mindspore/include/custom_op_api.h"

#ifndef MS_CUSTOM_OPS_UNLIKELY
#define MS_CUSTOM_OPS_LIKELY(x) __builtin_expect(!!(x), 1)
#define MS_CUSTOM_OPS_UNLIKELY(x) __builtin_expect(!!(x), 0)
#endif

namespace ms_custom_ops {
#define CHECK_TENSOR_DIM_GE(tensor_var, min_dim, op_name)                                            \
  do {                                                                                               \
    if ((tensor_var).size() < (min_dim)) {                                                           \
      MS_LOG(EXCEPTION) << (op_name) << "requires at least " << (min_dim) << " dimensions, but got " \
                        << (tensor_var).size();                                                      \
    }                                                                                                \
  } while (0)

#define CHECK_TENSOR_DIM_RANGE(shape_var, min_dim, max_dim, op_name, tensor_name, idx)                            \
  do {                                                                                                            \
    const auto &shape_ref = (shape_var);                                                                          \
    const size_t dim_size = shape_ref.size();                                                                     \
    if (dim_size < (min_dim)) {                                                                                   \
      MS_LOG(EXCEPTION) << (op_name) << " requires the input tensor '" << (tensor_name) << "' at index " << (idx) \
                        << " to have at least " << (min_dim) << " dimensions, but got " << dim_size;              \
    }                                                                                                             \
    if (dim_size > (max_dim)) {                                                                                   \
      MS_LOG(EXCEPTION) << (op_name) << " requires the input tensor '" << (tensor_name) << "' at index " << (idx) \
                        << " to have at most " << (max_dim) << " dimensions, but got " << dim_size;               \
    }                                                                                                             \
  } while (0)

#define MS_CUSTOM_OPS_EXCEPTION_IF_CHECK_FAIL(condition, error_info) \
  do {                                                               \
    if (!(condition)) {                                              \
      MS_LOG(EXCEPTION) << "Failure info [" << error_info << "].";   \
    }                                                                \
  } while (0)

#define MS_CUSTOM_OPS_CHECK_VALUE(cond, msg)        \
  do {                                              \
    if (MS_CUSTOM_OPS_LIKELY(!(cond))) {            \
      MS_EXCEPTION(mindspore::ValueError) << (msg); \
    }                                               \
  } while (0)
}  // namespace ms_custom_ops

#endif  // __MS_CUSTOM_OPS_C_API_UTILS_CHECK_UTILS_H_
