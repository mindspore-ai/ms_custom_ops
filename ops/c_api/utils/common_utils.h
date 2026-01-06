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

#ifndef __MS_CUSTOM_OPS_C_API_UTILS_COMMON_UTILS_H_
#define __MS_CUSTOM_OPS_C_API_UTILS_COMMON_UTILS_H_

#include <cstdint>
#include <string>
#include <vector>
#include "mindspore/include/custom_op_api.h"

namespace ms_custom_ops {
bool IsSoc910b();
bool IsSoc910_93();
bool IsSoc310p();
bool IsSoc910a();
bool IsSoc910BC();
}  // namespace ms_custom_ops

#endif  // __MS_CUSTOM_OPS_C_API_UTILS_COMMON_UTILS_H_
