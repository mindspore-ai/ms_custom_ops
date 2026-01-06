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

#include <pybind11/pybind11.h>
#include <iostream>
#include "ops/framework/module.h"
#include "mindspore/include/custom_op_api.h"

namespace ms_custom_ops {
void LoadOpApiLibray() {
  MS_LOG(INFO) << "ms_custom_ops load op api library";
  mindspore::device::ascend::LoadOpApiLib();
}
}  // namespace ms_custom_ops

MS_CUSTOM_OPS_EXTENSION_MODULE(m) {
  m.def("load_op_api_library", &ms_custom_ops::LoadOpApiLibray, "Load the library by loading op api library");
}
