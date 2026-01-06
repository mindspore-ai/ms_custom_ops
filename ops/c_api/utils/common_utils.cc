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

#include <cstdint>
#include <string>
#include <vector>
#include <map>
#include <mutex>
#include "mindspore/include/custom_op_api.h"
#include "ops/framework/aclnn/graphmode/aclnn_kernel_mod.h"
#include "ops/framework/utils.h"
#include "ops/c_api/utils/common_utils.h"

namespace ms_custom_ops {

const std::map<std::string, std::string> AscendSocVersionMap = {
  {"Ascend910A", "ascend910"},        {"Ascend910B", "ascend910"},        {"Ascend910PremiumA", "ascend910"},
  {"Ascend910ProA", "ascend910"},     {"Ascend910ProB", "ascend910"},     {"Ascend910B1", "ascend910b"},
  {"Ascend910B2", "ascend910b"},      {"Ascend910B2C", "ascend910b"},     {"Ascend910B3", "ascend910b"},
  {"Ascend910B4", "ascend910b"},      {"Ascend910B4-1", "ascend910b"},    {"Ascend910_9391", "ascend910_93"},
  {"Ascend910_9392", "ascend910_93"}, {"Ascend910_9381", "ascend910_93"}, {"Ascend910_9382", "ascend910_93"},
  {"Ascend910_9372", "ascend910_93"}, {"Ascend910_9361", "ascend910_93"}, {"Ascend910_9362", "ascend910_93"},
  {"Ascend310P", "ascend310p"},       {"Ascend310P3", "ascend310p"},      {"Ascend310B4", "ascend310b"},
  {"Ascend310B1", "ascend310b"},      {"Ascend310", "ascend310"}};

static std::once_flag once_flag;

static std::string GetSocVersion() {
  static std::string soc_version;
  std::call_once(once_flag, [&]() {
    const char *soc_name_ptr = aclrtGetSocName();
    if (soc_name_ptr != nullptr) {
      auto iter = AscendSocVersionMap.find(soc_name_ptr);
      soc_version = (iter != AscendSocVersionMap.end()) ? iter->second : "";
    } else {
      soc_version.clear();
    }
  });
  return soc_version;
}

static bool CheckSocVersion(const std::string &expected_soc_version) {
  const auto &version = GetSocVersion();
  return !version.empty() && (version == expected_soc_version);
}

bool IsSoc910b() { return CheckSocVersion(kAscendVersion910b); }

bool IsSoc910_93() { return CheckSocVersion(kAscendVersion910_93); }

bool IsSoc310p() { return CheckSocVersion(kAscendVersion310p); }
bool IsSoc910a() { return CheckSocVersion(kAscendVersion910); }

bool IsSoc910BC() { return IsSoc910b() || IsSoc910_93(); }
}  // namespace ms_custom_ops
