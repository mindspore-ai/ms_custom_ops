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
#include <memory>
#include <string>
#include <vector>
#include <algorithm>

#include "pass/pass_registry.h"

// Compile-time log: Display HashMap type
#ifdef LOG_HASHMAP_TYPE
#ifdef ENABLE_FAST_HASH_TABLE
#ifdef __has_include
#if __has_include("include/robin_hood.h")
#pragma message("=== COMPILE INFO: Using robin_hood::unordered_map for HashMap ===")
#else
#pragma message("=== COMPILE INFO: ENABLE_FAST_HASH_TABLE=1 but robin_hood.h not found, using std::unordered_map ===")
#endif
#else
#pragma message("=== COMPILE INFO: ENABLE_FAST_HASH_TABLE=1 but __has_include missing; using std::unordered_map ===")
#endif
#else
#pragma message("=== COMPILE INFO: ENABLE_FAST_HASH_TABLE=0, using std::unordered_map for HashMap ===")
#endif
#endif

// Include MindSpore pass plugin interface
#include "mindspore/include/custom_pass_api.h"

namespace mindspore {
namespace opt {

// Plugin class implementation
class MSCustomPassPlugin : public CustomPassPlugin {
 public:
  std::string GetPluginName() const override {
    return "ms_custom_pass_plugin";
  }

  std::vector<std::string> GetAvailablePassNames() const override {
    auto &registry = PassRegistry::Instance();
    auto pass_names = registry.GetAllPassNames();
    MS_LOG(INFO) << "GetAvailablePassNames called, found " << pass_names.size() << " passes";
    return pass_names;
  }

  std::shared_ptr<Pass> CreatePass(const std::string& pass_name) const override {
    auto &registry = PassRegistry::Instance();
    auto factory = registry.GetPassFactory(pass_name);

    if (factory) {
      auto pass = factory();
      MS_LOG(INFO) << "Created pass '" << pass_name << "' successfully";
      return pass;
    } else {
      MS_LOG(WARNING) << "Pass '" << pass_name << "' not found in registry";
      return nullptr;
    }
  }
};

}  // namespace opt
}  // namespace mindspore

// Use standard plugin macro to automatically generate export function
EXPORT_CUSTOM_PASS_PLUGIN(mindspore::opt::MSCustomPassPlugin)
