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
#include <algorithm>

#include "pass/pass_registry.h"

namespace mindspore {
namespace opt {

void PassRegistry::RegisterPass(PassFactory factory) {
  factories_.push_back(factory);
}

void PassRegistry::RegisterNamedPass(const std::string& pass_name, PassFactory factory) {
  named_factories_[pass_name] = factory;
}

std::vector<PassFactory> PassRegistry::GetAllPasses() const {
  return factories_;
}

std::vector<std::string> PassRegistry::GetAllPassNames() const {
  std::vector<std::string> names(named_factories_.size());
  std::transform(named_factories_.begin(), named_factories_.end(), names.begin(),
                 [](const auto &named_factory) { return named_factory.first; });
  return names;
}

PassFactory PassRegistry::GetPassFactory(const std::string& pass_name) const {
  auto it = named_factories_.find(pass_name);
  if (it != named_factories_.end()) {
    return it->second;
  }
  return nullptr;
}

void PassRegistry::Clear() {
  factories_.clear();
  named_factories_.clear();
}

}  // namespace opt
}  // namespace mindspore
