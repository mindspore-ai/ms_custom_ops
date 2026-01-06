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
#ifndef MINDSPORE_CUSTOM_PASS_PASS_REGISTRY_H_
#define MINDSPORE_CUSTOM_PASS_PASS_REGISTRY_H_

#include <memory>
#include <vector>
#include <functional>
#include <string>
#include <map>
#include "mindspore/include/custom_pass_api.h"

namespace mindspore {
namespace opt {

/**
 * @brief Pass factory function type
 */
using PassFactory = std::function<std::shared_ptr<Pass>()>;

/**
 * @brief Simplified pass registry for basic pass registration
 *
 * This is a simplified version that removes complex thread safety
 * and metadata management while keeping essential functionality.
 */
class PassRegistry {
 public:
  static PassRegistry& Instance() {
    static PassRegistry instance;
    return instance;
  }

  /**
   * @brief Register a pass with the registry
   * @param factory Pass factory function
   */
  void RegisterPass(PassFactory factory);

  /**
   * @brief Register a named pass with the registry
   * @param pass_name Pass name
   * @param factory Pass factory function
   */
  void RegisterNamedPass(const std::string& pass_name, PassFactory factory);

  /**
   * @brief Get all registered passes
   * @return Vector of pass factory functions
   */
  std::vector<PassFactory> GetAllPasses() const;

  /**
   * @brief Get all registered pass names
   * @return Vector of pass names
   */
  std::vector<std::string> GetAllPassNames() const;

  /**
   * @brief Get pass factory by name
   * @param pass_name Pass name
   * @return Pass factory function or nullptr if not found
   */
  PassFactory GetPassFactory(const std::string& pass_name) const;

  /**
   * @brief Clear all registrations
   */
  void Clear();

 private:
  PassRegistry() = default;
  ~PassRegistry() = default;
  PassRegistry(const PassRegistry&) = delete;
  PassRegistry& operator=(const PassRegistry&) = delete;

  std::vector<PassFactory> factories_;
  std::map<std::string, PassFactory> named_factories_;  // pass_name -> factory
};

/**
 * @brief Simplified helper macro to register a pass
 *
 * Usage:
 * REGISTER_PASS(MyPass)
 */
#define REGISTER_PASS(PassClass) \
  static bool PassClass##_registered = []() { \
    auto factory = []() -> std::shared_ptr<mindspore::opt::Pass> { \
      return std::make_shared<PassClass>(); \
    }; \
    mindspore::opt::PassRegistry::Instance().RegisterPass(factory); \
    mindspore::opt::PassRegistry::Instance().RegisterNamedPass(#PassClass, factory); \
    return true; \
  }();

}  // namespace opt
}  // namespace mindspore

#endif  // MINDSPORE_CUSTOM_PASS_PASS_REGISTRY_H_
