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
#ifndef MINDSPORE_CUSTOM_PASS_ADD_RMS_NORM_FUSION_PASS_H_
#define MINDSPORE_CUSTOM_PASS_ADD_RMS_NORM_FUSION_PASS_H_

#include "pass/pass_registry.h"

namespace mindspore {
namespace opt {

/**
 * @brief Pass to fuse Add and RmsNorm operations for performance optimization
 *
 * Transforms RmsNorm(Add(x1, x2), gamma, eps) into AddRmsNorm(x1, x2, gamma, eps)
 * to reduce computation overhead and improve memory efficiency
 * Inherits from PatternToPatternPass to comply with MindSpore plugin system requirements
 */
class AddRmsNormFusionPass : public PatternToPatternPass {
 public:
  AddRmsNormFusionPass() : PatternToPatternPass("AddRmsNormFusionPass") {}

  void DefineSrcPattern(SrcPattern *src_pattern) override;
  void DefineDstPattern(DstPattern *dst_pattern) override;
  bool CheckMatchedDAG(const PatternMap &pattern_map,
                       const FuncGraphPtr &func_graph,
                       const AnfNodePtr &node) const override;

 private:
  static bool IsRmsNormNode(const AnfNodePtr &node);
  static bool IsAddNode(const AnfNodePtr &node);
  static bool CheckInputTypesCompatible(const AnfNodePtr &rms_norm_node, const AnfNodePtr &add_node);
  static bool CheckAddInputShapesCompatible(const AnfNodePtr &add_node);

  static AnfNodePtr BuildAddRmsNorm(const PatternMap &m, const AnfNodePtr &default_node);
};

}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CUSTOM_PASS_ADD_RMS_NORM_FUSION_PASS_H_
