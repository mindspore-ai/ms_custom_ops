/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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
#include "pass/convert_tuple_input_to_dynamic_input/convert_tuple_input_to_dynamic_input.h"
#include <memory>
#include <unordered_set>

namespace mindspore {
namespace opt {
constexpr auto kCustomGroupedMatmulV4 = "Custom_grouped_matmul_v4_cops";
const BaseRef ConvertTupleInputToDynamicInput::DefinePattern() const {
  VarPtr V = std::make_shared<Var>();
  VarPtr Xs = std::make_shared<SeqVar>();
  return VectorRef({V, Xs});
}

const AnfNodePtr ConvertTupleInputToDynamicInput::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                                          const EquivPtr &) const {
  if (node == nullptr || !node->isa<CNode>() || !AnfUtils::IsRealKernel(node)) {
    return nullptr;
  }
  static const std::unordered_set<std::string> need_unfold_calculate_node = {kCustomGroupedMatmulV4};
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  PrimitivePtr prim = common::AnfAlgo::GetCNodePrimitive(cnode);
  MS_EXCEPTION_IF_NULL(prim);
  auto node_name = prim->name();
  if (need_unfold_calculate_node.find(node_name) != need_unfold_calculate_node.end()) {
    return ConvertMakeTupleInputToPlantInputs(func_graph, cnode);
  }
  return nullptr;
}
REGISTER_PASS(ConvertTupleInputToDynamicInput)
}  // namespace opt
}  // namespace mindspore
