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
#include <iterator>
#include <memory>
#include <vector>

#include "pass/add_rms_norm_fusion/add_rms_norm_fusion_pass.h"

namespace mindspore {
namespace opt {

void AddRmsNormFusionPass::DefineSrcPattern(SrcPattern *src_pattern) {
  MS_LOG(INFO) << "Defining source pattern for AddRmsNormFusionPass";
  MS_EXCEPTION_IF_NULL(src_pattern);

  // Complete Add + RmsNorm fusion pattern: RmsNorm(Add(x1, x2), gamma, eps)
  (*src_pattern)
      .AddVar("x1")
      .AddVar("x2")
      .AddVar("gamma")
      .AddVar("eps")
      .AddCNode("add", {std::make_shared<Primitive>("Add"), "x1", "x2"})
      .AddCNode("rms_norm", {std::make_shared<Primitive>("RmsNorm"), "add", "gamma", "eps"});

  MS_LOG(INFO) << "Source pattern defined: RmsNorm(Add(x1, x2), gamma, eps)";
}

AnfNodePtr AddRmsNormFusionPass::BuildAddRmsNorm(const PatternMap &m, const AnfNodePtr &default_node) {
  auto rms_norm_node = m.Get("rms_norm")->cast<CNodePtr>();
  auto add_node = m.Get("add")->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(rms_norm_node);
  MS_EXCEPTION_IF_NULL(add_node);

  auto add_rms_norm_node = default_node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(add_rms_norm_node);

  // Copy RmsNorm node's scope to maintain execution context
  add_rms_norm_node->set_scope(rms_norm_node->scope());

  // Create proper abstract for AddRmsNorm with three outputs: [y, rstd, x]
  // y: same as RmsNorm output (normalized result)
  // rstd: float type with shape [num_row] (reciprocal standard deviation)
  // x: same as Add output (x1 + x2)

  auto rms_norm_abstract = rms_norm_node->abstract();
  auto add_abstract = add_node->abstract();

  if (rms_norm_abstract != nullptr && add_abstract != nullptr) {
    // Create AbstractTuple with three outputs: [y, rstd, x]
    // y: same as RmsNorm first output (normalized result)
    // rstd: same as RmsNorm second output (reciprocal standard deviation)
    // x: same as Add output (x1 + x2)

    std::vector<abstract::AbstractBasePtr> abstract_list;

    // Output 0: y (first output from RmsNorm)
    // Output 1: rstd (second output from RmsNorm)
    if (rms_norm_abstract->isa<abstract::AbstractTuple>()) {
      auto rms_tuple = rms_norm_abstract->cast<abstract::AbstractTuplePtr>();
      const auto &tuple_elements = rms_tuple->elements();
      abstract_list.reserve(tuple_elements.size() + 1);
      std::transform(tuple_elements.cbegin(), tuple_elements.cend(), std::back_inserter(abstract_list),
                     [](const abstract::AbstractBasePtr &element) { return element->Clone(); });
    } else {
      MS_LOG(EXCEPTION) << "RmsNorm abstract must be a tuple";
    }

    abstract_list.push_back(add_abstract->Clone());

    auto abstract_tuple = std::make_shared<abstract::AbstractTuple>(abstract_list);
    add_rms_norm_node->set_abstract(abstract_tuple);
  } else {
    // Throw exception if we can't create proper abstract
    MS_LOG(EXCEPTION) << "Failed to create AddRmsNorm abstract";
  }

  return add_rms_norm_node;
}

void AddRmsNormFusionPass::DefineDstPattern(DstPattern *dst_pattern) {
  MS_LOG(INFO) << "Defining destination pattern for AddRmsNormFusionPass";
  MS_EXCEPTION_IF_NULL(dst_pattern);

  // Use Custom_add_rms_norm kernel for custom implementation support
  // Alternative: AddRmsNorm for internal/aclnn kernel
  (*dst_pattern)
    .AddCNode("add_rms_norm",
              {std::make_shared<Primitive>("AddRmsNorm"), "x1", "x2", "gamma", "eps"},
              BuildAddRmsNorm);

  MS_LOG(INFO) << "Destination pattern defined: AddRmsNorm(x1, x2, gamma, eps)";
}

bool AddRmsNormFusionPass::CheckMatchedDAG(const PatternMap &pattern_map,
                                           const FuncGraphPtr &func_graph,
                                           const AnfNodePtr &node) const {
  auto rms_norm_node = pattern_map.Get("rms_norm");
  if (!rms_norm_node) {
    MS_LOG(ERROR) << "RmsNorm node not found in pattern match";
    return false;
  }

  if (!IsRmsNormNode(rms_norm_node)) {
    MS_LOG(ERROR) << "Node is not an RmsNorm node";
    return false;
  }

  auto add_node = pattern_map.Get("add");
  if (!add_node) {
    MS_LOG(ERROR) << "Add node not found in pattern match";
    return false;
  }

  if (!IsAddNode(add_node)) {
    MS_LOG(ERROR) << "Node is not an Add node";
    return false;
  }

  auto eps_node = pattern_map.Get("eps");
  if (!eps_node) {
    MS_LOG(ERROR) << "eps node not found in pattern match";
    return false;
  }

  auto gamma_node = pattern_map.Get("gamma");
  if (!gamma_node) {
    MS_LOG(ERROR) << "gamma node not found in pattern match";
    return false;
  }

  auto x1_node = pattern_map.Get("x1");
  if (!x1_node) {
    MS_LOG(ERROR) << "x1 node not found in pattern match";
    return false;
  }

  auto x2_node = pattern_map.Get("x2");
  if (!x2_node) {
    MS_LOG(ERROR) << "x2 node not found in pattern match";
    return false;
  }

  return true;
}

bool AddRmsNormFusionPass::IsRmsNormNode(const AnfNodePtr &node) {
  if (!node || !node->isa<CNode>()) {
    return false;
  }

  auto cnode = node->cast<CNodePtr>();
  if (!cnode) {
    return false;
  }

  // Retrieve primitive to verify operation type for RmsNorm identification
  auto prim = GetCNodePrimitive(cnode);
  if (!prim) {
    return false;
  }

  return prim->name() == "RmsNorm";
}

bool AddRmsNormFusionPass::IsAddNode(const AnfNodePtr &node) {
  if (!node || !node->isa<CNode>()) {
    return false;
  }

  auto cnode = node->cast<CNodePtr>();
  if (!cnode) {
    return false;
  }

  // Retrieve primitive to verify operation type for Add identification
  auto prim = GetCNodePrimitive(cnode);
  if (!prim) {
    return false;
  }

  return prim->name() == "Add";
}

bool AddRmsNormFusionPass::CheckInputTypesCompatible(const AnfNodePtr &rms_norm_node, const AnfNodePtr &add_node) {
  // Get Add output type and gamma type for compatibility validation
  auto add_output_type = common::AnfAlgo::GetOutputInferDataType(add_node, 0);
  auto gamma_type = common::AnfAlgo::GetPrevNodeOutputInferDataType(rms_norm_node, 1);

  // Ensure Add output type matches gamma type for proper fusion
  if (add_output_type != gamma_type) {
    MS_LOG(INFO) << "Add output type (" << add_output_type
                 << ") != gamma type (" << gamma_type << ")";
    return false;
  }

  // Get Add input types to verify they are compatible
  auto add_input1_type = common::AnfAlgo::GetPrevNodeOutputInferDataType(add_node, 0);
  auto add_input2_type = common::AnfAlgo::GetPrevNodeOutputInferDataType(add_node, 1);

  // Ensure both Add inputs have the same type for valid addition
  if (add_input1_type != add_input2_type) {
    MS_LOG(INFO) << "Add input types are different: " << add_input1_type << " vs " << add_input2_type;
    return false;
  }

  return true;
}

bool AddRmsNormFusionPass::CheckAddInputShapesCompatible(const AnfNodePtr &add_node) {
  // Get Add input shapes to verify they can be broadcasted together
  auto shape1 = common::AnfAlgo::GetPrevNodeOutputInferShape(add_node, 0);
  auto shape2 = common::AnfAlgo::GetPrevNodeOutputInferShape(add_node, 1);

  // Ensure shapes are compatible for addition operation
  if (shape1 != shape2) {
    MS_LOG(INFO) << "Add input shapes are different";
    return false;
  }

  return true;
}

// Auto-register the pass with the global registry
REGISTER_PASS(AddRmsNormFusionPass)

}  // namespace opt
}  // namespace mindspore
