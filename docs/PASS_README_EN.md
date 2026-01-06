# MsCustomOps Custom Pass

[Chinese Version](PASS_README.md) | English

This document provides a detailed description of the Pass auto-registration mechanism based on the MindSpore framework, implementing custom passes in MsCustomOps.

## Overview

## Custom Pass Mechanism

For detailed information about MindSpore CustomPass mechanism, please refer to [Custom Backend Pass Framework (Experimental Interface)](https://gitee.com/mindspore/mindspore/issues/ICYCV5?from=project-issue&search_text=RFC).

## Pass Development Guide

### 1. Pass Registration Implementation

Add the registration macro at the end of the Pass header file:

```cpp
// Example: Add at the end of add_rms_norm_fusion_pass.h
REGISTER_PASS(AddRmsNormFusionPass)
```

### 2. Pass Implementation Design

#### Pass Header File Design

```cpp
// my_custom_pass.h
#ifndef MINDSPORE_CUSTOM_PASS_MY_CUSTOM_PASS_H_
#define MINDSPORE_CUSTOM_PASS_MY_CUSTOM_PASS_H_

#include "include/backend/optimizer/pass.h"
#include "include/backend/optimizer/pattern_to_pattern.h"
#include "utils/log_adapter.h"
#include "pass_registry.h"

namespace mindspore {
namespace opt {

class MyCustomPass : public PatternToPatternPass {
 public:
  MyCustomPass() : PatternToPatternPass("MyCustomPass") {}

  void DefineSrcPattern(SrcPattern* src_pattern) override;
  void DefineDstPattern(DstPattern* dst_pattern) override;
  bool CheckMatchedDAG(const PatternMap& pattern_map,
                       const FuncGraphPtr& func_graph,
                       const AnfNodePtr& node) const override;

 private:
  static AnfNodePtr BuildOptimizedOp(const PatternMap& m, const AnfNodePtr& default_node);
};

}  // namespace opt
}  // namespace mindspore

// Auto-registration macro
REGISTER_PASS(MyCustomPass)

#endif  // MINDSPORE_CUSTOM_PASS_MY_CUSTOM_PASS_H_
```

#### Pass Logic Implementation Design

```cpp
// my_custom_pass.cc
#include "my_custom_pass.h"
#include "ir/primitive.h"

namespace mindspore {
namespace opt {

void MyCustomPass::DefineSrcPattern(SrcPattern* src_pattern) {
  // Define source pattern: match specific operation sequence
  (*src_pattern)
    .AddVar("input1")
    .AddVar("input2")
    .AddCNode("target_op", {prim::kPrimTargetOp, "input1", "input2"});
}

void MyCustomPass::DefineDstPattern(DstPattern* dst_pattern) {
  // Define target pattern: replace with optimized operation
  (*dst_pattern).AddCNode("optimized_op",
    {prim::kPrimOptimizedOp, "input1", "input2"},
    BuildOptimizedOp);
}

bool MyCustomPass::CheckMatchedDAG(const PatternMap& pattern_map,
                                   const FuncGraphPtr& func_graph,
                                   const AnfNodePtr& node) const {
  // Custom validation logic: check matching conditions
  // - Verify input node existence
  // - Check node type compatibility
  // - Validate shape and data type requirements
  return true;
}

AnfNodePtr MyCustomPass::BuildOptimizedOp(const PatternMap& m, const AnfNodePtr& default_node) {
  // Build optimized node
  auto input1 = m.Get("input1");
  auto input2 = m.Get("input2");

  if (!input1 || !input2) {
    return nullptr;
  }

  // Create new optimized node
  auto func_graph = default_node->func_graph();
  auto prim = prim::GetPrimitiveFromCNode(default_node);

  return func_graph->NewCNode({prim, input1, input2});
}

}  // namespace opt
}  // namespace mindspore
```

### 3. Pass Design Pattern Examples

#### AddRmsNormFusionPass Design

Optimization goal: Fuse `RmsNorm(Add(x1, x2), gamma, eps)` into `AddRmsNorm(x1, x2, gamma, eps)`

```cpp
// Source pattern definition
void AddRmsNormFusionPass::DefineSrcPattern(SrcPattern* src_pattern) {
  (*src_pattern)
    .AddVar("x1")
    .AddVar("x2")
    .AddVar("gamma")
    .AddVar("eps")
    .AddCNode("add", {prim::kPrimAdd, "x1", "x2"})
    .AddCNode("rms_norm", {prim::kPrimRmsNorm, "add", "gamma", "eps"});
}

// Target pattern definition
void AddRmsNormFusionPass::DefineDstPattern(DstPattern* dst_pattern) {
  (*dst_pattern).AddCNode("add_rms_norm",
    {std::make_shared<Primitive>("AddRmsNorm"), "x1", "x2", "gamma", "eps"},
    BuildAddRmsNorm);
}

// Validation logic design
bool AddRmsNormFusionPass::CheckMatchedDAG(const PatternMap& pattern_map,
                                           const FuncGraphPtr& func_graph,
                                           const AnfNodePtr& node) const {
  // Check RmsNorm and Add node validity
  auto rms_norm_node = pattern_map.Get("rms_norm");
  auto add_node = pattern_map.Get("add");

  if (!rms_norm_node || !add_node) {
    return false;
  }

  // Verify input type compatibility
  if (!CheckInputTypesCompatible(rms_norm_node, add_node)) {
    return false;
  }

  // Verify Add input shape compatibility
  if (!CheckAddInputShapesCompatible(add_node)) {
    return false;
  }

  return true;
}
```

#### Python Application Integration

`ms_custom_ops` provides a simple Python API for registering custom passes:

```python
import mindspore as ms
from mindspore import nn, Tensor
import ms_custom_ops
import numpy as np

# Register custom Pass (automatically finds plugin path)
ms_custom_ops.register_custom_pass("AddRmsNormFusionPass", backend="ascend")
ms_custom_ops.register_custom_pass("ReplaceAddNFusionPass", backend="cpu")

# Define network
class MyNetwork(nn.Cell):
    def __init__(self):
        super().__init__()
        self.dense = nn.Dense(10, 10)

    def construct(self, x):
        return self.dense(x)

# Set runtime environment
ms.set_context(device_target="Ascend")

# Create model and data
model = MyNetwork()
data = Tensor(np.random.rand(32, 10).astype(np.float32))

# Model inference - Pass will be automatically applied during compilation
output = model(data)
print(f"Output shape: {output.shape}")
```

#### API Parameter Description

- **pass_name**: Pass name, must match the registered Pass name in the plugin
- **backend**: Target backend device type for device-specific optimization (default: "ascend")
