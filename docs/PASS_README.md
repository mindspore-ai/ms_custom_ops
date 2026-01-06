# MsCustomOps 自定义Pass

中文版 | [English](PASS_README_EN.md)

本文档详细描述了基于 MindSpore 框架的 Pass 自动注册机制，在MsCustomOps中实现自定义 Pass。

## 概述

## 自定义Pass机制

具体MindSpore CustomPass机制参考 [自定义后端Pass框架(实验性接口)](https://gitee.com/mindspore/mindspore/issues/ICYCV5?from=project-issue&search_text=RFC) 。

## Pass 开发指南

### 1. Pass 注册实现

在 Pass 头文件的末尾添加注册宏：

```cpp
// 示例：在 add_rms_norm_fusion_pass.h 末尾添加
REGISTER_PASS(AddRmsNormFusionPass)
```

### 2. Pass 实现设计

#### Pass 头文件设计

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

} // namespace opt
} // namespace mindspore

// 自动注册宏
REGISTER_PASS(MyCustomPass)

#endif // MINDSPORE_CUSTOM_PASS_MY_CUSTOM_PASS_H_
```

#### Pass 逻辑实现设计

```cpp
// my_custom_pass.cc
#include "my_custom_pass.h"
#include "ir/primitive.h"

namespace mindspore {
namespace opt {

void MyCustomPass::DefineSrcPattern(SrcPattern* src_pattern) {
  // 定义源模式：匹配特定的操作序列
  (*src_pattern)
    .AddVar("input1")
    .AddVar("input2")
    .AddCNode("target_op", {prim::kPrimTargetOp, "input1", "input2"});
}

void MyCustomPass::DefineDstPattern(DstPattern* dst_pattern) {
  // 定义目标模式：替换为优化后的操作
  (*dst_pattern).AddCNode("optimized_op",
    {prim::kPrimOptimizedOp, "input1", "input2"},
    BuildOptimizedOp);
}

bool MyCustomPass::CheckMatchedDAG(const PatternMap& pattern_map,
                                   const FuncGraphPtr& func_graph,
                                   const AnfNodePtr& node) const {
  // 自定义验证逻辑：检查匹配条件
  // - 验证输入节点存在性
  // - 检查节点类型兼容性
  // - 验证形状和数据类型要求
  return true;
}

AnfNodePtr MyCustomPass::BuildOptimizedOp(const PatternMap& m, const AnfNodePtr& default_node) {
  // 构建优化后的节点
  auto input1 = m.Get("input1");
  auto input2 = m.Get("input2");

  if (!input1 || !input2) {
    return nullptr;
  }

  // 创建新的优化节点
  auto func_graph = default_node->func_graph();
  auto prim = prim::GetPrimitiveFromCNode(default_node);

  return func_graph->NewCNode({prim, input1, input2});
}

} // namespace opt
} // namespace mindspore
```

### 3. Pass 设计模式示例

#### AddRmsNormFusionPass 设计

优化目标：将 `RmsNorm(Add(x1, x2), gamma, eps)` 融合为 `AddRmsNorm(x1, x2, gamma, eps)`

```cpp
// 源模式定义
void AddRmsNormFusionPass::DefineSrcPattern(SrcPattern* src_pattern) {
  (*src_pattern)
    .AddVar("x1")
    .AddVar("x2")
    .AddVar("gamma")
    .AddVar("eps")
    .AddCNode("add", {prim::kPrimAdd, "x1", "x2"})
    .AddCNode("rms_norm", {prim::kPrimRmsNorm, "add", "gamma", "eps"});
}

// 目标模式定义
void AddRmsNormFusionPass::DefineDstPattern(DstPattern* dst_pattern) {
  (*dst_pattern).AddCNode("add_rms_norm",
    {std::make_shared<Primitive>("AddRmsNorm"), "x1", "x2", "gamma", "eps"},
    BuildAddRmsNorm);
}

// 验证逻辑设计
bool AddRmsNormFusionPass::CheckMatchedDAG(const PatternMap& pattern_map,
                                           const FuncGraphPtr& func_graph,
                                           const AnfNodePtr& node) const {
  // 检查 RmsNorm 和 Add 节点有效性
  auto rms_norm_node = pattern_map.Get("rms_norm");
  auto add_node = pattern_map.Get("add");

  if (!rms_norm_node || !add_node) {
    return false;
  }

  // 验证输入类型兼容性
  if (!CheckInputTypesCompatible(rms_norm_node, add_node)) {
    return false;
  }

  // 验证 Add 输入形状兼容性
  if (!CheckAddInputShapesCompatible(add_node)) {
    return false;
  }

  return true;
}
```

#### Python 应用集成

`ms_custom_ops` 提供了简洁的 Python API 用于注册自定义 Pass：

```python
import mindspore as ms
from mindspore import nn, Tensor
import ms_custom_ops
import numpy as np

# 注册自定义 Pass（自动查找插件路径）
ms_custom_ops.register_custom_pass("AddRmsNormFusionPass", backend="ascend")
ms_custom_ops.register_custom_pass("ReplaceAddNFusionPass", backend="cpu")

# 定义网络
class MyNetwork(nn.Cell):
    def __init__(self):
        super().__init__()
        self.dense = nn.Dense(10, 10)

    def construct(self, x):
        return self.dense(x)

# 设置运行环境
ms.set_context(device_target="Ascend")

# 创建模型和数据
model = MyNetwork()
data = Tensor(np.random.rand(32, 10).astype(np.float32))

# 模型推理 - Pass会在编译阶段自动应用
output = model(data)
print(f"Output shape: {output.shape}")
```

#### API参数说明

- **pass_name**: Pass名称，必须与插件中注册的Pass名称匹配
- **backend**: 目标后端设备类型，用于设备特定优化（默认: "ascend"）
