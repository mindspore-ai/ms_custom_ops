# MsCustomOps Pass 列表

| Pass名称 | 功能描述 | 输入模式 | 输出模式 | 默认使能 |
|---------|---------|---------|---------|---------|
| AddRmsNormFusionPass | 将Add和RmsNorm操作融合为单个AddRmsNorm操作，减少计算开销 | RmsNorm(Add(x1, x2), gamma, eps) | AddRmsNorm(x1, x2, gamma, eps) | 否 |
| ConvertTupleInputToDynamicInput | 将算子tuple/list类型输入展开，转换为动态输入 | op_func(var, list[tensor]) | op_func(var, tensor1, tensor2, ...) | 是 |
