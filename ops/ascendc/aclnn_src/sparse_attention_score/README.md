# Custom Sparse Attention Operator

本目录基于华为 Ascend 社区的开源项目 **CANN Ops Adv** 进行二次开发与功能扩展。

## 1. 开源来源说明

* **上游仓库**: [https://gitee.com/ascend/cann-ops-adv](https://gitee.com/ascend/cann-ops-adv)

* **基础算子**: 主要基于原仓库中的 `FlashAttention` 算子逻辑进行修改和适配。

## 2. 主要功能更新与改动

重点新增了稀疏注意力机制的支持，并对底层计算逻辑进行了多项优化。以下是详细的改动列表：

### 2.1 新增核心算子

* **Sparse Attention**: 基于 `FlashAttention` (`fa`) 逻辑派生出 `sparse_attention` (`sa`) 算子，支持更灵活的稀疏计算模式。

### 2.2 核心逻辑实现 (Sparse Logic)

* **索引选择机制 (`select_idx`)**:
    * 引入并实现了 `select_idx` 机制，用于支持非连续、稀疏的 Block 注意力计算。
    * 在 API 层面打通了 `selectIdx` 的传参支持。
    * 实现了基于 `vertical-slash` sparse mode 的 Block 生成逻辑。

* **代码迁移与重构**:
    * 完成从 `fa` 到 `sa` 的代码解耦与重命名。
    * 移除部分场景下的 L1 复用逻辑以适配稀疏计算特性。

### 2.3 性能优化与配置适配

* **Tiling 策略优化**: 引入 `balance tiling` 策略，优化切块逻辑以提升并行计算效率。

* **多尺寸 Block 支持**: 适配并验证了多种 Block Size 配置，包括但不限于：
    * 64 x 64
    * 128 x 128
    * 128 x 1024