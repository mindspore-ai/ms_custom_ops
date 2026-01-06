# ms_custom_ops算子开发指导

## 1 整体说明

本文主要描述通过ms_custom_ops将算子接入到MindSpore的基本方法和规则。<br>
根据需要接入的算子实现不同，分为以下几种类型：

- ascendc：以CANN AscendC标准自定义算子方式实现的kernel，会生成aclnn两段式接口。此类算子需要开发kenrel逻辑。
- c_api：由第三方完成kernel和host API开发，在ms_custom_ops中只是完成接入处理。此类算子不需要开发kernel逻辑，只需要链接第三方提供的二进制库。注意：通过submodule方式引入其他开源组件并基于原有工程编译出二进制库使用的也属于此类。
- dsl：Domain Specific Language。基于其他算子编译器语言编写的算子，如triton等。<br>

下文主要说明如何在ms_custom_ops中开发上述3中类型的算子。

## 2 术语解释

### 2.1 值依赖

一个完整的算子包含**Host计算**和**Device计算**两部分。其中**Device计算**就是用AscendC/CCE编写的代码，而**Host计算**则包括了InferShape/InferType/Tiling等计算过程。如果Host计算过程依赖算子输入的值，则称为值依赖算子。典型算子举例：

- Reshape：`mindspore.ops.reshape(input, shape)`，该算子InferShape计算需要读取参数`shape`的值，属于**InferShape值依赖**。在接口层面，参数`shape`的类型是`Tuple/List`，属于非Tensor输入。
- mla: `ms_custom_ops.mla`的Tiling计算需要读取`context_lens`和`q_seq_lens`的值，这两个参数是Tensor类型，这种场景称为**Tiling值依赖**。

## 3 ascendc类型算子实现

以`apply_rotary_pos_emb_ms`为例。

### 3.1 目录结构

```text
ops/ascendc/
├── aclnn_src/                     # AscendC实现Kernel相关代码
│   └── apply_rotary_pos_emb_ms/
│       ├── op_host/               # 算子逻辑host侧实现，包括类型注册、InferShape、Tiling等实现代码，需要分成3个文件分别存放，同时在CMakeList.txt中设置编译过程。
│       │   ├── apply_rotary_pos_emb_ms_def.cpp
│       │   ├── apply_rotary_pos_emb_ms_proto.cpp
│       │   └── apply_rotary_pos_emb_ms_tiling.cpp
│       ├── op_kernel/             # 算子逻辑kernel侧实现
│       └── CMakeLists.txt
└── ms_glue/                                 # MindSpore对接相关代码
    └── apply_rotary_pos_emb_ms
        ├── apply_rotary_pos_emb_ms_op.yaml  # 算子在MindSpore侧原型定义
        ├── apply_rotary_pos_emb_ms.md       # 算子通过MindSpore对外提供的接口说明文档
        └── apply_rotary_pos_emb_ms.cc       # 算子在MindSpore中的接入，包括InferShape（与op_host中的InferShape逻辑相同，但是实现接口不一致）、静态图
                                             # KernelMod接入、动态图等代码。动态图和静态图对接代码可以放在同一个文件中也可以分文件存放。
```

### 3.2 kernel开发

包括op_host和op_kernel目录下代码，具体参照[AscendC编程](https://www.hiascend.com/cann/ascend-c)。如果算子编译Device代码需要特殊编译选项，可以在aclnn_src/`YOUR_OP_DIR`/CMakeLists.txt中添加，例如：

```text
add_ops_compile_options(
        OP_NAME ApplyRotaryPosEmbMS
        OPTIONS --cce-auto-sync=off
                -Wno-deprecated-declarations
                -Werror
)
```

> **Note**：op_kernel下非头文件必须以`.cpp`结尾。

### 3.3 MindSpore静态图接入

#### 3.3.1 算子原型定义

各字段含义参考：[MindSpore算子yaml说明](https://gitee.com/mindspore/mindspore/blob/master/mindspore/ops/op_def/yaml/README.md)。

#### 3.3.2 Infer实现

继承类`OpFuncImpl`，一般只要重写`InferShape`、`InferType`和`GeneralInferRegistered`方法。其中`GeneralInferRegistered`方法固定返回`true`。
`ApplyRotaryPosEmbMSOpFuncImpl`类名需要满足规则：`op_name` + `OpFuncImpl`，`op_name`应与yaml文件中原型定义的算子名称保持一致。

```c++
class OPS_API ApplyRotaryPosEmbMSOpFuncImpl : public OpFuncImpl {
 public:
  ShapeArray InferShape(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const override {
    ...
    return {query_shape, key_shape};
  }

  std::vector<TypeId> InferType(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const override {
    ...
    return {query_dtype, key_dtype};
  }

  bool GeneralInferRegistered() const override { return true; }
};
```

> **Note**：
> - 如果算子存在**值依赖**，并且存在**Tensor输入值依赖**，则需要重写`GetValueDependArgIndices`方法，并且返回所有依赖的参数的索引，包括**Tensor输入**和**非Tensor输入**
> - 如果算子只存在**非Tensor输入值依赖**，则框架可以自动识别，不需要重写`GetValueDependArgIndices`方法。

#### 3.3.3 KernelMod实现

继承类`AclnnCustomKernelMod`。需要实现构造函数，并重写`Launch`和`GetWorkSpaceInfo`函数。

```c++
class ApplyRotaryPosEmbMSAscend : public AclnnCustomKernelMod {
 public:
  ApplyRotaryPosEmbMSAscend() : AclnnCustomKernelMod(std::move("aclnnApplyRotaryPosEmbMS")) {}
  ~ApplyRotaryPosEmbMSAscend() = default;

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs, void *stream_ptr) override {
    ...
    RunOp(
      stream_ptr, workspace, inputs[static_cast<size_t>(ApplyRotaryPosEmbMSInputIndex::kApplyRotaryPosEmbMSQueryIndex)],
      inputs[static_cast<size_t>(ApplyRotaryPosEmbMSInputIndex::kApplyRotaryPosEmbMSKeyIndex)],
      inputs[static_cast<size_t>(ApplyRotaryPosEmbMSInputIndex::kApplyRotaryPosEmbMSCosIndex)],
      inputs[static_cast<size_t>(ApplyRotaryPosEmbMSInputIndex::kApplyRotaryPosEmbMSSinIndex)], layout_, rotary_mode_);
    return true;
  }

  void GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                        const std::vector<KernelTensor *> &outputs) override {
    ...
    GetWorkspaceForResize(inputs[static_cast<size_t>(ApplyRotaryPosEmbMSInputIndex::kApplyRotaryPosEmbMSQueryIndex)],
                          inputs[static_cast<size_t>(ApplyRotaryPosEmbMSInputIndex::kApplyRotaryPosEmbMSKeyIndex)],
                          inputs[static_cast<size_t>(ApplyRotaryPosEmbMSInputIndex::kApplyRotaryPosEmbMSCosIndex)],
                          inputs[static_cast<size_t>(ApplyRotaryPosEmbMSInputIndex::kApplyRotaryPosEmbMSSinIndex)],
                          layout_, rotary_mode_);
  }

 private:
  DEFINE_GET_WORKSPACE_FOR_RESIZE();
  ...
};
```

#### 3.3.4 注册接口与绑定KernelMod

**REG_GRAPH_MODE_OP(***python接口名称***, ***Infer实现类***, ***KernelMod类***)**

```c++
REG_GRAPH_MODE_OP(apply_rotary_pos_emb_ms, ms_custom_ops::ApplyRotaryPosEmbMSOpFuncImpl,
                  ms_custom_ops::ApplyRotaryPosEmbMSAscend);
```

### 3.4 MindSpore动态图接入

#### 3.4.1 实现C++侧调用函数

```c++
std::vector<ms::Tensor> apply_rotary_pos_emb_ms_custom(const ms::Tensor &query, const ms::Tensor &key,
                                                       const ms::Tensor &cos, const ms::Tensor &sin,
                                                       const std::string layout_str, const std::string rotary_mode) {
  std::string op_name = "ApplyRotaryPosEmbMS";
  // 1). 创建runner，AscendC算子采用aclnn两段式接口，所以只需要继承预定义的`ms::pynative::AclnnOpRunner`类即可
  auto runner = std::make_shared<ms::pynative::AclnnOpRunner>(op_name);
  // 输入shape检查
  ApplyRotaryPosEmbMSCheckInputsShape(op_name, query.shape(), key.shape(), cos.shape(), sin.shape());
  // 输入dtype检查
  ApplyRotaryPosEmbMSCheckInputsType(op_name, query.data_type(), key.data_type(), cos.data_type(), sin.data_type());

  // 2). 推导输出Tensor，包括shape和dtype信息。`apply_rotary_pos_emb_ms`属于原地更新算子，不需要推导输出

  // 3). 设置launch Function
  // 此处"aclnnApplyRotaryPosEmbMS", 是算子库函数表中名字前面加上aclnn
  // 可通过 nm -D ./build/xxx/xxx/ms_custom_ops.xxx.so | grep "ApplyRotaryPosEmbMS" 来确认
  // 如果是覆写算子（inplace），不必添加输出参数
  runner->SetLaunchFunc(LAUNCH_ACLNN_FUNC(aclnnApplyRotaryPosEmbMS, query, key, cos, sin, layout_str, rotary_mode));
  // 4). 执行算子。如果是覆写算子（inplace），输出参数为空
  runner->Run({query, key, cos, sin}, {});
  // 5). 返回输出
  return {query, key};
}
```

#### 3.4.2 绑定python接口

```c++
MS_CUSTOM_OPS_EXTENSION_MODULE(m) {
  m.def("apply_rotary_pos_emb_ms",  // python接口名称
        &pyboost_apply_rotary_pos_emb_ms,  // 绑定到python的c++接口
        "ApplyRotaryPosEmbMS",  // 算子描述
        pybind11::arg("query"), // 以下为参数
        pybind11::arg("key"), pybind11::arg("cos"), pybind11::arg("sin"), pybind11::arg("layout"),
        pybind11::arg("rotary_mode"));
}
```

### 3.5 编写算子文档

参考[`apply_rotary_pos_emb_ms.md`](../ops/ascendc/apply_rotary_pos_emb_ms/apply_rotary_pos_emb_ms.md)。新增算子后需要同步更新[`op_list.md`](op_list.md)，调用脚本可以自动生成：

``` python
python scripts/generate_op_list.py
```

## 4 c_api类型算子实现

此类型算子不需要实现kernel，只需要开发MindSpore接入代码即可。为了获得最优的执行性能，需要将第三方的API以合理的方式嵌入到MindSpore静态图KernelMod的`Init`、`Resize`、`Launch`函数中，一般需满足以下指导原则：

- `Init`函数每个KernelMod实例的生命周期中指挥调用一次。因此一个进程中，算子实例创建后就固定的信息在`Init`函数中处理，如环境判断、MindSpore接口与下层接口输入映射等；
- `Resize`函数会在每次shape发生变化时调用，主要处理与Shape强相关的逻辑，包括输出大小计算、tiling计算、tiling cache操作等；
- `Launch`函数每次都会执行，要最小化Launch相关处理逻辑。<br>

动态图由于不存在图编译过程，主要通过cache机制来减少算子重复调用的开销。<br>
参考：[`internal_kernel_mod.cc`](../ops/framework/ms_kernels_internal/graphmode/internal_kernel_mod.cc)、[`internal_pyboost_runner.cc`](../ops/framework/ms_kernels_internal/pyboost/internal_pyboost_runner.cc)。<br>
所有类型算子的MindSpore接入中算子原型定义、Infer实现、注册接口与绑定KernelMod都与`3.3.1`、`3.3.2`、`3.3.4`一致，不再重复描述。<br>
`c_api`类型的算子实现文件基本相同，以[`mla`](../ops/c_api/mla)为例：

```test
mla/
├── mla_op.yaml        # 算子在MindSpore侧原型定义
├── mla_doc.md         # 算子通过MindSpore对外提供的接口说明文档
├── mla_graph.cc       # MindSpore静态图模式算子接入代码
├── mla_pynative.cc    # MindSpore动态图模式算子接入代码
└── mla_common.h       # 算子公共头文件
```

### 4.1 ms_kernels_internal算子接入

以[`mla`](../ops/c_api/mla)为例。

#### 4.1.1 MindSpore静态图接入KernelMod实现

继承类`InternalKernelMod`。必须重写`CreateKernel`和`InitKernelInputsOutputsIndex`方法，按需重写`UpdateParam`、`IsNeedRecreate`和`GenerateTilingKey`方法。

```c++
class Mla : public InternalKernelMod {
 public:
  Mla() : InternalKernelMod() {}
  ~Mla() = default;

 protected:
  internal_v2::InternalOpPtr CreateKernel(const internal_v2::InputsImmutableInfoList &inputs,
                                          const internal_v2::OutputsImmutableInfoList &outputs,
                                          const std::vector<KernelTensor *> &ms_inputs,
                                          const std::vector<KernelTensor *> &ms_outputs) override {
    ...
    return internal_v2::CreateMLAOp(inputs, outputs, param_, internal_v2::kInternalMLAOpName);
  }

  bool UpdateParam(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override {
    ...
    return true;
  }

  uint64_t GenerateTilingKey(const std::vector<KernelTensor *> &inputs) override {
    // User defined CacheKey, the inputs should include all the factors which
    // will affect tiling result.
    return InternalTilingCache::GenerateKey(kernel_name_, inputs, param_.q_seq_len, param_.kv_seq_len);
  }

  void InitKernelInputsOutputsIndex() override {
    kernel_inputs_index_ = {kMlaInputQnopeIndex,      kMlaInputQropeIndex,       kMlaInputKvCacheIndex,
                            kMlaInputKropeIndex,      kMlaInputBlockTablesIndex, kMlaInputAttnMaskIndex,
                            kMlaInputDeqScaleQkIndex, kMlaInputDeqScalePvIndex};
    kernel_outputs_index_ = {0, 1};
  }
  ...
};
```

> **Note**：在`InternalKernelMod`的实现中，已经考虑了**Tensor输入的shape、非Tensor输入的值**对`InternalOp`创建、Tiling操作的影响。如果遇到一下情况则需要重写对应函数：
> - 一个进程中自从KernelMod实例化后，**Tensor输入的值变化导致需要重新创建`InternalOp`的**，需要重写`IsNeedRecreate`方法。通常情况下不涉及。
> - 一个进程中自从KernelMod实例化后，**Tensor输入的值变化导致需要重新Tiling的**，则需要重写`GenerateTilingKey`方法，例如`q_seq_len`和`kv_seq_len`会影响`mla`的tiling过程。如果还存在其他特殊情况需要重新tiling的，也要重写。
> - 需要将**Tensor输入的值设置到`InternalOp`的`Param`中**，则需要重写`UpdateParam`方法。例如`mla`的`q_seq_len`和`kv_seq_len`保存在`internal_v2::MLAParam`中。

#### 4.1.2 MindSpore动态图接入

##### 4.1.2.1 实现Runner

继承类`InternalPyboostRunner`。实现构造函数和自定义Param设置函数，必须重写`CreateKernel`方法，按需重写`UpdateParam`方法（触发条件与静态图一致）。

```c++
class MlaRunner : public InternalPyboostRunner {
 public:
  explicit MlaRunner(const std::string &op_name) : InternalPyboostRunner(op_name) {}
  ~MlaRunner() = default;

  void SetParam(int32_t head_size, float tor, int32_t kv_head, mindspore::internal_v2::MLAParam::MaskType mask_type,
                int32_t is_ring, const std::vector<int32_t> &q_seq_len, const std::vector<int32_t> &kv_seq_len) {
    ...
  }

  void SetInputFormat(MlaInputFormat input_format) { input_format_ = input_format; }

 protected:
  bool UpdateParam() override {
    ...
    return true;
  }

  internal_v2::InternalOpPtr CreateKernel(const internal_v2::InputsImmutableInfoList &inputs,
                                          const internal_v2::OutputsImmutableInfoList &outputs) override {
    ...
    return mindspore::internal_v2::CreateMLAOp(inputs, outputs, param_, internal_v2::kInternalMLAOpName);
  }

  ...
};
```

##### 4.1.2.2 实现C++侧调用函数

```c++
std::vector<ms::Tensor> mla_atb(const ms::Tensor &q_nope, const ms::Tensor &q_rope, const ms::Tensor &ctkv,
                                const ms::Tensor &k_rope, const ms::Tensor &block_tables,
                                const std::optional<ms::Tensor> &attn_mask,
                                const std::optional<ms::Tensor> &deq_scale_qk,
                                const std::optional<ms::Tensor> &deq_scale_pv,
                                const std::optional<ms::Tensor> &q_seq_lens,
                                const std::optional<ms::Tensor> &context_lens, int64_t head_num, double scale_value,
                                int64_t kv_head_num, int64_t mask_type, int64_t input_format, int64_t is_ring) {
  static auto op_name = "Mla";
  // 1). 实例化上一步骤中的Runner
  auto runner = std::make_shared<MlaRunner>(op_name);
  ...

  // 2). 设置param
  runner->SetParam(static_cast<int32_t>(head_num), static_cast<float>(scale_value), static_cast<int32_t>(kv_head_num),
                   static_cast<mindspore::internal_v2::MLAParam::MaskType>(mask_type), static_cast<int32_t>(is_ring),
                   q_seq_lens_value, context_lens_value);

  ...
  runner->SetInputFormat(static_cast<MlaInputFormat>(input_format));

  // 3). 调用Setup：
  runner->Setup(op_name, q_nope, q_rope, ctkv, k_rope, block_tables, attn_mask, deq_scale_qk, deq_scale_pv, q_seq_lens,
                context_lens, head_num, scale_value, kv_head_num, mask_type, input_format, is_ring);

  // 4). 推导输出Tensor，包括shape和dtype信息:
  auto attn_out = ms::Tensor(q_nope.data_type(), q_nope.shape());
  auto lse_out = ms::Tensor(q_nope.data_type(), {0});

  ...
  // 5). 创建`InternalOp`：
  runner->GetOrCreateKernel(inputs, outputs);
  // 6). 执行算子：
  runner->Run(inputs, outputs);
  // 7). 返回输出：
  return outputs;
}

auto pyboost_mla(const ms::Tensor &q_nope, const ms::Tensor &q_rope, const ms::Tensor &ctkv, const ms::Tensor &k_rope,
                 const ms::Tensor &block_tables, const std::optional<ms::Tensor> &attn_mask,
                 const std::optional<ms::Tensor> &deq_scale_qk, const std::optional<ms::Tensor> &deq_scale_pv,
                 const std::optional<ms::Tensor> &q_seq_lens, const std::optional<ms::Tensor> &context_lens,
                 int64_t head_num, double scale_value, int64_t kv_head_num, int64_t mask_type, int64_t input_format,
                 int64_t is_ring) {
  // 2表示算子的返回值个数
  return ms::pynative::PyboostRunner::Call<2>(mla_atb, q_nope, q_rope, ctkv, k_rope, block_tables, attn_mask,
                                              deq_scale_qk, deq_scale_pv, q_seq_lens, context_lens, head_num,
                                              scale_value, kv_head_num, mask_type, input_format, is_ring);
}
```

##### 4.1.2.3 绑定python接口

```c++
MS_CUSTOM_OPS_EXTENSION_MODULE(m) {
  m.def("mla",  // python接口名称
        &ms_custom_ops::pyboost_mla,  // 绑定到python的c++接口
        "Multi-head Latent Attention", // 算子说明
        pybind11::arg("q_nope"),  // 以下为参数，'='后面的为默认值
        pybind11::arg("q_rope"), pybind11::arg("ctkv"), pybind11::arg("k_rope"), pybind11::arg("block_tables"),
        pybind11::arg("attn_mask") = std::nullopt, pybind11::arg("deq_scale_qk") = std::nullopt,
        pybind11::arg("deq_scale_pv") = std::nullopt, pybind11::arg("q_seq_lens") = std::nullopt,
        pybind11::arg("context_lens") = std::nullopt, pybind11::arg("head_num") = 32,
        pybind11::arg("scale_value") = 0.0, pybind11::arg("kv_head_num") = 1, pybind11::arg("mask_type") = 0,
        pybind11::arg("input_format") = 0, pybind11::arg("is_ring") = 0);
}
```

### 4.2 aclnn算子接入

MindSpore接入代码与ascendc类型算子一致，见`3.3`和`3.4`。

## 5 dsl类型算子

TODO。

## 6 其他原则

- 动态图和静态图实现规格需要保持一致，包括接口名称、参数顺序、参数规格校验等。特别注意：静态图输入参数的默认值在yaml中指定，动态图输入参数的默认值在`MS_CUSTOM_OPS_EXTENSION_MODULE`中指定，而`PYBOOST_CALLER`是不能设置默认值的，使用时需注意一致性。

## 7 其他参考文档

- [MindSpore自定义编程](https://www.mindspore.cn/tutorials/zh-CN/r2.7.0/custom_program/op_custom.html)