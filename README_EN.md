# MindSpore Custom Operators Extension (ms_custom_ops)

[Chinese Version](README.md) | English

## Introduction

MindSpore Custom Operators Extension (ms_custom_ops) is an independent operator extension plugin package built upon MindSpore's native custom operator and custom pass capabilities. Its core positioning is to provide key high-performance operator support for cutting-edge AI domain models such as large models, helping models break through computational efficiency bottlenecks; in terms of operator integration, this plugin package has extremely high flexibility, comprehensively covering various types of operator technologies including third-party operator libraries, AscendC operators, Triton DSL, and other technical approaches, fully meeting operator integration requirements in different development scenarios, and providing underlying support for efficient iteration of AI models.

<div align="center">
  <img src="docs/arch.png" alt="Description" width="800" />
</div>

## Directory Structure

```
ms_custom_ops/
├── CMakeLists.txt        # CMake build configuration
├── README.md             # Project documentation
├── OWNERS                # Project maintainers
├── requirements.txt      # Python dependencies
├── setup.py              # Python package configuration
├── version.txt           # Version information
├── 3rdparty/             # Third-party dependencies
├── cmake/                # CMake build scripts
├── ops/                  # Custom operator kernel source code and integration code
│   ├── ascendc/          # AscendC operator implementation and integration code
│   ├── c_api/            # Operators integrated via pre-packaged API calls
│   ├── framework/        # Operator integration common code
│   └── dsl/              # DSL (Domain Specific Language) operator source code
├── pass/                 # Custom fusion passes
├── prebuild/             # Pre-compiled binary libraries
├── python/               # Python bindings and extensions
├── scripts/              # Build and utility scripts
└── tests/                # Test cases
```

## Quick Start

### Prerequisites

- **Python**: >= 3.9
- **MindSpore**: >= 2.7.1
- **Huawei Ascend Software**: CANN toolkit >= 8.3.RC1
- **Compiler**: GCC 7.3 or later
- **CMake**: 3.16 or later
- **Ninja**: 1.11 or later

### Environment Setup

1. **Install Huawei Ascend CANN toolkit**:  
   Download and install the CANN toolkit from the [Huawei Ascend official website](https://www.hiascend.com/developer/download/community/result?module=cann)

2. **Set Ascend environment**:
   ```bash
   export ASCEND_HOME_PATH=${YOUR_INSTALL_PATH}$/ascend-toolkit/latest
   source ${ASCEND_HOME_PATH}/../set_env.sh
   ```

3. **Install MindSpore**:  
   Download and install from [MindSpore official website](https://www.mindspore.cn/install)

### Build and Installation

1. **Clone the repository**:
   ```bash
   git clone https://gitee.com/mindspore/ms-custom-ops.git
   cd ms_custom_ops
   ```

2. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Use the build.sh script (recommended)**:

   ```bash
   # View build options
   bash build.sh -h
   
   # Default build (Release mode)
   bash build.sh
   
   # Debug build
   bash build.sh -d
   
   # Build specified operators
   bash build.sh -p ${absolute_op_dir_path}
   
   # Build specified operators
   bash build.sh -p ${absolute_op_dir_path}
   eg. bash build.sh -p /home/ms_custom_ops/ccsrc/ops/ascendc/add,/home/ms_custom_ops/ccsrc/ops/ascendc/add
   
   # Build with specified SOC Version
   eg. bash build.sh -v ascend910b4
   ```

4. **Install using setup.py**

   ```bash
   # Install (automatically compiles custom operators)
   python setup.py install
   
   # Or build wheel package
   python setup.py bdist_wheel
   ```

## Basic Usage

   After installation, you can use the custom operations in your MindSpore code:

   ```python
   import mindspore as ms
   import ms_custom_ops
   
   # Example usage of a custom operation (actual API may vary)
   # result = ms_custom_ops.some_custom_operation(input_tensor)
   ```

## Reference Documentation
- [MindSpore Tutorials](https://www.mindspore.cn/tutorials/en/r2.7.0/index.html)
- [MindSpore Custom Programming](https://www.mindspore.cn/tutorials/en/r2.7.0/custom_program/op_custom.html)
- [AscendC Programming](https://www.hiascend.com/cann/ascend-c)

## Contributing

We welcome contributions to this project. Please see the [CONTRIBUTING.md](https://www.mindspore.cn/vllm_mindspore/docs/en/master/developer_guide/contributing.html) file for guidelines on how to contribute.
We welcome and value any form of contribution and collaboration. Please inform us via Issue of any bugs you encounter, or submit your feature requests, improvement suggestions, and technical proposals.