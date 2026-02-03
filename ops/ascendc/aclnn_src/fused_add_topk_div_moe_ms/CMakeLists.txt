# Copyright 2025 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

set(OP_PREFIX fused_add_topk_div_moe_ms)
# custom op_kernel compile options if necessary
# add_ops_compile_options(
#         OP_NAME your_kernel_name  # The name is identical with the class name defined in op_host/xxx_def.cpp
#         OPTIONS --cce-auto-sync=off
#                 -Wno-deprecated-declarations
#                 -Werror
# )

target_sources(op_host_aclnn PRIVATE
        op_host/${OP_PREFIX}_def.cpp
)

target_sources(optiling PRIVATE
        op_host/${OP_PREFIX}_tiling.cpp
)

target_sources(opsproto PRIVATE
        op_host/${OP_PREFIX}_proto.cpp
)
