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
# pylint: skip-file
"""Fused Add TopK Div MoE kernel implementation using AscendC."""
import os
import sys
import shutil
from pathlib import Path

import numpy as np
from swft.api import *
from swft.core import *
from swft.runtime import *

parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))
batch_size = 2048
dim = 128
inf = 65500
k = 8
CORE_NUM = 8
OP_NAME = 'fused_add_topk_div_moe'
os.system(f"mkdir -p temp/{OP_NAME}")
os.system(f"mkdir -p temp/{OP_NAME}/input")
os.system(f"mkdir -p temp/{OP_NAME}/output")

# Numpy Test
# ===============================================================================
def softmax(x, axis=-1):
    x_exp = np.exp(x)
    return x_exp / np.sum(x_exp, axis=axis, keepdims=True)


class Model:
    """Model class for fused add topk div moe operation."""
    def __init__(self):
        """Initialize the model."""

    def before_fused(self, logits, bias):
        """Process logits and bias before fusion."""
        sigmoid_out = softmax(logits)
        add_out = sigmoid_out + bias
        idx = np.argpartition(add_out, -k, -1)
        topk_idx = np.take(idx, range(-k, 0), -1)
        topk_scores = np.take_along_axis(add_out, topk_idx, -1)
        sorted_indices = np.argsort(-topk_scores, axis=-1)
        sorted_topk_scores = np.take_along_axis(
            topk_scores, sorted_indices, -1)
        sorted_topk_idx = np.take_along_axis(topk_idx, sorted_indices, -1)
        sum_out = np.sum(sorted_topk_scores, axis=-1, keepdims=True)
        div_out = sorted_topk_scores / sum_out
        sorted_topk_idx = sorted_topk_idx.astype(np.int32)
        return div_out, sorted_topk_idx

    def __call__(self, logits, bias):
        """Call the model with logits and bias."""
        expert_weight, expert_index = self.before_fused(logits, bias)
        return expert_weight, expert_index

def gen_data():
    """Generate test data for the kernel."""
    logits = np.random.rand(batch_size, dim).astype(np.float32)
    bias = np.zeros(dim, dtype=np.float32)
    model = Model()
    expert_weight, expert_index = model(
        logits, bias
    )
    num_tokens = np.array([batch_size], dtype=np.int32)
    logits.tofile(f"./temp/{OP_NAME}/input/logits.bin")
    bias.tofile(f"./temp/{OP_NAME}/input/bias.bin")
    num_tokens.tofile(f"./temp/{OP_NAME}/input/num_tokens.bin")
    expert_weight.tofile(f"./temp/{OP_NAME}/output/expert_weight_golden.bin")
    expert_index.tofile(f"./temp/{OP_NAME}/output/expert_index_golden.bin")

def part_sort_128(ub_x, ub_indices):
    """Perform partial sort on 128 elements."""
    concat_local = vconcat(ub_x, ub_indices)
    sorted_local = vsort16(concat_local)
    sort_tmp_local_lst = []
    for i in range(2):
        sorted_0 = slice_to_ub(sorted_local, [512 * i], [128])
        sorted_1 = slice_to_ub(sorted_local, [512 * i + 128], [128])
        sorted_2 = slice_to_ub(sorted_local, [512 * i + 256], [128])
        sorted_3 = slice_to_ub(sorted_local, [512 * i + 384], [128])
        sort_tmp_local = vmrgsort4(
            sorted_0, sorted_1, sorted_2, sorted_3, 16, 16, 16, 16)
        sort_tmp_local_lst.append(sort_tmp_local)
    sort_tmp_local = vmrgsort4(sort_tmp_local_lst[0], sort_tmp_local_lst[1],
                               None, None, 32, 32, 32, 32, rep=1)
    ub_sorted, ub_index = vextract(sort_tmp_local)
    return ub_sorted, ub_index

@sub_kernel(core_num=CORE_NUM)
def fused_add_topk_div_moe(
        logits, bias, num_tokens, expert_weight, expert_index):
    """Fused add topk div moe kernel implementation."""
    block_idx = get_block_idx()
    percore_size = ((num_tokens + CORE_NUM - 1) // CORE_NUM).copy()
    percore_size = (percore_size + 7) // 8 * 8
    token_num = Scalar("INT32", 0)
    if (block_idx + 1) * percore_size < num_tokens:
        token_num.load(percore_size)
    elif block_idx * percore_size < num_tokens:
        token_num.load(num_tokens - block_idx * percore_size)
    else:
        token_num.load(Scalar("INT32", 0))
    current_extra_score = slice_to_ub(bias, [0], slicesize=[dim])
    for i in dynamic_loop(token_num):
        current_score = slice_to_ub(logits, [block_idx * percore_size + i, 0], slicesize=[1, dim])
        current_score = vexp(current_score)
        sum_exp = vcadd(current_score, reduce_axis=-1)
        sum_exp = move_to_scalar(sum_exp)
        current_score = vdivs(current_score, sum_exp)
        current_score = change_view(current_score, new_shape=[dim])
        current_score = vadd(current_score, current_extra_score)
        ub_indices = arange(0, 128, dtype="INT32")
        ub_indices = vconv(ub_indices, "FP32")
        current_score, ub_indices = part_sort_128(current_score, ub_indices)
        ub_indices = vconv(ub_indices, "INT32", "a")
        topk_vals = slice_to_ub(current_score, [0], slicesize=[k])
        topk_indices = slice_to_ub(ub_indices, [0], slicesize=[k])
        sum_val = vcadd(topk_vals, reduce_axis=-1)
        sum_val = move_to_scalar(sum_val)
        topk_vals = vdivs(topk_vals, sum_val)

        insert_to_gm(expert_weight, topk_vals, [block_idx * percore_size + i, 0],
                     slicesize=[1, k], no_autopad=True)
        insert_to_gm(expert_index, topk_indices, [block_idx * percore_size + i, 0],
                     slicesize=[1, k], no_autopad=True)

if __name__ == '__main__':
    set_context("310P")
    gen_data()
    logits = Tensor("GM", "FP32", [batch_size, dim],
                    format="ND", multi_core=False)
    bias = Tensor("GM", "FP32", [dim], format="ND", multi_core=False)
    expert_weight = Tensor(
        "GM", "FP32", [batch_size, k], format="ND", multi_core=False)
    expert_index = Tensor(
        "GM", "INT32", [batch_size, k], format="ND", multi_core=False)
    num_tokens = Scalar("INT32")
    compile_func(fused_add_topk_div_moe, globals())(
        logits, bias, num_tokens, expert_weight, expert_index)
    compile_kernel(f"./temp/{OP_NAME}/{OP_NAME}.cce", OP_NAME)
    exec_kernel(OP_NAME, locals(), prefix_path="temp", inputs=[
                'logits', 'bias', 'num_tokens'],
                outputs=['expert_weight', 'expert_index'], device_id=1)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = f'./temp/{OP_NAME}/input'
    output_dir = f'./temp/{OP_NAME}/output'

    # Verify expert_index
    return_code1 = os.system(
        f'python3 {script_dir}/../verify_result.py '
        f'./temp/{OP_NAME}/output/expert_index_actual.bin '
        f'./temp/{OP_NAME}/output/expert_index_golden.bin int32 4e-2 1e-2 4e-3')
    # Verify expert_weight
    return_code2 = os.system(
        f'python3 {script_dir}/../verify_result.py '
        f'./temp/{OP_NAME}/output/expert_weight_actual.bin '
        f'./temp/{OP_NAME}/output/expert_weight_golden.bin float32 4e-2 1e-2 4e-3')

    # Check if both verifications passed
    exit_code1 = return_code1 >> 8
    exit_code2 = return_code2 >> 8
    all_passed = (exit_code1 == 0) and (exit_code2 == 0)

    # Clean up directories if all verifications passed
    if all_passed:
        try:
            if os.path.exists(input_dir):
                shutil.rmtree(input_dir)
            if os.path.exists(output_dir):
                shutil.rmtree(output_dir)
        except Exception as e:
            print(f"Warning: Failed to clean up directories: {e}")

    # Exit with error code if any verification failed
    if exit_code1 != 0:
        sys.exit(exit_code1)
    sys.exit(exit_code2)
