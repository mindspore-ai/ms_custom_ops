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
"""Tests for ms_custom_ops.group_topk."""

import os
import time
import tempfile
import signal
import subprocess
import numpy as np
import pytest

import mindspore as ms
from mindspore import nn
from mindspore.common.np_dtype import bfloat16

import ms_custom_ops


np.random.seed(42)
ms.set_device("Ascend")


# Global constants
DYNAMIC_ITER_STEP = 3  # The number of iterations for dynamic shape
IDX_ARR_SIZE = 1024  # Fixed size of idx_arr
DEFAULT_NUM_TOKENS = 512
DEFAULT_EXPERT_NUM = 256
DEFAULT_GROUP_NUM = 8
DEFAULT_K = 8
DEFAULT_K_INNER = 2


class GroupTopkCell(nn.Cell):
    def construct(self, token, idx_arr, group_num, k, k_inner):
        ms_custom_ops.group_topk(token, idx_arr, group_num, k, k_inner)
        return token


def numpy_topk(arr, k, axis=-1):
    sorted_indices = np.argsort(arr, axis=axis)
    if axis < 0:
        axis = arr.ndim + axis
    topk_indices = np.take(sorted_indices, np.arange(-k, 0), axis=axis)
    topk_values = np.take_along_axis(arr, topk_indices, axis=axis)
    return topk_values, topk_indices


def compute_np_expected_result(inputx, token_num, expert_num, group_num, k, k_inner):
    """
    Calculate the result form numpy as a reference result.
    """
    token_input = inputx.reshape((token_num, group_num, expert_num // group_num))
    output = np.copy(token_input)
    token_input = token_input.astype(np.float32)
    group_tensor, _ = numpy_topk(token_input, k_inner)
    group_tensor = np.sum(group_tensor, axis=-1)
    sort_index = np.argsort(-group_tensor, kind='stable')
    cols_to_use = np.arange(k, group_num, dtype=np.int64)
    row_indices = np.repeat(
        np.arange(sort_index.shape[0]), cols_to_use.shape[0])
    col_indices = sort_index[:, cols_to_use].reshape(-1)
    output[row_indices, col_indices] = 0
    return np.reshape(output, (token_num, expert_num))


def get_ms_dtype(np_dtype):
    if np_dtype == np.float16:
        ms_dtype = ms.float16
    elif np_dtype == bfloat16:
        ms_dtype = ms.bfloat16
    else:
        ms_dtype = ms.float32
    return ms_dtype


def run_group_topk_common(token_num, expert_num, input_param, token_dtype, net):
    """
    Execute the grouptopk common process.
    """
    input_shape = (token_num, expert_num)
    token_input = np.random.uniform(-2, 2, input_shape).astype(token_dtype)
    index_array = np.arange(IDX_ARR_SIZE, dtype=np.int32)
    token_input_tensor = ms.Tensor(token_input, dtype=get_ms_dtype(token_dtype))
    index_array_tensor = ms.Tensor(index_array, dtype=ms.int32)

    ms_out = net(token_input_tensor, index_array_tensor, *input_param)
    np_out = compute_np_expected_result(token_input, token_num, expert_num, *input_param)

    np.testing.assert_allclose(
        ms_out.astype(ms.float32).asnumpy(), np_out.astype(np.float32), rtol=1e-2, atol=1e-2)


def run_group_topk_dynamic(token_num, expert_num, input_param, token_dtype, net):
    token_input_dyn = ms.Tensor(shape=[None, None], dtype=get_ms_dtype(token_dtype))
    index_array_dyn = ms.Tensor(shape=[None], dtype=ms.int32)
    net.set_inputs(token_input_dyn, index_array_dyn, *input_param)

    for item in range(DYNAMIC_ITER_STEP):
        token_num = token_num + item
        run_group_topk_common(token_num, expert_num, input_param, token_dtype, net)


@pytest.mark.level0
@pytest.mark.platform_ascend910b
@pytest.mark.env_onecard
@pytest.mark.parametrize('token_dtype', [np.float16, bfloat16])
@pytest.mark.parametrize('token_num, expert_num, input_param',
                         [(64, 1024, [8, 3, 2]), (359, 256, [8, 4, 2])])
@pytest.mark.parametrize('is_dynamic', [False, True])
def test_group_topk_graph_mode(token_dtype, token_num, expert_num, input_param, is_dynamic):
    """
    Feature: group_topk st
    Description: under graph mode
        - token_dtype: the dtype of token input
        - input_param: list of [group_num, k, k_inner]
        - is_dynamic: whether to apply dynamic shape
    Expectation: success
    """
    ms.set_context(mode=ms.GRAPH_MODE,
                   jit_config={"jit_level": "O0"})

    net = GroupTopkCell()

    if is_dynamic:
        run_group_topk_dynamic(token_num, expert_num, input_param, token_dtype, net)
    else:
        run_group_topk_common(token_num, expert_num, input_param, token_dtype, net)


@pytest.mark.level0
@pytest.mark.platform_ascend910b
@pytest.mark.env_onecard
@pytest.mark.parametrize('token_dtype', [np.float16, bfloat16])
@pytest.mark.parametrize('token_num, expert_num, input_param',
                         [(7, 1024, [8, 3, 2]), (41, 256, [16, 4, 3]), (512, 256, [8, 8, 2])])
def test_group_topk_pynative_mode(token_dtype, token_num, expert_num, input_param):
    """
    Feature: group_topk st
    Description: under pynative mode
        - token_dtype: the dtype of token input
        - input_param: list of [group_num, k, k_inner]
    Expectation: success
    """
    ms.set_context(mode=ms.PYNATIVE_MODE)
    net = GroupTopkCell()
    run_group_topk_common(token_num, expert_num, input_param, token_dtype, net)


def generate_default_test_inputs():
    token = ms.Tensor(np.random.rand(DEFAULT_NUM_TOKENS, DEFAULT_EXPERT_NUM), ms.float16)
    idx_arr = ms.Tensor(np.arange(IDX_ARR_SIZE, dtype=np.int32))
    group_num = DEFAULT_GROUP_NUM
    k = DEFAULT_K
    k_inner = DEFAULT_K_INNER
    return token, idx_arr, group_num, k, k_inner


@pytest.mark.level0
@pytest.mark.platform_ascend910b
@pytest.mark.env_onecard
def test_group_topk_token_value_3d():
    """
    Feature: group_topk st
    Description: test token value with 3 dimension.
    Expectation: success
    """
    ms.set_context(mode=ms.PYNATIVE_MODE)

    _, idx_arr, group_num, k, k_inner = generate_default_test_inputs()
    token1 = ms.Tensor(np.random.rand(4097, 2, 512), ms.float16)
    expert_num = token1.shape[-1]
    token2 = token1.reshape(-1, expert_num)

    ms_custom_ops.group_topk(token1, idx_arr, group_num, k, k_inner)
    ms_custom_ops.group_topk(token2, idx_arr, group_num, k, k_inner)
    np.testing.assert_allclose(token1.reshape(-1, expert_num).asnumpy(), token2.asnumpy())


@pytest.mark.level0
@pytest.mark.platform_ascend910b
@pytest.mark.env_onecard
def test_group_topk_idx_arr_mismatch_1024():
    """
    Feature: group_topk st
    Description: test idx_arr value smaller than 1024, but greater than expert_num.
    Expectation: success
    """
    ms.set_context(mode=ms.PYNATIVE_MODE)

    token1, idx_arr1, group_num, k, k_inner = generate_default_test_inputs()
    token2 = ms.Tensor(token1.asnumpy(), ms.float16)
    idx_arr2 = ms.Tensor(np.arange(512, dtype=np.int32))

    ms_custom_ops.group_topk(token1, idx_arr1, group_num, k, k_inner)
    ms_custom_ops.group_topk(token2, idx_arr2, group_num, k, k_inner)
    np.testing.assert_allclose(token1.asnumpy(), token2.asnumpy())


@pytest.mark.level0
@pytest.mark.platform_ascend910b
@pytest.mark.env_onecard
@pytest.mark.parametrize('jit_mode', [False, True])
def test_group_topk_default_k_inner(jit_mode):
    """
    Feature: group_topk st
    Description: test default value of k_inner.
    Expectation: success
    """
    ms.set_context(mode=ms.PYNATIVE_MODE)
    token1, idx_arr, group_num, k, _ = generate_default_test_inputs()
    token2 = ms.Tensor(token1.asnumpy(), ms.float16)
    k_inner = 1

    if jit_mode:
        group_topk_func = ms.jit(ms_custom_ops.group_topk)
    else:
        group_topk_func = ms_custom_ops.group_topk

    group_topk_func(token1, idx_arr, group_num, k, k_inner=k_inner)
    group_topk_func(token2, idx_arr, group_num, k)
    np.testing.assert_allclose(token1.asnumpy(), token2.asnumpy())


def get_key_counter_from_log(log_name, key):
    """
    Search for error keywords in the program execution log.
    """
    log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), log_name)
    if not os.path.exists(log_path):
        return 0

    with open(log_path, 'r', encoding='utf-8') as f:
        content = f.read()
    return content.count(key)


def create_temp_script(input_code):
    """
    Create a temporary script file for execution
    """
    with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.py') as temp_script:
        script_path = temp_script.name
        script_code = f"""
import mindspore as ms
import numpy as np
import ms_custom_ops

ms.set_context(mode=ms.PYNATIVE_MODE)
ms.runtime.launch_blocking()

# Global constants
DYNAMIC_ITER_STEP = 3
IDX_ARR_SIZE = 1024
DEFAULT_NUM_TOKENS = 512
DEFAULT_EXPERT_NUM = 256
DEFAULT_GROUP_NUM = 8
DEFAULT_K = 8
DEFAULT_K_INNER = 2

def generate_default_test_inputs():
    token = ms.Tensor(np.random.rand(DEFAULT_NUM_TOKENS, DEFAULT_EXPERT_NUM), ms.float16)
    idx_arr = ms.Tensor(np.arange(IDX_ARR_SIZE, dtype=np.int32))
    group_num = DEFAULT_GROUP_NUM
    k = DEFAULT_K
    k_inner = DEFAULT_K_INNER
    return token, idx_arr, group_num, k, k_inner

token, idx_arr, group_num, k, k_inner = generate_default_test_inputs()
# replace the actual input code for the test case below
{input_code}

try:
    ms_custom_ops.group_topk(token=token, idx_arr=idx_arr, group_num=group_num, k=k, k_inner=k_inner)
except Exception as e:
    print("Exception caught:", str(e))
    raise
        """
        temp_script.write(script_code)
        temp_script.flush()
    return script_path


def execute_script(script_path, log_file_path):
    """
    Execute script and redirect log files to specified directory
    """
    execution_command = f"python {script_path} > {log_file_path} 2>&1"
    process = subprocess.Popen(  # pylint: disable=consider-using-with
        execution_command,
        shell=True,
        executable='/bin/bash',
        stdout=None,
        stderr=None,
        start_new_session=True
    )
    return process


def wait_for_error_in_log(log_file_path, expected_error, max_wait_cycles=20):
    """
    Waiting for a 'Tiling error' to appear in the log, indicating completion of the test case
    and return exception as expected. Then check if it contains expected error messages.
    """
    wait_count = 0
    while wait_count < max_wait_cycles:
        if get_key_counter_from_log(log_file_path, "Tiling error") > 0:
            break
        time.sleep(1)
        wait_count += 1
    else:
        with open(log_file_path, 'r', encoding='utf-8') as f:
            err_log = f.read()
        raise RuntimeError(f"Abnormal information mismatch: {err_log}")

    if get_key_counter_from_log(log_file_path, expected_error) <= 0:
        raise AssertionError(f"Cannot find '{expected_error}' in ERROR information.")


def check_group_topk_exception(input_code, log_file_name, expected_error_message):
    """
    Execute group_topk and verify the corresponding error message.
    """
    log_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), log_file_name)
    script_path = create_temp_script(input_code)
    process = execute_script(script_path, log_file_path)
    wait_for_error_in_log(log_file_path, expected_error_message)

    # clean up
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait()

    if os.path.exists(log_file_path):
        os.remove(log_file_path)


@pytest.mark.level1
@pytest.mark.platform_ascend910b
@pytest.mark.env_onecard
def test_group_topk_k_value_error():
    """
    Feature: group_topk st
    Description: test k value error, k > group_num.
    Expectation: catch exception
    """
    log_name = "test_group_topk_k_value_error"
    input_code = (
        "group_num = 4\n"
        "k = 5"
    )
    expected_error = "Should be k <= group_num < expert, but k=5, group_num=4, expert=256"

    check_group_topk_exception(input_code, log_name, expected_error)


@pytest.mark.level1
@pytest.mark.platform_ascend910b
@pytest.mark.env_onecard
def test_group_topk_idx_arr_value_error():
    """
    Feature: group_topk st
    Description: test idx_arr value error, expert > idx_arr.size.
    Expectation: catch exception
    """
    log_name = "test_group_topk_idx_arr_value_error"
    input_code = "idx_arr = ms.Tensor(np.arange(128, dtype=np.int32))"
    expected_error = "Should be expert <= idx_arr.size, but expert=256, idx_arr.size=128"

    check_group_topk_exception(input_code, log_name, expected_error)


@pytest.mark.level1
@pytest.mark.platform_ascend910b
@pytest.mark.env_onecard
def test_group_topk_expert_num_greater_than_1024():
    """
    Feature: group_topk st
    Description: test expert_num value error, expert_num > 1024.
    Expectation: catch exception
    """
    log_name = "test_group_topk_expert_num_greater_than_1024"
    input_code = "token = ms.Tensor(np.random.rand(512, 2048), ms.float16)"
    expected_error = "Should be expert <= idx_arr.size, but expert=2048, idx_arr.size=1024"

    check_group_topk_exception(input_code, log_name, expected_error)


@pytest.mark.level1
@pytest.mark.platform_ascend910b
@pytest.mark.env_onecard
def test_group_topk_expert_num_value_error():
    """
    Feature: group_topk st
    Description: test expert_num value error, expert_num cannot be divided by group_num.
    Expectation: catch exception
    """
    log_name = "test_group_topk_expert_num_value_error"
    input_code = "token = ms.Tensor(np.random.rand(512, 255), ms.float16)"
    expected_error = "The number of experts should be divisible by the number of groups, but expert=255, group_num=8"

    check_group_topk_exception(input_code, log_name, expected_error)


@pytest.mark.level1
@pytest.mark.platform_ascend910b
@pytest.mark.env_onecard
def test_group_topk_token_value_error():
    """
    Feature: group_topk st
    Description: test token value error, whose dimension > 3.
    Expectation: catch exception
    """
    log_name = "test_group_topk_token_value_error"
    input_code = "token = ms.Tensor(np.random.rand(2, 256, 512, 256), ms.float16)"
    expected_error = "The input0'shape should be 2 dim or 3 dim, input1's shape should be 1 dim"

    check_group_topk_exception(input_code, log_name, expected_error)



@pytest.mark.level1
@pytest.mark.platform_ascend910b
@pytest.mark.env_onecard
def test_group_topk_k_inner_value_error():
    """
    Feature: group_topk st
    Description: test k_inner value error, k_inner > expert_num/group_num.
    Expectation: catch exception
    """
    ms.set_context(mode=ms.PYNATIVE_MODE)
    ms.runtime.launch_blocking()

    token, idx_arr, group_num, k, k_inner = generate_default_test_inputs()
    k_inner = token.shape[1]/group_num + 1

    try:
        ms_custom_ops.group_topk(token=token, idx_arr=idx_arr, group_num=group_num,
                                     k=k, k_inner=k_inner)
        assert False
    except TypeError as e:
        assert "incompatible function arguments" in str(e), "Abnormal information mismatch."
