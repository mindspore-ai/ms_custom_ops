# Copyright 2024 Huawei Technologies Co., Ltd
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

import os
import sys
import logging
import numpy as np
import pytest
import mindspore as ms
from mindspore import Profiler
from mindspore import context
from mindspore.common.np_dtype import bfloat16
from mindspore.common.api import jit
from mindspore._c_expression import MSContext
from functools import wraps
from st_utils import custom_compare
import ms_custom_ops

np.set_printoptions(precision=2, suppress=True, linewidth=200)

def jit_for_graph_mode(fn):
    """
    A decorator that conditionally applies jit to a function at runtime based on the context mode.
    """
    jitted_fn = jit(fn)
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if context.get_context("mode") == context.GRAPH_MODE:
            return jitted_fn(*args, **kwargs)
        return fn(*args, **kwargs)
    return wrapper

ND = 0
FRACTAL_NZ = 1


class MatMulCustom(ms.nn.Cell):
    def __init__(self, weight, ta, tb, x1_format=ND, x2_format=ND, output_format=ND):
        super().__init__()
        self.weight = ms.Parameter(weight, requires_grad=False)
        self.weight.set_data(weight)
        self.trans_a = ta
        self.trans_b = tb
        self.x1_format = x1_format
        self.x2_format = x2_format
        self.output_format = output_format

    @jit_for_graph_mode
    def construct(self, i0):
        if self.x1_format == FRACTAL_NZ:
            i0 = ms_custom_ops.trans_data(i0, transdata_type=1)  # ND_TO_FRACTAL_NZ
        output = ms_custom_ops.mat_mul(i0, self.weight, 
                                       transpose_a=self.trans_a, 
                                       transpose_b=self.trans_b,
                                       x1_format=self.x1_format,
                                       x2_format=self.x2_format,
                                       output_format=self.output_format)
        if self.output_format == FRACTAL_NZ:
            output = ms_custom_ops.trans_data(output, transdata_type=0)  # FRACTAL_NZ_TO_ND
        return output


def matmul(m, k, n, trans_a=False, trans_b=False, with_bias=False, mstype=ms.float16, profiling=False, x1_format=ND, x2_format=ND, output_format=ND):
    os.environ['USE_LLM_CUSTOM_MATMUL'] = "off"
    os.environ['INTERNAL_PRINT_TILING'] = "on"

    if ms.float16 == mstype:
        np_type = np.float16
    elif ms.float32 == mstype:
        np_type = np.float32
    elif ms.bfloat16 == mstype:
        np_type = bfloat16
    if trans_a:
        i0_host = np.random.normal(0.0, 0.5, size=[k, m]).astype(np_type)
    else:
        i0_host = np.random.normal(0.0, 0.5, size=[m, k]).astype(np_type)

    if trans_b:
        i1_host = np.random.normal(0.0, 0.5, size=[n, k]).astype(np_type)
    else:
        i1_host = np.random.normal(0.0, 0.5, size=[k, n]).astype(np_type)

    i0_host_fp32 = i0_host.astype(np.float32)
    i1_host_fp32 = i1_host.astype(np.float32)
    if trans_a == False and trans_b == False:
        expect = np.matmul(i0_host_fp32, i1_host_fp32)
    elif trans_a == False and trans_b == True:
        expect = np.matmul(i0_host_fp32, i1_host_fp32.T)
    elif trans_a == True and trans_b == False:
        expect = np.matmul(i0_host_fp32.T, i1_host_fp32)
    elif trans_a == True and trans_b == True:
        expect = np.matmul(i0_host_fp32.T, i1_host_fp32.T)
    if with_bias:
        bias_host = i1_host.flatten()[:n]
        expect = expect + bias_host
    if with_bias:
        bias_host = i1_host.flatten()[:n]
        expect = expect + bias_host
    print("numpy compute done")

    input1 = ms.Tensor(i0_host_fp32, mstype)
    input2 = ms.Tensor(i1_host_fp32, mstype)
    if x2_format == FRACTAL_NZ or MSContext.get_instance().get_ascend_soc_version() == "ascend310p":
        input2 = ms_custom_ops.trans_data(input2, transdata_type=1)  # ND_TO_FRACTAL_NZ
        x2_format = FRACTAL_NZ

    net = MatMulCustom(input2, trans_a, trans_b, x1_format, x2_format, output_format)

    if profiling:
        for i in range(50):
            output = net(input1)
        return

    output = net(input1)
    output_fp32 = output.astype(ms.float32)
    output_np = output_fp32.asnumpy()
    res = custom_compare(expect, output_np, mstype)
    assert res, "matmul compare fail."


@pytest.mark.level0
@pytest.mark.platform_ascend310p
@pytest.mark.parametrize("exec_mode", [context.GRAPH_MODE, context.PYNATIVE_MODE])
@pytest.mark.parametrize('trans_b', [False, True])
@pytest.mark.parametrize('mstype', [ms.float16])
@pytest.mark.parametrize('x1_format', [ND, FRACTAL_NZ])
@pytest.mark.parametrize('output_format', [ND, FRACTAL_NZ])
@pytest.mark.env_onecard
def test_matmul_1024_1024_1024_nz_input_fp16(exec_mode, trans_b, mstype, x1_format, output_format, request):
    """
    Feature: test matmul operator in graph and pynative mode, m n k must be aligned to 16 for pynative mode on 310p.
    Description: test matmul.
    Expectation: the result is correct
    """
    ms.set_context(device_target="Ascend", mode=exec_mode)
    ms.set_context(jit_config={"jit_level": "O0", "infer_boost": "on"})
    matmul(1024, 1024, 1024, trans_a=False, trans_b=trans_b, mstype=mstype, x1_format=x1_format, output_format=output_format)

@pytest.mark.level0
@pytest.mark.platform_ascend910b
@pytest.mark.parametrize("exec_mode", [context.GRAPH_MODE, context.PYNATIVE_MODE])
@pytest.mark.parametrize('trans_b', [False, True])
@pytest.mark.parametrize('mstype', [ms.bfloat16])
@pytest.mark.env_onecard
def test_matmul_1024_1024_1024_input_bfp16(exec_mode, trans_b, mstype, request):
    """
    Feature: test matmul operator in graph mode
    Description: test matmul.
    Expectation: the result is correct
    """
    ms.set_context(device_target="Ascend", mode=exec_mode)
    ms.set_context(jit_config={"jit_level": "O0", "infer_boost": "on"})
    if "platform_ascend310p" in request.config.getoption("-m") and mstype is ms.bfloat16:
        pytest.skip("Skipping ms.bfloat16 for 310p mark")
    matmul(1024, 1024, 1024, trans_a=False, trans_b=trans_b, mstype=mstype)

@pytest.mark.level0
@pytest.mark.platform_ascend910b
@pytest.mark.platform_ascend310p
@pytest.mark.parametrize("exec_mode", [context.GRAPH_MODE, context.PYNATIVE_MODE])
@pytest.mark.parametrize('trans_b', [False, True])
@pytest.mark.parametrize('mstype', [ms.float16])
@pytest.mark.env_onecard
def test_matmul_2048_2048_2048_nd_input_fp16(exec_mode, trans_b, mstype, request):
    """
    Feature: test matmul operator in graph mode
    Description: test matmul.
    Expectation: the result is correct
    """
    ms.set_context(device_target="Ascend", mode=exec_mode)
    ms.set_context(jit_config={"jit_level": "O0", "infer_boost": "on"})
    matmul(2048, 2048, 2048, trans_a=False, trans_b=trans_b, mstype=mstype)

@pytest.mark.level2
@pytest.mark.platform_ascend910b
@pytest.mark.parametrize("exec_mode", [context.GRAPH_MODE, context.PYNATIVE_MODE])
@pytest.mark.parametrize('trans_b', [False, True])
@pytest.mark.parametrize('mstype', [ms.bfloat16])
@pytest.mark.env_onecard
def test_matmul_2048_2048_2048_nd_input_bfp16(exec_mode, trans_b, mstype, request):
    """
    Feature: test matmul operator in graph mode
    Description: test matmul.
    Expectation: the result is correct
    """
    ms.set_context(device_target="Ascend", mode=exec_mode)
    ms.set_context(jit_config={"jit_level": "O0", "infer_boost": "on"})
    if "platform_ascend310p" in request.config.getoption("-m") and mstype is ms.bfloat16:
        pytest.skip("Skipping ms.bfloat16 for 310p mark")
    matmul(2048, 2048, 2048, trans_a=False, trans_b=trans_b, mstype=mstype)

@pytest.mark.level0
@pytest.mark.platform_ascend310p
@pytest.mark.parametrize("exec_mode", [context.GRAPH_MODE, context.PYNATIVE_MODE])
@pytest.mark.parametrize('trans_b', [True])
@pytest.mark.parametrize('mstype', [ms.float16])
@pytest.mark.parametrize('x1_format', [ND, FRACTAL_NZ])
@pytest.mark.parametrize('output_format', [ND, FRACTAL_NZ])
@pytest.mark.env_onecard
def test_matmul_1024_1234_1234_input_unaligned_k_n_fp16(exec_mode, trans_b, mstype, x1_format, output_format, request):
    """
    Feature: test matmul operator in graph and pynative mode, m n k must be aligned to 16 for pynative mode on 310p.
    Description: Test that unaligned large/edge n dimension raise exception
    Expectation: CheckDimensionAlignment validation rejects unaligned inputs
    """
    ms.set_context(device_target="Ascend", mode=exec_mode)
    ms.set_context(jit_config={"jit_level": "O0", "infer_boost": "on"})
    with pytest.raises(RuntimeError, match="dimension must be aligned"):
        matmul(1024, 1234, 1234, trans_a=False, trans_b=trans_b, mstype=mstype, x1_format=x1_format, output_format=output_format)
    logging.info(
        "Unaligned dimension correctly rejected: shape=%s",
        (1234, 1234)
    )


@pytest.mark.level2
@pytest.mark.platform_ascend910b
@pytest.mark.parametrize("exec_mode", [context.GRAPH_MODE, context.PYNATIVE_MODE])
@pytest.mark.parametrize('trans_b', [False, True])
@pytest.mark.parametrize('mstype', [ms.bfloat16])
@pytest.mark.env_onecard
def test_matmul_1024_1234_1234_nz_input_unaligned_k_n_bfp16(exec_mode, trans_b, mstype, request):
    """
    Feature: test matmul operator in graph mode
    Description: test matmul.
    Expectation: the result is correct
    """
    ms.set_context(device_target="Ascend", mode=exec_mode)
    ms.set_context(jit_config={"jit_level": "O0", "infer_boost": "on"})
    if "platform_ascend310p" in request.config.getoption("-m") and mstype is ms.bfloat16:
        pytest.skip("Skipping ms.bfloat16 for 310p mark")
    matmul(1024, 1234, 1234, trans_a=False, trans_b=trans_b, mstype=mstype)

@pytest.mark.level2
@pytest.mark.platform_ascend910b
@pytest.mark.parametrize("exec_mode", [context.GRAPH_MODE, context.PYNATIVE_MODE])
@pytest.mark.parametrize('trans_b', [False, True])
@pytest.mark.parametrize('mstype', [ms.float16])
@pytest.mark.env_onecard
def test_matmul_1024_2048_2234_nd_input_unaligned_n_fp16(exec_mode, trans_b, mstype, request):
    """
    Feature: test matmul operator in graph mode
    Description: test matmul.
    Expectation: the result is correct
    """
    ms.set_context(device_target="Ascend", mode=exec_mode)
    ms.set_context(jit_config={"jit_level": "O0", "infer_boost": "on"})
    matmul(1024, 2048, 2234, trans_a=False, trans_b=trans_b, mstype=mstype)

@pytest.mark.level2
@pytest.mark.platform_ascend910b
@pytest.mark.parametrize("exec_mode", [context.GRAPH_MODE, context.PYNATIVE_MODE])
@pytest.mark.parametrize('trans_b', [False, True])
@pytest.mark.parametrize('mstype', [ms.bfloat16])
@pytest.mark.env_onecard
def test_matmul_1024_2048_2234_nd_input_unaligned_n_bfp16(exec_mode, trans_b, mstype, request):
    """
    Feature: test matmul operator in graph mode
    Description: test matmul.
    Expectation: the result is correct
    """
    ms.set_context(device_target="Ascend", mode=exec_mode)
    ms.set_context(jit_config={"jit_level": "O0", "infer_boost": "on"})
    if "platform_ascend310p" in request.config.getoption("-m") and mstype is ms.bfloat16:
        pytest.skip("Skipping ms.bfloat16 for 310p mark")
    matmul(1024, 2048, 2234, trans_a=False, trans_b=trans_b, mstype=mstype)


@pytest.mark.level0
@pytest.mark.platform_ascend310p
@pytest.mark.parametrize("exec_mode", [context.GRAPH_MODE, context.PYNATIVE_MODE])
@pytest.mark.parametrize('m', [1, 10, 20])
@pytest.mark.parametrize('x1_format', [ND])
@pytest.mark.parametrize('output_format', [ND])
@pytest.mark.env_onecard
def test_matmul_m_4096_4096_False_True_float16_nd_input(exec_mode, m, x1_format, output_format):
    """
    Feature: test matmul operator in graph and pynative mode, when x1_format and out_format is ND, m no need to be aligned to 16.
    Description: test matmul.
    Expectation: the result is correct
    """
    ms.set_context(device_target="Ascend", mode=exec_mode)
    ms.set_context(jit_config={"jit_level": "O0", "infer_boost": "on"})
    matmul(m, 4096, 4096, trans_a=False, trans_b=True, mstype=ms.float16, x1_format=x1_format, output_format=output_format)


@pytest.mark.level0
@pytest.mark.platform_ascend310p
@pytest.mark.parametrize("exec_mode", [context.GRAPH_MODE])
@pytest.mark.parametrize('m', [1, 10, 20])
@pytest.mark.parametrize('x1_format', [FRACTAL_NZ])
@pytest.mark.parametrize('output_format', [ND, FRACTAL_NZ])
@pytest.mark.env_onecard
def test_matmul_m_4096_4096_False_True_float16_nz_input(exec_mode, m, x1_format, output_format):
    """
    Feature: matmul operator input_x1 dimension alignment validation for edge cases
    Description: Test that unaligned large/edge m dimension raise exception
    Expectation: CheckDimensionAlignment validation rejects unaligned inputs
    """
    ms.set_context(device_target="Ascend", mode=exec_mode)
    ms.set_context(jit_config={"jit_level": "O0", "infer_boost": "on"})
    matmul(m, 4096, 4096, trans_a=False, trans_b=True, mstype=ms.float16, x1_format=x1_format, output_format=output_format)


@pytest.mark.level0
@pytest.mark.platform_ascend310p
@pytest.mark.parametrize("exec_mode", [context.PYNATIVE_MODE])
@pytest.mark.parametrize('m', [1, 10, 20])
@pytest.mark.parametrize('x1_format', [FRACTAL_NZ])
@pytest.mark.parametrize('output_format', [ND, FRACTAL_NZ])
@pytest.mark.env_onecard
def test_matmul_m_4096_4096_False_True_float16_nz_input_pynative(exec_mode, m, x1_format, output_format):
    """
    Feature: matmul operator input_x1 dimension alignment validation for edge cases
    Description: Test that unaligned large/edge m dimension raise exception
    Expectation: CheckDimensionAlignment validation rejects unaligned inputs
    """
    ms.set_context(device_target="Ascend", mode=exec_mode)
    ms.set_context(jit_config={"jit_level": "O0", "infer_boost": "on"})
    with pytest.raises(RuntimeError, match="dimension must be aligned"):
        matmul(m, 4096, 4096, trans_a=False, trans_b=True, mstype=ms.float16, x1_format=x1_format, output_format=output_format)
    logging.info(
        "Unaligned dimension correctly rejected: shape=%s",
        (m, 4096)
    )


@pytest.mark.level2
@pytest.mark.platform_ascend910b
@pytest.mark.parametrize("exec_mode", [context.GRAPH_MODE, context.PYNATIVE_MODE])
@pytest.mark.parametrize('m', [1, 256, 1024])
@pytest.mark.env_onecard
def test_matmul_m_4096_4096_False_True_float16_nz_input_unaligned_k_n(exec_mode, m):
    """
    Feature: test matmul operator in graph mode
    Description: test matmul.
    Expectation: the result is correct
    """
    ms.set_context(device_target="Ascend", mode=exec_mode)
    ms.set_context(jit_config={"jit_level": "O0", "infer_boost": "on"})
    matmul(m, 1234, 1234, trans_a=False, trans_b=True, mstype=ms.float16)

@pytest.mark.level2
@pytest.mark.platform_ascend910b
@pytest.mark.parametrize("exec_mode", [context.GRAPH_MODE, context.PYNATIVE_MODE])
@pytest.mark.parametrize('m', [1, 32, 1024])
@pytest.mark.env_onecard
def test_matmul_m_2048_2234_False_True_float16_nd_input_unaligned_n(exec_mode, m):
    """
    Feature: test matmul operator in graph mode
    Description: test matmul.
    Expectation: the result is correct
    """
    ms.set_context(device_target="Ascend", mode=exec_mode)
    ms.set_context(jit_config={"jit_level": "O0", "infer_boost": "on"})
    matmul(m, 2048, 2234, trans_a=False, trans_b=True, mstype=ms.float16)


@pytest.mark.level2
@pytest.mark.platform_ascend910b
@pytest.mark.platform_ascend310p
@pytest.mark.parametrize("exec_mode", [context.GRAPH_MODE, context.PYNATIVE_MODE])
@pytest.mark.env_onecard
def test_matmul_in_real_shape_increment(exec_mode):
    """
    Feature: test matmul operator in graph mode
    Description: test matmul.
    Expectation: the result is correct
    """
    ms.set_context(device_target="Ascend", mode=exec_mode)
    ms.set_context(jit_config={"jit_level": "O0", "infer_boost": "on"})
    prof_flag = False
    profiler = Profiler(start_profile=False, output_path="profiler")
    profiler.start()
    matmul(16, 2752, 4096, trans_a=False, trans_b=True, mstype=ms.float16, profiling=prof_flag)
    matmul(16, 32, 4096, trans_a=False, trans_b=True, mstype=ms.float16, profiling=prof_flag)
    matmul(16, 4096, 32, trans_a=False, trans_b=True, mstype=ms.float16, profiling=prof_flag)
    matmul(16, 4096, 8256, trans_a=False, trans_b=True, mstype=ms.float16, profiling=prof_flag)
    matmul(32, 2752, 4096, trans_a=False, trans_b=True, mstype=ms.float16, profiling=prof_flag)
    matmul(32, 32, 4096, trans_a=False, trans_b=True, mstype=ms.float16, profiling=prof_flag)
    matmul(32, 4096, 32, trans_a=False, trans_b=True, mstype=ms.float16, profiling=prof_flag)
    matmul(32, 4096, 8256, trans_a=False, trans_b=True, mstype=ms.float16, profiling=prof_flag)
    matmul(48, 2752, 4096, trans_a=False, trans_b=True, mstype=ms.float16, profiling=prof_flag)
    matmul(48, 32, 4096, trans_a=False, trans_b=True, mstype=ms.float16, profiling=prof_flag)
    matmul(48, 4096, 32, trans_a=False, trans_b=True, mstype=ms.float16, profiling=prof_flag)
    matmul(48, 4096, 8256, trans_a=False, trans_b=True, mstype=ms.float16, profiling=prof_flag)
    matmul(64, 2752, 4096, trans_a=False, trans_b=True, mstype=ms.float16, profiling=prof_flag)
    matmul(64, 32, 4096, trans_a=False, trans_b=True, mstype=ms.float16, profiling=prof_flag)
    matmul(64, 4096, 32, trans_a=False, trans_b=True, mstype=ms.float16, profiling=prof_flag)
    matmul(64, 4096, 8256, trans_a=False, trans_b=True, mstype=ms.float16, profiling=prof_flag)
    # matmul(1, 4096, 38108, trans_a=False, trans_b=True, mstype=ms.float16, profiling=prof_flag) # precisioin error, unalign cases
    matmul(1, 2752, 4096, trans_a=False, trans_b=True, mstype=ms.float16, profiling=prof_flag)
    matmul(1, 32, 4096, trans_a=False, trans_b=True, mstype=ms.float16, profiling=prof_flag)
    matmul(1, 4096, 32, trans_a=False, trans_b=True, mstype=ms.float16, profiling=prof_flag)
    matmul(1, 4096, 8256, trans_a=False, trans_b=True, mstype=ms.float16, profiling=prof_flag)
    # matmul(128, 4096, 38108, trans_a=False, trans_b=True, mstype=ms.float16, profiling=prof_flag) # precisioin error, unalign cases
    matmul(128, 2752, 4096, trans_a=False, trans_b=True, mstype=ms.float16, profiling=prof_flag)
    matmul(128, 32, 4096, trans_a=False, trans_b=True, mstype=ms.float16, profiling=prof_flag)
    matmul(128, 4096, 32, trans_a=False, trans_b=True, mstype=ms.float16, profiling=prof_flag)
    matmul(128, 4096, 8256, trans_a=False, trans_b=True, mstype=ms.float16, profiling=prof_flag)
    profiler.stop()
    profiler.analyse()


@pytest.mark.level2
@pytest.mark.platform_ascend910b
@pytest.mark.platform_ascend310p
@pytest.mark.env_onecard
@pytest.mark.parametrize("exec_mode", [context.GRAPH_MODE, context.PYNATIVE_MODE])
@pytest.mark.parametrize('m', [514, 1024])
@pytest.mark.parametrize("prof_flag_str", [0, 1])
def test_matmul_in_real_shape_prefill(exec_mode, m, prof_flag_str):
    """
    Feature: test matmul operator in graph mode
    Description: test matmul.
    Expectation: the result is correct
    """
    ms.set_context(device_target="Ascend", mode=exec_mode)
    ms.set_context(jit_config={"jit_level": "O0", "infer_boost": "on"})
    prof_flag = bool(int(prof_flag_str))
    profiler = Profiler(start_profile=False, output_path="profiler")
    profiler.start()
    matmul(m, 2752, 4096, trans_a=False, trans_b=True, mstype=ms.float16, profiling=prof_flag)
    matmul(m, 32, 4096, trans_a=False, trans_b=True, mstype=ms.float16, profiling=prof_flag)
    matmul(m, 4096, 32, trans_a=False, trans_b=True, mstype=ms.float16, profiling=prof_flag)
    matmul(m, 4096, 8256, trans_a=False, trans_b=True, mstype=ms.float16, profiling=prof_flag)
    matmul(m, 5504, 4096, trans_a=False, trans_b=True, mstype=ms.float16, profiling=prof_flag)
    profiler.stop()
    profiler.analyse()
