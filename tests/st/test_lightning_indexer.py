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
""" tests_custom_pyboost_ascend """

import numpy as np
import mindspore as ms
import pytest
import ms_custom_ops


def _get_data_from_pa_cache(key, block_table, act_s2):
    """从PagedAttention缓存中获取数据，使用MindSpore接口实现"""
    key_np = key.asnumpy()
    block_table_np = block_table.asnumpy()
    
    _, block_size, n2, d = key.shape
    if n2 != 1:
        raise ValueError("n2 only support 1")
    need_block_num = (act_s2 + block_size - 1) // block_size
    act_s2_align = need_block_num * block_size
    
    out_np = np.zeros((act_s2_align, d), dtype=key_np.dtype)
    for i in range(need_block_num):
        out_np[i*block_size:(i+1)*block_size, :] = key_np[block_table_np[i], ...].reshape(block_size, d)

    return ms.Tensor(out_np[:act_s2, :], dtype=key.dtype)


def _lightning_indexer(query, key, weights, actual_seq_lengths_query, actual_seq_lengths_key, block_table,
                       layout_query="BSND", sparse_count=2048, sparse_mode=3):
    """
    MindSpore接口实现的lightning indexer函数
    """
    batch_size = query.shape[0]
    if layout_query == "TND":
        batch_size = actual_seq_lengths_query.shape[0]
    
    # 获取输出形状
    out_shape = list(query.shape)
    n2 = key.shape[2]
    d = query.shape[-1]
    n1 = query.shape[-2]
    out_shape[-1] = sparse_count
    out_shape[-2] = n2
    
    # 使用MindSpore张量操作初始化输出
    out = ms.ops.fill(ms.int32, tuple(out_shape), -1)  # 初始化为全-1
    out = ms.ops.reshape(out, (-1, n2, sparse_count))
    
    act_s1 = 0
    act_s2 = 0
    process_q_len = 0
    
    for batch_id in range(batch_size):
        if actual_seq_lengths_query is None:
            # 只能为BSND格式
            act_s1 = query.shape[1]
        else:
            if layout_query == "TND":  # TND格式时actual_seq_lengths_query为前缀和
                act_s1_np = actual_seq_lengths_query[batch_id].asnumpy()
                act_s1 = int(act_s1_np) - process_q_len
            else:
                act_s1_np = actual_seq_lengths_query[batch_id].asnumpy()
                act_s1 = int(act_s1_np)
        act_s2_np = actual_seq_lengths_key[batch_id].asnumpy()
        act_s2 = int(act_s2_np)
        
        # 使用MindSpore操作进行张量变换
        reshaped_query = ms.ops.reshape(query, (-1, n1, d))
        query_slice = ms.ops.slice(reshaped_query, (process_q_len, 0, 0), (act_s1, n1, d))
        now_q = ms.ops.transpose(query_slice, (1, 0, 2))  # 相当于transpose(0, 1)
        now_q = now_q.astype(ms.float32)
        
        reshaped_weights = ms.ops.reshape(weights, (-1, n1, 1))
        weights_slice = ms.ops.slice(reshaped_weights, (process_q_len, 0, 0), (act_s1, n1, 1))
        now_weights = ms.ops.transpose(weights_slice, (1, 0, 2))
        now_weights = now_weights.astype(ms.float32)
        
        process_q_len += act_s1
        now_block_table = block_table[batch_id, :]
        
        # 使用MindSpore接口处理key
        now_k = _get_data_from_pa_cache(key, now_block_table, act_s2)
        now_k = ms.ops.transpose(now_k, (1, 0))  # 相当于transpose(0, 1)
        now_k = now_k.astype(ms.float32)
        
        # 使用MindSpore的矩阵乘法
        matmul_out = ms.ops.matmul(now_q, now_k)
        relu_out = ms.ops.maximum(matmul_out, 0)
        weight_out = relu_out * now_weights
        
        # 使用MindSpore的求和操作
        reduce_out = ms.ops.reduce_sum(weight_out, axis=0)  # 沿axis 0求和
        
        # sparse场景下三角置为-inf
        tmp_s1, tmp_s2 = int(reduce_out.shape[0]), int(reduce_out.shape[1])
        if sparse_mode == 3:
            # 将MindSpore张量转换为numpy进行复杂的索引操作
            reduce_out_np = reduce_out.asnumpy()
            for i in range(tmp_s1):
                row_idx = tmp_s1 - 1 - i
                col_start = tmp_s2 - i
                if col_start < tmp_s2:
                    reduce_out_np[row_idx, col_start:] = float('-inf')
            reduce_out = ms.Tensor(reduce_out_np, dtype=ms.float32)
        
        # 使用MindSpore的排序操作
        sorted_indices = ms.ops.argsort(reduce_out, axis=1, descending=True)
        sorted_indices = sorted_indices.astype(ms.int32)
        
        return_s2 = min(sparse_count, tmp_s2)
        
        # 将结果存储到输出张量中
        # 转换为numpy进行复杂索引操作，然后转回MindSpore张量
        out_np = out.asnumpy()
        sorted_indices_np = sorted_indices.asnumpy()
        
        # 限制输出维度并赋值
        out_start = process_q_len - act_s1
        out_end = process_q_len
        selected_indices_np = sorted_indices_np[:, :return_s2]
        
        # 更新输出数组
        out_np[out_start:out_end, 0, :return_s2] = selected_indices_np
        
        out = ms.Tensor(out_np, dtype=ms.int32)

    out = ms.ops.reshape(out, out_shape)
    return out.asnumpy()


def run_ms_lightning_indexer(query_tensor, key_tensor, weights_tensor, q_lens_tensor, 
                             k_lens_tensor, block_table_tensor, layout_query, 
                             layout_key, sparse_count, sparse_mode):
    return ms_custom_ops.lightning_indexer(query_tensor, key_tensor, weights_tensor, q_lens_tensor, 
                                           k_lens_tensor, block_table_tensor, layout_query=layout_query, 
                                           layout_key=layout_key, sparse_count=sparse_count, sparse_mode=sparse_mode)

@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_ascend910b
@pytest.mark.parametrize('graph_mode', [True, False])
def test_bsnd_lightning_indexer(graph_mode):
    """
    Feature: test_lightning_indexer
    Description: base function
    Expectation: result is OK
    """
    b = 1
    s1 = 1
    s2 = 8192
    n1 = 64
    n2 = 1
    d = 128
    block_size = 256
    t = 8192
    layout_query = 'BSND'

    np.random.seed(0)
    query = np.random.uniform(-10, 10, (b, s1, n1, d)).astype(np.float32)
    key = np.random.uniform(-10, 10, (b*(s2//block_size), block_size, n2, d)).astype(np.float32)
    weights = np.random.uniform(-1, 1, (b, s1, n1, 1)).astype(np.float32)
    actual_seq_lengths_query = np.random.uniform(s1, s1, (b)).astype(np.int32)
    actual_seq_lengths_key = np.random.uniform(s2, s2, (b)).astype(np.int32)
    block_table = np.array([range(b*s2//block_size)], dtype=np.int32).reshape(b, -1)
    layout_key = 'PA_BSND'
    sparse_count = 2048
    sparse_mode = 3

    query_tensor = ms.Tensor(query, dtype=ms.bfloat16)
    key_tensor = ms.Tensor(key, dtype=ms.bfloat16)
    weights_tensor = ms.Tensor(weights, dtype=ms.bfloat16)
    q_lens_tensor = ms.Tensor(actual_seq_lengths_query)
    k_lens_tensor = ms.Tensor(actual_seq_lengths_key)
    block_table_tensor = ms.Tensor(block_table)

    std_out = _lightning_indexer(query_tensor, key_tensor, weights_tensor, q_lens_tensor, k_lens_tensor, block_table_tensor,
                                 layout_query, sparse_count, sparse_mode)

    # start run custom ops
    if graph_mode:
        npu_out = ms.jit(run_ms_lightning_indexer)(
            query_tensor, key_tensor, weights_tensor, q_lens_tensor, 
            k_lens_tensor, block_table_tensor, layout_query=0, 
            layout_key=0, sparse_count=sparse_count, sparse_mode=sparse_mode)
    else:
        npu_out = run_ms_lightning_indexer(
            query_tensor, key_tensor, weights_tensor, q_lens_tensor, 
            k_lens_tensor, block_table_tensor, layout_query=0, 
            layout_key=0, sparse_count=sparse_count, sparse_mode=sparse_mode)

    # compare result
    npu_out = npu_out.asnumpy().astype(np.float32).reshape(-1, sparse_count)
    std_out = std_out.reshape(-1, sparse_count)
    t = npu_out.shape[0]
    for i in range(t):
        for j in range(sparse_count):
            if npu_out[i][j] != std_out[i][j]:
                print("t K npu cpu = ", i, j, npu_out[i][j], std_out[i][j])
    
    assert np.allclose(npu_out, std_out, 0.0001, 0.0001)



@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_ascend910b
@pytest.mark.parametrize('graph_mode', [True, False])
def test_bsnd_lightning_indexer_exception(graph_mode):
    """
    Feature: test_sfa
    Description: expcetion cases
    Expectation: raise exception
    """
    b = 1
    s1 = 1
    s2 = 8192
    n1 = 64
    n2 = 1
    d = 128
    block_size = 256
    layout_query = 0

    np.random.seed(0)
    query = np.random.uniform(-10, 10, (b, s1, n1, d)).astype(np.float32)
    key = np.random.uniform(-10, 10, (b*(s2//block_size), block_size, n2, d)).astype(np.float32)
    weights = np.random.uniform(-1, 1, (b, s1, n1, 1)).astype(np.float32)
    actual_seq_lengths_query = np.random.uniform(s1, s1, (b)).astype(np.int32)
    actual_seq_lengths_key = np.random.uniform(s2, s2, (b)).astype(np.int32)
    block_table = np.array([range(b*s2//block_size)], dtype=np.int32).reshape(b, -1)
    layout_key = 0
    sparse_count = 2048
    sparse_mode = 3

    query_tensor = ms.Tensor(query, dtype=ms.bfloat16)
    key_tensor = ms.Tensor(key, dtype=ms.bfloat16)
    weights_tensor = ms.Tensor(weights, dtype=ms.bfloat16)
    q_lens_tensor = ms.Tensor(actual_seq_lengths_query)
    k_lens_tensor = ms.Tensor(actual_seq_lengths_key)
    block_table_tensor = ms.Tensor(block_table)

    # invalid layout_query
    with pytest.raises(Exception):
        layout_query = 2
        if graph_mode:
            out = ms.jit(run_ms_lightning_indexer)(
                query_tensor, key_tensor, weights_tensor, q_lens_tensor, 
                k_lens_tensor, block_table_tensor, layout_query=layout_query, 
                layout_key=layout_key, sparse_count=sparse_count, sparse_mode=sparse_mode)
        else:
            out = run_ms_lightning_indexer(
                query_tensor, key_tensor, weights_tensor, q_lens_tensor, 
                k_lens_tensor, block_table_tensor, layout_query=layout_query, 
                layout_key=layout_key, sparse_count=sparse_count, sparse_mode=sparse_mode)
        out.asnumpy()

    # invalid layout_key
    with pytest.raises(Exception):
        layout_key = 1
        if graph_mode:
            out = ms.jit(run_ms_lightning_indexer)(
                query_tensor, key_tensor, weights_tensor, q_lens_tensor, 
                k_lens_tensor, block_table_tensor, layout_query=layout_query, 
                layout_key=layout_key, sparse_count=sparse_count, sparse_mode=sparse_mode)
        else:
            out = run_ms_lightning_indexer(
                query_tensor, key_tensor, weights_tensor, q_lens_tensor, 
                k_lens_tensor, block_table_tensor, layout_query=layout_query, 
                layout_key=layout_key, sparse_count=sparse_count, sparse_mode=sparse_mode)
        out.asnumpy()

    # invalid query_shape
    with pytest.raises(Exception):
        query = np.random.uniform(-10, 10, (s1, n1, d)).astype(np.float32)
        if graph_mode:
            out = ms.jit(run_ms_lightning_indexer)(
                query_tensor, key_tensor, weights_tensor, q_lens_tensor, 
                k_lens_tensor, block_table_tensor, layout_query=layout_query, 
                layout_key=layout_key, sparse_count=sparse_count, sparse_mode=sparse_mode)
        else:
            out = run_ms_lightning_indexer(
                query_tensor, key_tensor, weights_tensor, q_lens_tensor, 
                k_lens_tensor, block_table_tensor, layout_query=layout_query, 
                layout_key=layout_key, sparse_count=sparse_count, sparse_mode=sparse_mode)
        out.asnumpy()
