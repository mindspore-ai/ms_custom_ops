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

np.random.seed(21)  # 固定随机种子
np.set_printoptions(suppress=True)


def run_ms_sfa(query_tensor, key_tensor, value_tensor, sparse_indices_tensor, block_table, act_seq_q_tensor, act_seq_kv_tensor,
           query_rope_tensor, key_rope_tensor, scale_value, sparse_block_size, layout_q, layout_kv, sparse_mode):
    return ms_custom_ops.sparse_flash_attention(
           query_tensor, key_tensor, value_tensor, sparse_indices_tensor, block_table, act_seq_q_tensor, act_seq_kv_tensor,
           query_rope_tensor, key_rope_tensor, scale_value, sparse_block_size, layout_q, layout_kv, sparse_mode)

def gather_kv(k_tensor, v_tensor, sparse_indices, sparse_block_size, sparse_count, batch, n2_idx, s1_idx,
             cur_actual_seq_lengths_kv):
    s2_sparse = []
    for sparse_id in sparse_indices:
        if sparse_id == -1: 
            break
        begin_idx = sparse_id * sparse_block_size
        end_idx = begin_idx + sparse_block_size \
                if begin_idx + sparse_block_size <= cur_actual_seq_lengths_kv else cur_actual_seq_lengths_kv
        s2_sparse.extend(np.arange(begin_idx, end_idx))

    k_sparse, v_sparse = k_tensor[batch, n2_idx, s2_sparse, :], v_tensor[batch, n2_idx, s2_sparse, :]

    return k_sparse, v_sparse


def mask(res, cur_actual_seq_q, cur_actual_seq, topk_indices, s1_idx, sparse_blocksize):
    # 求尾块ID和尾块长度
    sparse_tail_idx = np.ceil(cur_actual_seq / sparse_blocksize)
    sparse_tail_seq_len = cur_actual_seq % sparse_blocksize
    if sparse_tail_seq_len == 0:
        sparse_tail_seq_len = sparse_blocksize

    delta_s = cur_actual_seq - cur_actual_seq_q
    threshold = delta_s + s1_idx + 1

    s_idx = 0
    for _, sparse_id in enumerate(topk_indices):
        if sparse_id == -1:
            break
        begin_idx = sparse_id * sparse_blocksize
        block_len = sparse_blocksize if sparse_id != sparse_tail_idx - 1 else sparse_tail_seq_len
        end_idx = begin_idx + block_len
        if begin_idx < threshold and end_idx <= threshold:
            s_idx += block_len
            continue
        if end_idx > threshold:
            local_offset = 0 if threshold <= begin_idx else threshold - begin_idx
            mask_begin = s_idx + local_offset
            mask_end = s_idx + block_len

            res[:, mask_begin: mask_end] = -1e12
        s_idx += block_len

    return res


def softmax(x):
    x = x.astype(np.float32)
    x_max = x.max(axis=-1, keepdims=True)
    x_sub = x - x_max
    y = np.exp(x_sub)
    x_sum = y.sum(axis=-1, keepdims=True)
    ans = y / x_sum
    return ans


def cpu_sparse_flash_attention(
    query, key, value, sparse_indices, scale_value, sparse_block_size,
    actual_seq_lengths_query, actual_seq_lengths_kv,
    query_rope=None, key_rope=None,
    layout_query='BSND', layout_kv='BSND', sparse_mode=3, block_table=None):
    query = query.astype(np.float32)
    key = key.astype(np.float32)
    value = value.astype(np.float32)
    query_rope = query_rope.astype(np.float32)
    key_rope = key_rope.astype(np.float32)
    batch_size = len(actual_seq_lengths_query)
    num_heads = query.shape[2]
    num_kv_heads = key.shape[2]
    sparse_count = sparse_indices.shape[-1]
    g = num_heads // num_kv_heads

    q_bnsd_tensor = np.transpose(np.concatenate([query, query_rope], axis=-1), axes=(0, 2, 1, 3))
    k_bnsd_tensor = np.transpose(np.concatenate([key, key_rope], axis=-1), axes=(0, 2, 1, 3))
    v_bnsd_tensor = np.transpose(value, axes=(0, 2, 1, 3))
    matmul_dtype = np.float32
    out_shape_bnsd = list(q_bnsd_tensor.shape)
    out_shape_bnsd[-1] = out_shape_bnsd[-1] - query_rope.shape[-1]
    y = np.zeros(out_shape_bnsd, dtype=np.float32)

    for batch in range(batch_size):
        cur_acutal_seq_lengths_q = actual_seq_lengths_query[batch]
        cur_actual_seq_lengths_kv = actual_seq_lengths_kv[batch]
        for n2_idx in range(num_kv_heads):
            for s1_idx in range(cur_acutal_seq_lengths_q):
                q_curr = q_bnsd_tensor[batch, n2_idx * g: (n2_idx + 1) * g, s1_idx, :]
                cur_sparse_indices = sparse_indices[batch, n2_idx, s1_idx, :]
                k_sparse, v_sparse = gather_kv(k_bnsd_tensor, v_bnsd_tensor, cur_sparse_indices, sparse_block_size,
                                              sparse_count, batch, n2_idx, s1_idx, cur_actual_seq_lengths_kv)
                mm1_res = np.matmul(q_curr.astype(np.float32), k_sparse.astype(np.float32).T, dtype=matmul_dtype)
                scale_res = mm1_res * scale_value
                if sparse_mode == 3:
                    mask_res = mask(scale_res, cur_acutal_seq_lengths_q, cur_actual_seq_lengths_kv,
                                    cur_sparse_indices, s1_idx, sparse_block_size)
                else:
                    mask_res = scale_res
                softmax_res = softmax(mask_res)
                mm2_res = np.matmul(softmax_res, v_sparse, dtype=matmul_dtype)
                y[batch, n2_idx * g: (n2_idx + 1) * g, s1_idx, :] = mm2_res
    return np.transpose(y, axes=(0, 2, 1, 3))


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_ascend910b
@pytest.mark.parametrize('graph_mode', [True, False])
def test_sfa(graph_mode):
    """
    Feature: test_sfa
    Description: base function
    Expectation: result is OK
    """
    scale_value = 0.041666666666666664
    sparse_block_size = 1

    query = np.random.uniform(-10, 10, (4, 1, 128, 512)).astype(np.float16)
    key = np.random.uniform(-5, 10, (4, 8192, 1, 512)).astype(np.float16)
    value = key
    sparse_indices = np.random.uniform(0, 16, (4, 1, 1, 2048)).astype(np.int32)
    query_rope = np.random.uniform(-10, 10, (4, 1, 128, 64)).astype(np.float16)
    key_rope = np.random.uniform(-10, 10, (4, 8192, 1, 64)).astype(np.float16)
    act_seq_q = [1] * 4
    act_seq_kv = [4096] * 4

    # start run custom ops
    query_tensor = ms.Tensor(query)
    key_tensor = ms.Tensor(key)
    value_tensor = ms.Tensor(value)
    sparse_indices_tensor = ms.Tensor(sparse_indices)
    query_rope_tensor = ms.Tensor(query_rope)
    key_rope_tensor = ms.Tensor(key_rope)
    act_seq_q_tensor = ms.Tensor(act_seq_q, dtype=ms.int32)
    act_seq_kv_tensor = ms.Tensor(act_seq_kv, dtype=ms.int32)

    if graph_mode:
        npu_out = ms.jit(run_ms_sfa)(query_tensor, key_tensor, value_tensor, sparse_indices_tensor, None,
                            act_seq_q_tensor, act_seq_kv_tensor, query_rope_tensor, key_rope_tensor,
                            scale_value, sparse_block_size, 0, 0, 3)
    else:
        npu_out = run_ms_sfa(query_tensor, key_tensor, value_tensor, sparse_indices_tensor, None,
                             act_seq_q_tensor, act_seq_kv_tensor, query_rope_tensor, key_rope_tensor,
                             scale_value, sparse_block_size, 0, 0, 3)

    # compare result
    cpu_out = cpu_sparse_flash_attention(
        query, key, value, sparse_indices, scale_value, sparse_block_size,
        actual_seq_lengths_query=act_seq_q, actual_seq_lengths_kv=act_seq_kv,
        query_rope=query_rope, key_rope=key_rope,
        layout_query='BSND', layout_kv='BSND', sparse_mode=3, block_table=None)

    res = np.isclose(npu_out.astype(np.float32), cpu_out, rtol=0.005, atol=0.0001, equal_nan=False)
    true_ratio = np.mean(res)
    if true_ratio < 0.99:
        print("npu output:\n", npu_out, npu_out.shape)
        print("cpu output:\n", cpu_out, cpu_out.shape)
        print("correct ratio of cpu vs npu is:", true_ratio * 100, "%")
    assert true_ratio > 0.99


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_ascend910b
@pytest.mark.parametrize('graph_mode', [True, False])
def test_sfa_exception(graph_mode):
    """
    Feature: test_sfa
    Description: expcetion cases
    Expectation: raise exception
    """
    scale_value = 0.041666666666666664
    sparse_block_size = 1

    query = np.random.uniform(-10, 10, (4, 1, 128, 512)).astype(np.float16)
    key = np.random.uniform(-5, 10, (4, 8192, 1, 512)).astype(np.float16)
    value = key
    sparse_indices = np.random.uniform(0, 16, (4, 1, 1, 2048)).astype(np.int32)
    query_rope = np.random.uniform(-10, 10, (4, 1, 128, 64)).astype(np.float16)
    key_rope = np.random.uniform(-10, 10, (4, 8192, 1, 64)).astype(np.float16)
    act_seq_q = [1] * 4
    act_seq_kv = [4096] * 4

    # start run custom ops
    query_tensor = ms.Tensor(query)
    key_tensor = ms.Tensor(key)
    value_tensor = ms.Tensor(value)
    sparse_indices_tensor = ms.Tensor(sparse_indices)
    query_rope_tensor = ms.Tensor(query_rope)
    key_rope_tensor = ms.Tensor(key_rope)
    act_seq_q_tensor = ms.Tensor(act_seq_q, dtype=ms.int32)
    act_seq_kv_tensor = ms.Tensor(act_seq_kv, dtype=ms.int32)

    # test invalid layout_query
    with pytest.raises(Exception):
        if graph_mode:
            out = ms.jit(run_ms_sfa)(query_tensor, key_tensor, value_tensor, sparse_indices_tensor, None,
                                act_seq_q_tensor, act_seq_kv_tensor, query_rope_tensor, key_rope_tensor,
                                scale_value, sparse_block_size, 2, 0, 3)
        else:
            out = run_ms_sfa(query_tensor, key_tensor, value_tensor, sparse_indices_tensor, None,
                                act_seq_q_tensor, act_seq_kv_tensor, query_rope_tensor, key_rope_tensor,
                                scale_value, sparse_block_size, 2, 0, 3)
        out.asnumpy()
    
    # test invalid layout_kv
    with pytest.raises(Exception):
        if graph_mode:
            out = ms.jit(run_ms_sfa)(query_tensor, key_tensor, value_tensor, sparse_indices_tensor, None,
                                act_seq_q_tensor, act_seq_kv_tensor, query_rope_tensor, key_rope_tensor,
                                scale_value, sparse_block_size, 0, 2, 3)
        else:
            out = run_ms_sfa(query_tensor, key_tensor, value_tensor, sparse_indices_tensor, None,
                                act_seq_q_tensor, act_seq_kv_tensor, query_rope_tensor, key_rope_tensor,
                                scale_value, sparse_block_size, 0, 2, 3)
        out.asnumpy()

    # test invalid sparse_mode
    with pytest.raises(Exception):
        if graph_mode:
            out = ms.jit(run_ms_sfa)(query_tensor, key_tensor, value_tensor, sparse_indices_tensor, None,
                                act_seq_q_tensor, act_seq_kv_tensor, query_rope_tensor, key_rope_tensor,
                                scale_value, sparse_block_size, 0, 0, 2)
        else:
            out = run_ms_sfa(query_tensor, key_tensor, value_tensor, sparse_indices_tensor, None,
                                act_seq_q_tensor, act_seq_kv_tensor, query_rope_tensor, key_rope_tensor,
                                scale_value, sparse_block_size, 0, 0, 2)
        out.asnumpy()
