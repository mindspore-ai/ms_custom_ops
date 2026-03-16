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
"""Batch invariance tests for MLA paged attention."""

import math
import numpy as np
import mindspore as ms
from mindspore import Tensor, context

from test_custom_paged_attention import (
    INPUT_FORMAT_ND,
    INPUT_LAYOUT_BSND,
    MASK_UNDEFINED,
    PagedAttentionDataGenerator,
    PagedAttentionNet,
    QUANT_QKV_OFFLINE,
    QUANT_QKV_ONLINE,
    QUANT_UNQUANT,
    _paged_attention_test,
    _set_dynamic_shapes_for_pa,
)


def _execute_paged_attention_with_inputs(
    test_config: dict,
    run_mode: int,
    inputs_dict: dict,
    dynamic: bool = False,
):
    """Execute paged attention with prebuilt inputs and return the output tensor."""
    context.set_context(device_target="Ascend", mode=run_mode)

    num_heads = test_config["num_heads"]
    kv_heads = test_config["kv_heads"]
    head_size = test_config["head_size"]
    q_dtype = test_config["q_dtype"]
    kv_dtype = test_config["kv_dtype"]
    mask_type = test_config["mask_type"]
    qk_scale = test_config.get("qk_scale", 1.0 / math.sqrt(head_size))
    mla_v_dim = test_config.get("mla_v_dim", 0)

    query = Tensor(inputs_dict["query"], dtype=q_dtype)
    key_cache = Tensor(inputs_dict["key_cache"], dtype=kv_dtype)
    value_cache = Tensor(inputs_dict["value_cache"], dtype=kv_dtype)
    block_tables = Tensor(inputs_dict["block_tables"], dtype=ms.int32)
    q_seq_lens = Tensor(inputs_dict["q_seq_lens"], dtype=ms.int32)
    kv_seq_lens = Tensor(inputs_dict["kv_seq_lens"], dtype=ms.int32)
    mask = (
        Tensor(inputs_dict["mask"], dtype=inputs_dict["mask_dtype"])
        if inputs_dict["mask"] is not None
        else None
    )

    k_descale = (Tensor(inputs_dict["k_descale"], dtype=ms.float32)
                 if inputs_dict["k_descale"] is not None else None)
    v_descale = (Tensor(inputs_dict["v_descale"], dtype=ms.float32)
                 if inputs_dict["v_descale"] is not None else None)
    k_offset = (Tensor(inputs_dict["k_offset"], dtype=ms.int32)
                if inputs_dict["k_offset"] is not None else None)
    v_offset = (Tensor(inputs_dict["v_offset"], dtype=ms.int32)
                if inputs_dict["v_offset"] is not None else None)
    k_scale_per_head = (Tensor(inputs_dict["k_scale_per_head"], dtype=ms.float32)
                        if inputs_dict["k_scale_per_head"] is not None else None)
    v_scale_per_head = (Tensor(inputs_dict["v_scale_per_head"], dtype=ms.float32)
                        if inputs_dict["v_scale_per_head"] is not None else None)
    p_scale = (Tensor(inputs_dict["p_scale"], dtype=ms.float32)
               if inputs_dict["p_scale"] is not None else None)

    batch_run_status = test_config.get("batch_run_status")
    input_format = test_config.get("input_format", INPUT_FORMAT_ND)
    net = PagedAttentionNet(
        q_head_num=num_heads,
        qk_scale=qk_scale,
        kv_head_num=kv_heads,
        mask_type=mask_type,
        batch_run_status_enable=test_config.get("batch_run_status_enable", False),
        quant_type=test_config.get("quant_type", QUANT_UNQUANT),
        out_data_type=test_config.get("out_data_type", -1),
        has_quant_offset=test_config.get("has_quant_offset", False),
        compress_type=test_config.get("compress_type", 0),
        calc_type=test_config.get("calc_type", 0),
        scale_type=test_config.get("scale_type", 0),
        input_layout=test_config.get("input_layout", INPUT_LAYOUT_BSND),
        mla_v_dim=mla_v_dim,
        input_format=input_format,
    )

    quant_type = test_config.get("quant_type", QUANT_UNQUANT)
    if quant_type in [QUANT_QKV_OFFLINE, QUANT_QKV_ONLINE]:
        final_k_descale = k_scale_per_head
        final_v_descale = v_scale_per_head
    else:
        final_k_descale = k_descale
        final_v_descale = v_descale

    if dynamic:
        dynamic_config = dict(test_config)
        dynamic_config["context_lens"] = inputs_dict["kv_seq_lens"].tolist()
        _set_dynamic_shapes_for_pa(net, dynamic_config)

    return net(
        query, key_cache, value_cache, block_tables,
        mask, batch_run_status,
        final_k_descale, k_offset, final_v_descale, v_offset,
        None, p_scale, None,
        q_seq_lens, kv_seq_lens
    )


def _build_pa_sample_inputs(generator: PagedAttentionDataGenerator, test_config: dict,
                            q_seq_len: int, kv_seq_len: int) -> dict:
    """Build one deterministic sample for later rebatching."""
    block_size = int(test_config["block_size"])
    num_blocks = max(1, (int(kv_seq_len) + block_size - 1) // block_size)
    sample_inputs = generator.generate_inputs(
        test_config["num_heads"],
        test_config["kv_heads"],
        test_config["head_size"],
        block_size,
        num_blocks,
        [int(q_seq_len)],
        [int(kv_seq_len)],
        test_config["q_dtype"],
        test_config["kv_dtype"],
        test_config["mask_type"],
        test_config.get("quant_type", QUANT_UNQUANT),
        test_config.get("has_quant_offset", False),
        test_config.get("mla_v_dim", 0),
        test_config.get("expected_dtype"),
    )
    sample_inputs["block_tables"] = np.arange(num_blocks, dtype=np.int32).reshape(1, num_blocks)
    return sample_inputs


def _assemble_pa_batch_inputs(generator: PagedAttentionDataGenerator, test_config: dict,
                              sample_inputs_list: list) -> dict:
    """Assemble several single-sample inputs into one batch with disjoint KV blocks."""
    q_seq_lens = [int(sample["q_seq_lens"][0]) for sample in sample_inputs_list]
    kv_seq_lens = [int(sample["kv_seq_lens"][0]) for sample in sample_inputs_list]

    query = np.concatenate([sample["query"] for sample in sample_inputs_list], axis=0)
    key_cache = np.concatenate([sample["key_cache"] for sample in sample_inputs_list], axis=0)
    value_cache = np.concatenate([sample["value_cache"] for sample in sample_inputs_list], axis=0)

    max_blocks = max(sample["block_tables"].shape[1] for sample in sample_inputs_list)
    block_tables = np.zeros((len(sample_inputs_list), max_blocks), dtype=np.int32)
    block_offset = 0
    for batch_idx, sample in enumerate(sample_inputs_list):
        sample_blocks = sample["key_cache"].shape[0]
        block_count = sample["block_tables"].shape[1]
        block_tables[batch_idx, :block_count] = sample["block_tables"][0] + block_offset
        block_offset += sample_blocks

    max_context_len = max(kv_seq_lens)
    mask_dtype = sample_inputs_list[0]["mask_dtype"]
    mask = generator._generate_mask(
        test_config["mask_type"],
        int(sum(q_seq_lens)),
        max_context_len,
        q_seq_lens,
        kv_seq_lens,
        mask_dtype,
        test_config["num_heads"],
        test_config.get("mla_v_dim", 0) > 0,
    )

    return {
        "query": query,
        "key_cache": key_cache,
        "value_cache": value_cache,
        "block_tables": block_tables,
        "q_seq_lens": np.array(q_seq_lens, dtype=np.int32),
        "kv_seq_lens": np.array(kv_seq_lens, dtype=np.int32),
        "mask": mask,
        "k_descale": None,
        "v_descale": None,
        "k_offset": None,
        "v_offset": None,
        "k_scale_per_head": None,
        "v_scale_per_head": None,
        "p_scale": None,
        "q_dtype": sample_inputs_list[0]["q_dtype"],
        "kv_dtype": sample_inputs_list[0]["kv_dtype"],
        "mask_dtype": mask_dtype,
    }


def _split_pa_output_by_sample(output: np.ndarray, q_seq_lens: np.ndarray) -> list:
    """Split flattened token output back into per-sample slices."""
    outputs = []
    start = 0
    for q_len in q_seq_lens.tolist():
        end = start + int(q_len)
        outputs.append(output[start:end])
        start = end
    return outputs


def _assert_pa_batch_invariant_random_mix(generator: PagedAttentionDataGenerator, test_config: dict,
                                          q_seq_len: int, kv_seq_len_candidates: list,
                                          random_batch_cases: int, run_mode: int):
    """Mix four random request lengths into each batch and compare with single-sample baselines."""
    assert test_config.get("mla_v_dim", 0) > 0, "Batch invariance test must run the MLA path."
    assert test_config.get("kv_heads") == 1, "MLA path requires kv_heads == 1."
    assert test_config.get("input_layout", INPUT_LAYOUT_BSND) == INPUT_LAYOUT_BSND
    assert len(kv_seq_len_candidates) >= 4, "Need at least four candidate lengths to build mixed batches."

    inconsistent_cases = []
    kv_seq_len_candidates = np.asarray(kv_seq_len_candidates, dtype=np.int32)
    for case_idx in range(int(random_batch_cases)):
        selected_kv_seq_lens = generator.rng.choice(
            kv_seq_len_candidates,
            size=4,
            replace=False,
        )
        selected_kv_seq_lens = selected_kv_seq_lens[generator.rng.permutation(4)].tolist()

        sample_inputs_list = []
        baseline_outputs = []
        for kv_seq_len in selected_kv_seq_lens:
            sample_inputs = _build_pa_sample_inputs(
                generator,
                test_config,
                q_seq_len,
                int(kv_seq_len),
            )
            sample_inputs_list.append(sample_inputs)
            baseline_outputs.append(
                _execute_paged_attention_with_inputs(
                    test_config,
                    run_mode,
                    sample_inputs,
                ).asnumpy().astype(np.float32)
            )

        batch_inputs = _assemble_pa_batch_inputs(
            generator,
            test_config,
            sample_inputs_list,
        )
        batch_output = _execute_paged_attention_with_inputs(
            test_config,
            run_mode,
            batch_inputs,
        ).asnumpy()
        split_outputs = _split_pa_output_by_sample(batch_output, batch_inputs["q_seq_lens"])

        for sample_idx, (kv_seq_len, output, baseline_output) in enumerate(
            zip(selected_kv_seq_lens, split_outputs, baseline_outputs)
        ):
            output_fp32 = output.astype(np.float32)
            max_diff = float(np.max(np.abs(output_fp32 - baseline_output)))
            if not np.array_equal(output_fp32, baseline_output):
                inconsistent_cases.append(
                    (
                        case_idx,
                        tuple(int(length) for length in selected_kv_seq_lens),
                        sample_idx,
                        int(kv_seq_len),
                        max_diff,
                    )
                )

    assert not inconsistent_cases, (
        "Batch invariance mismatches found for mixed-length cases: "
        + ", ".join(
            "case#{case_idx} lens={batch_kv_seq_lens} sample={sample_idx} "
            "kv_seq_len={kv_seq_len}(max_diff={max_diff:.6e})".format(
                case_idx=case_idx,
                batch_kv_seq_lens=list(batch_kv_seq_lens),
                sample_idx=sample_idx,
                kv_seq_len=kv_seq_len,
                max_diff=max_diff,
            )
            for case_idx, batch_kv_seq_lens, sample_idx, kv_seq_len, max_diff in inconsistent_cases
        )
    )


@_paged_attention_test
def test_pa_mla_batch_invariant():
    """
    Feature: PagedAttention + MLA - batch invariance
    Description: Randomly choose four different request lengths, assemble them
    into one mixed batch, and compare each batched result against its own
    single-request baseline.
    Expectation: Graph mode executes on the MLA non-MTP path; every sample in
    the mixed-length batch should match its single-sample baseline.
    """
    generator = PagedAttentionDataGenerator(7700)
    test_config = {
        "num_heads": 128,
        "kv_heads": 1,
        "head_size": 576,
        "block_size": 128,
        "q_dtype": ms.bfloat16,
        "kv_dtype": ms.bfloat16,
        "mask_type": MASK_UNDEFINED,
        "qk_scale": 1.0 / math.sqrt(576),
        "mla_v_dim": 512,
    }
    q_seq_len = 1
    kv_seq_lens = [255, 256, 257, 511, 512, 513, 2048, 4096, 8192]
    random_batch_cases = 16

    _assert_pa_batch_invariant_random_mix(
        generator,
        test_config,
        q_seq_len,
        kv_seq_lens,
        random_batch_cases,
        context.GRAPH_MODE,
    )
