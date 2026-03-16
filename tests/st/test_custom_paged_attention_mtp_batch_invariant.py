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
"""Batch invariance tests for MLA paged attention in MTP mode."""

import os
import math
import numpy as np
import mindspore as ms
from mindspore import context
from mindspore import Profiler
from mindspore.profiler import AicoreMetrics, ProfilerActivity, ProfilerLevel

from test_custom_paged_attention import (
    MASK_SPEC,
    PagedAttentionDataGenerator,
    _paged_attention_test,
)
from test_custom_paged_attention_batch_invariant import (
    _assemble_pa_batch_inputs,
    _build_pa_sample_inputs,
    _execute_paged_attention_with_inputs,
    _split_pa_output_by_sample,
)


def _profiling_enabled() -> bool:
    """Whether to enable optional MindSpore profiling for this test."""
    return os.environ.get("MS_CUSTOM_OPS_PA_MTP_PROFILE", "0") == "1"


def _profiling_output_path() -> str:
    """Profiling output directory."""
    return os.environ.get("MS_CUSTOM_OPS_PA_MTP_PROFILE_DIR", "./profiler_pa_mtp")


def _execute_paged_attention_with_inputs_maybe_profile(
    test_config: dict,
    run_mode: int,
    inputs_dict: dict,
    dynamic: bool = False,
):
    """Execute paged attention, optionally wrapping one call in MindSpore profiling."""
    if not _profiling_enabled():
        return _execute_paged_attention_with_inputs(test_config, run_mode, inputs_dict, dynamic)

    profiler = Profiler(
        start_profile=False,
        output_path=_profiling_output_path(),
        profiler_level=ProfilerLevel.Level2,
        activities=[ProfilerActivity.CPU, ProfilerActivity.NPU],
        aic_metrics=AicoreMetrics.AiCoreNone,
        data_simplification=False,
    )
    profiler.start()
    try:
        output = _execute_paged_attention_with_inputs(test_config, run_mode, inputs_dict, dynamic)
    finally:
        profiler.stop()
        profiler.analyse()
    return output


def _assert_pa_batch_invariant_scan_maybe_profile(generator: PagedAttentionDataGenerator, test_config: dict,
                                                  q_seq_len: int, kv_seq_lens: list,
                                                  batch_sizes: list, run_mode: int):
    """Scan context lengths and batch sizes, with optional profiling around each PA run."""
    inconsistent_cases = []
    for kv_seq_len in kv_seq_lens:
        sample_inputs = _build_pa_sample_inputs(generator, test_config, q_seq_len, kv_seq_len)
        baseline_output = _execute_paged_attention_with_inputs_maybe_profile(
            test_config,
            run_mode,
            sample_inputs,
        ).asnumpy().astype(np.float32)

        for batch_size in batch_sizes:
            batch_inputs = _assemble_pa_batch_inputs(
                generator,
                test_config,
                [sample_inputs] * int(batch_size),
            )
            batch_output = _execute_paged_attention_with_inputs_maybe_profile(
                test_config,
                run_mode,
                batch_inputs,
            ).asnumpy()
            split_outputs = _split_pa_output_by_sample(batch_output, batch_inputs["q_seq_lens"])
            batch_failed = False
            max_diff = 0.0
            for output in split_outputs:
                output_fp32 = output.astype(np.float32)
                diff = np.max(np.abs(output_fp32 - baseline_output))
                max_diff = max(max_diff, float(diff))
                if not np.array_equal(output_fp32, baseline_output):
                    batch_failed = True
                    break
            if batch_failed:
                inconsistent_cases.append((int(kv_seq_len), int(batch_size), max_diff))

    if inconsistent_cases:
        case_desc = ", ".join(
            f"(context_len={context_len}, batch_size={batch_size}, max_diff={max_diff})"
            for context_len, batch_size, max_diff in inconsistent_cases
        )
        raise AssertionError(f"Batch invariance failed for MLA MTP PagedAttention: {case_desc}")


@_paged_attention_test
def test_pa_mla_mtp_batch_invariant():
    """
    Feature: PagedAttention + MLA + MTP - batch invariance
    Description: Construct a look-ahead MLA MTP case close to the profiled
    deployment shape pattern, then repeat a single sample into sparse larger
    batches and report which combinations break invariance.
    Expectation: Graph mode executes on the
    internal_PagedMultiLatentAttentionMultiTokenPredictionMaskNdKernel path;
    all repeated batches should match the single-sample baseline.
    """
    generator = PagedAttentionDataGenerator(8192)
    test_config = {
        "num_heads": 128,
        "kv_heads": 1,
        "head_size": 576,
        "block_size": 128,
        "q_dtype": ms.bfloat16,
        "kv_dtype": ms.bfloat16,
        # MTP in the MindSpore/internal PA stack is driven by look-ahead mask
        # semantics plus q_seq_lens > 1. Use SPEC mask to match that path.
        "mask_type": MASK_SPEC,
        "qk_scale": 1.0 / math.sqrt(576),
        "mla_v_dim": 512,
        "calc_type": 1,
    }
    # Use a multi-token query window so the internal PA stack keeps q_seq_lens
    # and routes to the MLA MTP kernel instead of clearing q_seq_lens as a
    # single-token decode case.
    q_seq_len = 2
    # Keep the exact profiled long-context case as the anchor and add nearby
    # powers-of-two boundaries for sparse coverage.
    kv_seq_lens = [2048, 4096, 8192]
    batch_sizes = [200, 216, 232, 248, 256]

    _assert_pa_batch_invariant_scan_maybe_profile(
        generator,
        test_config,
        q_seq_len,
        kv_seq_lens,
        batch_sizes,
        context.GRAPH_MODE,
    )
