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

import math
import mindspore as ms
from mindspore import context

from test_custom_paged_attention import (
    MASK_UNDEFINED,
    PagedAttentionDataGenerator,
    _paged_attention_test,
)
from test_custom_paged_attention_batch_invariant import _assert_pa_batch_invariant_scan


@_paged_attention_test
def test_pa_mla_mtp_batch_invariant():
    """
    Feature: PagedAttention + MLA + MTP - batch invariance
    Description: Run one MLA MTP sample as baseline, then repeat it into sparse
    large batch sizes under several context length thresholds and report which
    combinations break invariance.
    Expectation: Graph mode executes on the MLA MTP path; all repeated batches
    should match the single-sample baseline.
    """
    generator = PagedAttentionDataGenerator(7800)
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
        "calc_type": 1,
    }
    q_seq_len = 4
    kv_seq_lens = [255, 256, 257, 511, 512, 513, 2048, 4096, 8192]
    batch_sizes = [200, 208, 216, 224, 232, 240, 248, 256]

    _assert_pa_batch_invariant_scan(
        generator,
        test_config,
        q_seq_len,
        kv_seq_lens,
        batch_sizes,
        context.GRAPH_MODE,
    )
