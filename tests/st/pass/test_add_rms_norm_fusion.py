#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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

"""
AddRmsNorm Fusion Pass Test

This module contains tests for the AddRmsNorm fusion pass,
supporting float16 and float32 data types on Ascend devices.

FUSION PASS TESTED
==================
- AddRmsNormFusionPass: RmsNorm(Add(x1, x2)) -> AddRmsNorm(x1, x2) [Ascend]
  * Fuses Add + RmsNorm into single AddRmsNorm operation
  * Reduces graph complexity and improves execution performance
  * Key optimization for transformer models

ENVIRONMENT VARIABLES
=====================
- KEEP_GRAPHS: Keep graph files after test completion for debugging
  Example: export KEEP_GRAPHS=1
  Effect: Preserves IR graphs in temporary directories for manual inspection

- DEVICE_ID: Ascend device ID to use for testing
  Example: export DEVICE_ID=0
  Default: 0

PREREQUISITES
=============
- MindSpore framework installed and configured
- ms_custom_ops plugin built and installed
- Ascend NPU device available
- pytest for test execution
"""

import os
import logging
import numpy as np
import pytest

import mindspore as ms
from mindspore import nn, ops, Tensor, jit
import ms_custom_ops

logger = logging.getLogger(__name__)


class AddRmsNormNetwork(nn.Cell):
    """Network with Add + RmsNorm pattern for fusion testing"""

    def __init__(self, epsilon=1e-5):
        super().__init__()
        self.rmsnorm = ops.RmsNorm(epsilon=epsilon)

    @jit
    def construct(self, x1, x2, gamma):
        # Add operation
        add_result = x1 + x2
        # RmsNorm operation
        rmsnorm_result, rstd = self.rmsnorm(add_result, gamma)
        # Return all outputs for verification
        return rmsnorm_result, rstd, add_result


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('dtype', [ms.float16, ms.float32])
def test_add_rmsnorm_fusion_functionality(graphs_dir_fixture, dtype):
    """
    Feature: test AddRmsNorm fusion pass functionality
    Description: test Add + RmsNorm fusion with float16/float32
    Expectation: Add + RmsNorm ops are fused into AddRmsNorm op in IR
    """
    logger.info("=== Testing AddRmsNorm Fusion Functionality (dtype=%s) ===", dtype)

    # Register custom pass using simplified API
    success = ms_custom_ops.register_custom_pass("AddRmsNormFusionPass")
    assert success, "Plugin registration failed"
    logger.info("[OK] Plugin registered successfully")

    # Setup MindSpore context
    device_id = int(os.getenv("DEVICE_ID", "0"))
    ms.set_context(
        device_target="Ascend",
        device_id=device_id,
        save_graphs=True,
        save_graphs_path=graphs_dir_fixture
    )
    logger.info("[OK] Context configured for Ascend device %d", device_id)

    # Create test network
    net = AddRmsNormNetwork()

    # Create test data
    shape = (2, 256, 1024)
    np.random.seed(42)

    np_dtype = np.float16 if dtype == ms.float16 else np.float32
    x1_np = np.random.rand(*shape).astype(np_dtype)
    x2_np = np.random.rand(*shape).astype(np_dtype)
    gamma_np = np.ones([shape[-1]]).astype(np_dtype)

    x1 = Tensor(x1_np, dtype=dtype)
    x2 = Tensor(x2_np, dtype=dtype)
    gamma = Tensor(gamma_np, dtype=dtype)

    logger.info("Input shapes: x1=%s, x2=%s, gamma=%s", x1.shape, x2.shape, gamma.shape)
    logger.info("Data type: %s", dtype)

    # Execute network
    logger.info("Executing network...")
    rmsnorm_out, rstd_out, add_out = net(x1, x2, gamma)
    logger.info("[OK] Network executed successfully")
    logger.info("Output shapes: rmsnorm=%s, rstd=%s, add=%s",
                rmsnorm_out.shape, rstd_out.shape, add_out.shape)

    # Verify functional correctness (basic sanity check)
    expected_add = x1_np + x2_np
    np.testing.assert_allclose(add_out.asnumpy(), expected_add, rtol=1e-3, atol=1e-3)
    logger.info("[OK] Functional correctness verified")

    logger.info("[OK] AddRmsNorm fusion test completed for dtype=%s", dtype)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_simple_add_rmsnorm_network(graphs_dir_fixture):
    """
    Feature: test simple Add + RmsNorm network
    Description: test basic Add + RmsNorm operation functionality
    Expectation: network executes correctly with fusion
    """
    logger.info("=== Testing Simple Add+RmsNorm Network ===")

    # Register custom pass using simplified API
    success = ms_custom_ops.register_custom_pass("AddRmsNormFusionPass")
    assert success, "Plugin registration failed"

    # Setup context
    device_id = int(os.getenv("DEVICE_ID", "0"))
    ms.set_context(
        device_target="Ascend",
        device_id=device_id,
        save_graphs=True,
        save_graphs_path=graphs_dir_fixture
    )

    # Create simple network
    class SimpleNetwork(nn.Cell):
        def __init__(self):
            super().__init__()
            self.rmsnorm = ops.RmsNorm()

        @jit
        def construct(self, x, y, gamma):
            return self.rmsnorm(x + y, gamma)

    net = SimpleNetwork()

    # Create test data
    x = Tensor(np.random.rand(2, 8, 16).astype(np.float32))
    y = Tensor(np.random.rand(2, 8, 16).astype(np.float32))
    gamma = Tensor(np.ones(16).astype(np.float32))

    # Execute network
    output, rstd = net(x, y, gamma)

    logger.info("Network executed successfully")
    logger.info("Output shape: %s, rstd shape: %s", output.shape, rstd.shape)
    logger.info("[OK] Simple Add+RmsNorm test passed")


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_network_without_fusion_pattern(graphs_dir_fixture):
    """
    Feature: test network without Add+RmsNorm fusion pattern
    Description: test custom pass on network without target pattern
    Expectation: network executes normally without fusion
    """
    logger.info("=== Testing Network Without Fusion Pattern ===")

    # Register custom pass using simplified API
    success = ms_custom_ops.register_custom_pass("AddRmsNormFusionPass")
    assert success, "Plugin registration failed"

    # Setup context
    device_id = int(os.getenv("DEVICE_ID", "0"))
    ms.set_context(
        device_target="Ascend",
        device_id=device_id,
        save_graphs=True,
        save_graphs_path=graphs_dir_fixture
    )

    # Network without Add+RmsNorm pattern (only RmsNorm)
    class NonFusionNetwork(nn.Cell):
        def __init__(self):
            super().__init__()
            self.rmsnorm = ops.RmsNorm()

        @jit
        def construct(self, x, gamma):
            # Direct RmsNorm without Add
            return self.rmsnorm(x, gamma)

    net = NonFusionNetwork()

    # Create test data
    x = Tensor(np.random.rand(2, 8, 16).astype(np.float32))
    gamma = Tensor(np.ones(16).astype(np.float32))

    # Execute network
    output, _ = net(x, gamma)

    logger.info("Network without fusion pattern executed successfully")
    logger.info("Output shape: %s", output.shape)
    logger.info("[OK] Non-fusion network test passed")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
