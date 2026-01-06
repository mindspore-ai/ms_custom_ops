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
Plugin Management Tests

This module contains tests for plugin loading, registration, and management.

ENVIRONMENT VARIABLES
=====================
- DEVICE_ID: Device ID to use for testing
  Example: export DEVICE_ID=0
  Default: 0

PREREQUISITES
=============
- MindSpore framework installed and configured
- ms_custom_ops plugin built and installed
- pytest for test execution
"""

import os
import logging
import pytest

import mindspore as ms
import ms_custom_ops

logger = logging.getLogger(__name__)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_plugin_existence():
    """
    Feature: test custom pass plugin existence
    Description: verify plugin file exists and has correct properties
    Expectation: plugin file exists with valid size and permissions
    """
    logger.info("=== Testing Plugin Existence ===")

    # Get plugin path from internal function
    # pylint: disable=protected-access
    plugin_path = ms_custom_ops._find_plugin()

    # Verify plugin file exists
    assert os.path.exists(plugin_path), f"Plugin file does not exist: {plugin_path}"
    logger.info("[OK] Plugin file exists: %s", plugin_path)

    # Verify plugin file size
    file_size = os.path.getsize(plugin_path)
    assert file_size > 0, f"Plugin file is empty: {file_size} bytes"
    logger.info("[OK] Plugin file size: %s bytes", file_size)

    # Verify plugin file permissions
    assert os.access(plugin_path, os.R_OK), f"Plugin file is not readable: {plugin_path}"
    logger.info("[OK] Plugin file is readable")

    # Verify plugin file extension
    assert plugin_path.endswith('.so'), f"Plugin file does not have .so extension: {plugin_path}"
    logger.info("[OK] Plugin file has correct extension")


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_plugin_registration():
    """
    Feature: test custom pass plugin registration
    Description: test plugin registration to different devices
    Expectation: plugin registers successfully to all device types
    """
    logger.info("=== Testing Plugin Registration ===")

    # Test registration for different devices
    devices = ["cpu", "gpu", "ascend", "all"]

    for device in devices:
        logger.info("Testing registration for device: %s", device)
        success = ms_custom_ops.register_custom_pass("ReplaceAddNFusionPass", backend=device)
        assert success, f"Plugin registration failed for device: {device}"
        logger.info("[OK] Registration successful for device: %s", device)

    # Test duplicate registration (should still succeed)
    logger.info("Testing duplicate registration")
    success = ms_custom_ops.register_custom_pass("ReplaceAddNFusionPass", backend="cpu")
    assert success, "Duplicate plugin registration failed"
    logger.info("[OK] Duplicate registration successful")


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_invalid_plugin_path():
    """
    Feature: test error handling for invalid plugin paths
    Description: test plugin registration with non-existent file
    Expectation: registration fails gracefully
    """
    logger.info("=== Testing Invalid Plugin Path ===")

    # Test with non-existent file
    invalid_path = "/non/existent/path/libpass.so"
    success = ms.graph.register_custom_pass("ReplaceAddNFusionPass", invalid_path, "cpu")
    assert not success, "Registration should fail for invalid path"
    logger.info("[OK] Registration correctly failed for invalid path")

    # Test with empty path
    success = ms.graph.register_custom_pass("ReplaceAddNFusionPass", "", "cpu")
    assert not success, "Registration should fail for empty path"
    logger.info("[OK] Registration correctly failed for empty path")


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_invalid_pass_name():
    """
    Feature: test error handling for invalid pass names
    Description: test plugin registration with non-existent pass name
    Expectation: registration fails gracefully
    """
    logger.info("=== Testing Invalid Pass Name ===")

    # Test with non-existent pass name
    success = ms_custom_ops.register_custom_pass("NonExistentPass", backend="cpu")
    assert not success, "Registration should fail for invalid pass name"
    logger.info("[OK] Registration correctly failed for invalid pass name")

    # Test with empty pass name
    success = ms_custom_ops.register_custom_pass("", backend="cpu")
    assert not success, "Registration should fail for empty pass name"
    logger.info("[OK] Registration correctly failed for empty pass name")


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_multi_device_registration():
    """
    Feature: test multiple device registration
    Description: test registering same pass to multiple devices
    Expectation: all registrations succeed independently
    """
    logger.info("=== Testing Multi-Device Registration ===")

    # Test registering to all devices
    devices = ["cpu", "gpu", "ascend"]
    results = {}

    for device in devices:
        success = ms_custom_ops.register_custom_pass("ReplaceAddNFusionPass", backend=device)
        results[device] = success
        logger.info("Registration for %s: %s", device, 'SUCCESS' if success else 'FAILED')

    # Verify at least CPU registration succeeded
    assert results["cpu"], "CPU registration should succeed"
    logger.info("[OK] Multi-device registration test completed")


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_multiple_pass_registration():
    """
    Feature: test multiple pass registration
    Description: test registering both ReplaceAddN and AddRmsNorm passes
    Expectation: both passes register successfully
    """
    logger.info("=== Testing Multiple Pass Registration ===")

    # ReplaceAddN should work on CPU
    success1 = ms_custom_ops.register_custom_pass("ReplaceAddNFusionPass", backend="cpu")
    assert success1, "ReplaceAddNFusionPass registration failed on CPU"
    logger.info("[OK] ReplaceAddNFusionPass registered on CPU")

    # AddRmsNorm should work on Ascend
    success2 = ms_custom_ops.register_custom_pass("AddRmsNormFusionPass", backend="ascend")
    assert success2, "AddRmsNormFusionPass registration failed on Ascend"
    logger.info("[OK] AddRmsNormFusionPass registered on Ascend")
    logger.info("[OK] Multiple pass registration test completed")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
