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
Custom pass test configuration and fixtures.

This module provides fixtures for custom pass testing.
"""

import os
import shutil
import logging
from pathlib import Path
import pytest

logger = logging.getLogger(__name__)


@pytest.fixture(scope="function", name="graphs_dir_fixture")
def graphs_temp_dir():
    """Create temporary directory for graph dumps.

    Environment variables:
    - KEEP_GRAPHS=1: Keep graph files after test completion

    Returns:
        str: Path to graphs directory
    """
    import time
    import threading

    test_name = f"test_{int(time.time())}_{threading.current_thread().ident}"
    temp_base = Path("/tmp") / "ms_custom_ops_tests"
    graphs_dir = temp_base / test_name / "graphs"
    graphs_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Graphs directory: %s", graphs_dir)

    # Check if graphs should be kept for debugging
    keep_graphs = os.getenv("KEEP_GRAPHS", "").lower() in ("1", "true", "yes")
    if keep_graphs:
        logger.info("KEEP_GRAPHS=1, graph files will be preserved for debugging")

    yield str(graphs_dir)

    # Cleanup graphs directory after test function (unless KEEP_GRAPHS=1)
    if not keep_graphs:
        try:
            if graphs_dir.exists():
                test_temp_dir = graphs_dir.parent
                shutil.rmtree(test_temp_dir)
                logger.debug("Cleaned up test directory: %s", test_temp_dir)
        except (OSError, PermissionError) as e:
            logger.warning("Cleanup warning for graphs: %s", e)
    else:
        logger.info("Graphs preserved at: %s", graphs_dir)
