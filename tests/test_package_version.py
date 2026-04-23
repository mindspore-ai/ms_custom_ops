# Copyright 2026 Huawei Technologies Co., Ltd
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
"""Tests for package metadata."""

import subprocess
import sys
from pathlib import Path

from packaging.version import Version


ROOT_DIR = Path(__file__).resolve().parents[1]


def test_setup_version_uses_root_version_file():
    """The wheel version should be sourced from the repository version.txt."""
    expected_version = Version((ROOT_DIR / "version.txt").read_text(encoding="utf-8").strip()).public

    result = subprocess.run(
        [sys.executable, "setup.py", "--version"],
        cwd=ROOT_DIR,
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.strip().splitlines()[-1] == expected_version
