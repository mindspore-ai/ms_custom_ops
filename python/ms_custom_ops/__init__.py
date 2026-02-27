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

"""MS Custom Ops Package

This module provides the public API for MindSpore Custom Operations.
It includes functions for registering custom operators and managing
their lifecycle within the MindSpore framework.

Example usage:
    >>> import ms_custom_ops
    >>> ms_custom_ops.register_op("my_custom_op", ...)
"""

import os
import glob
import logging
import mindspore


def _load_op_api_lib():
    """Load op api library."""
    current_path = os.path.dirname(os.path.abspath(__file__))
    env_path = os.path.join(current_path, "vendors", "customize")
    cann_recipes_infer_path = os.path.join(current_path, "3rdparty/cann-recipes-infer/vendors/customize")
    env_path = env_path + ":" + cann_recipes_infer_path
    origin_env_path = os.getenv("ASCEND_CUSTOM_OPP_PATH")
    if origin_env_path:
        env_path = env_path + ":" + origin_env_path
    os.environ["ASCEND_CUSTOM_OPP_PATH"] = env_path
    # pylint: disable=import-outside-toplevel
    from .ms_custom_ops import load_op_api_library
    load_op_api_library()


def _find_plugin():
    """Find plugin .so file path automatically."""
    current_path = os.path.dirname(os.path.abspath(__file__))
    pattern = os.path.join(current_path, "ms_custom_ops.cpython-*.so")
    matches = glob.glob(pattern)
    if not matches:
        raise FileNotFoundError(f"Plugin .so not found in {current_path}")
    return matches[0]


def register_custom_pass(pass_name, backend="ascend"):
    """Register custom pass without exposing plugin path logic.

    Args:
        pass_name: Name of the custom pass to register
        backend: Target backend (default: "ascend")

    Returns:
        bool: True if registration succeeded
    """
    plugin_path = _find_plugin()
    return mindspore.graph.register_custom_pass(pass_name, plugin_path, backend)


_load_op_api_lib()
register_custom_pass("ConvertTupleInputToDynamicInput")

# pylint: disable=wrong-import-position
from .ms_custom_ops import *
# Import generated ops interfaces
try:
    from .gen_ops_def import *
except ImportError:
    pass  # Generated files may not exist during development

try:
    from .gen_ops_prim import *
except ImportError:
    pass  # Generated files may not exist during development

def _warn_missing_batchinvariant_package():
    """Warn if ops-batchinvariant package directory is not found in known locations."""
    found = False

    # 1) Check ASCEND_CUSTOM_OPP_PATH entries
    opp_path = os.getenv("ASCEND_CUSTOM_OPP_PATH", "")
    for raw_path in opp_path.split(":"):
        path = raw_path.strip()
        if not path:
            continue
        real_path = os.path.realpath(path)
        if os.path.isdir(real_path) and os.path.basename(real_path) == "custom_batchinvariant":
            found = True
            break

    # 2) Check CANN install directories under opp/vendors
    if not found:
        cann_roots = []
        for env_key in (
            "ASCEND_HOME_PATH",
            "ASCEND_CANN_PACKAGE_PATH",
            "ASCEND_TOOLKIT_HOME",
            "LOCAL_ASCEND",
        ):
            env_val = os.getenv(env_key, "")
            env_val = env_val.strip()
            if env_val:
                cann_roots.append(env_val)
        for root in cann_roots:
            root_path = os.path.realpath(root)
            candidate = os.path.join(root_path, "opp", "vendors", "custom_batchinvariant")
            if os.path.isdir(candidate):
                found = True
                break

    # 3) Check ASCEND_OPP_PATH directly if set
    if not found:
        opp_root = os.getenv("ASCEND_OPP_PATH", "").strip()
        if opp_root:
            opp_root = os.path.realpath(opp_root)
            candidate = os.path.join(opp_root, "vendors", "custom_batchinvariant")
            if os.path.isdir(candidate):
                found = True

    if not found:
        message = (
            "[ms_custom_ops] Optional package 'ops-batchinvariant-dev' was not detected from "
            "ASCEND_CUSTOM_OPP_PATH or CANN opp/vendors. Operators [matmul_batch_invariant, "
            "reduce_sum_batch_invariant] are unavailable. Please install from "
            "https://gitcode.com/cann/ops-batchinvariant-dev and follow the docs at "
            "https://gitcode.com/cann/ops-batchinvariant-dev/tree/master/docs. "
            "If the package is already installed, ensure ASCEND_CUSTOM_OPP_PATH points to its "
            "custom_batchinvariant package directory or the CANN install path is set correctly.",
        )
        print(message)
        logging.getLogger(__name__).warning(message)


_warn_missing_batchinvariant_package()

# Expose generated interfaces
__all__ = ['register_custom_pass']

# Add ops from gen_ops_def if available
try:
    import ms_custom_ops.gen_ops_def as gen_ops_def  # pylint: disable=consider-using-from-import
    if hasattr(gen_ops_def, '__all__'):
        __all__.extend(gen_ops_def.__all__)
    else:
        # If no __all__ defined, add all public functions
        __all__.extend([name for name in dir(gen_ops_def) if not name.startswith('_')])
except ImportError:
    pass

# Add ops from gen_ops_prim if available
try:
    import ms_custom_ops.gen_ops_prim as gen_ops_prim  # pylint: disable=consider-using-from-import
    if hasattr(gen_ops_prim, '__all__'):
        __all__.extend(gen_ops_prim.__all__)
    else:
        # If no __all__ defined, add all public functions
        __all__.extend([name for name in dir(gen_ops_prim) if not name.startswith('_')])
except ImportError:
    pass
