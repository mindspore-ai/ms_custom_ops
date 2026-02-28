#!/usr/bin/env python3
# encoding: utf-8
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
"""setup package for ms_custom_ops."""

import glob
import logging
import os
import platform
import sys
import shutil
import multiprocessing
import subprocess
from typing import List
from pathlib import Path
from setuptools import find_packages, setup
from setuptools.command.build_ext import build_ext
from setuptools import Extension


ROOT_DIR = os.path.dirname(__file__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

package_name = "ms_custom_ops"


# =============================================================================
# 编译选项解析
# =============================================================================
class BuildOptions:
    """Build options container for ms_custom_ops.
    
    All build options are controlled via environment variables:
        DEBUG_MODE          Set to 'on' to enable debug build
        FORCE_CLEAN         Set to 'on' to force clean build
        ASCEND_HOME_PATH    Path to Ascend toolkit
        CMAKE_THREAD_NUM    Number of threads for compilation
        SOC_VERSION         Target SoC version
        OP_DIRS             Operator directories to compile
    """

    def __init__(self):
        # 从环境变量读取编译选项
        self.debug_mode = os.getenv("DEBUG_MODE", "").lower() == "on"
        self.force_clean = os.getenv("FORCE_CLEAN", "").lower() == "on"

    def print_options(self):
        """Print current build options."""
        separator = "=" * 60
        logger.info(separator)
        logger.info("MS Custom Ops Build Options")
        logger.info(separator)
        logger.info("  Debug Mode:       %s", "ON" if self.debug_mode else "OFF")
        logger.info("  Force Clean:      %s", "ON" if self.force_clean else "OFF")
        logger.info("  Build Type:       %s", "Debug" if self.debug_mode else "Release")
        logger.info("  ASCEND_HOME_PATH: %s", os.environ.get("ASCEND_HOME_PATH", "(default)"))
        logger.info(separator)
        logger.info("Environment Variables:")
        logger.info("  DEBUG_MODE=%s", os.getenv("DEBUG_MODE", ""))
        logger.info("  FORCE_CLEAN=%s", os.getenv("FORCE_CLEAN", ""))
        logger.info("  CMAKE_THREAD_NUM=%s", os.getenv("CMAKE_THREAD_NUM", "(auto)"))
        logger.info("  SOC_VERSION=%s", os.getenv("SOC_VERSION", "(default)"))
        logger.info(separator)


# 全局编译选项实例
build_options = BuildOptions()

if not sys.platform.startswith("linux"):
    logger.warning(
        "ms_custom_ops only supports Linux platform."
        "Building on %s, "
        "so ms_custom_ops may not be able to run correctly",
        sys.platform,
    )


def get_path(*filepath) -> str:
    return os.path.join(ROOT_DIR, *filepath)


def read_readme() -> str:
    """Read the README file if present."""
    p = get_path("README.md")
    if os.path.isfile(p):
        with open(get_path("README.md"), encoding="utf-8") as f:
            return f.read()
    else:
        return ""


def get_requirements() -> List[str]:
    """Get Python package dependencies from requirements.txt."""

    def _read_requirements(filename: str) -> List[str]:
        requirements_path = get_path(filename)
        if not os.path.exists(requirements_path):
            return []

        with open(requirements_path, encoding='utf-8') as f:
            requirements = f.read().strip().split("\n")
        resolved_requirements = []
        for line in requirements:
            if line.startswith("-r "):
                resolved_requirements += _read_requirements(line.split()[1])
            elif line.startswith("--"):
                continue
            elif "http" in line:
                continue
            elif line.strip() == "":
                continue
            else:
                resolved_requirements.append(line)
        return resolved_requirements

    requirements = _read_requirements("requirements.txt")
    return requirements


def write_commit_id():
    """Write the current Git commit information to a file.

    This function retrieves the current Git branch name and the latest commit log,
    then writes this information to a file for version tracking purposes.

    The commit information is written to:
        ./python/ms_custom_ops/.commit_id

    If Git is not available or the commands fail, a warning message is logged
    and placeholder text is written to the file.

    Returns:
        None
    """
    commit_info = ""
    try:
        commit_info += subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"]
        ).decode("utf-8")
        commit_info += subprocess.check_output(
            ["git", "log", "--abbrev-commit", "-1"]
        ).decode("utf-8")
    except subprocess.CalledProcessError:
        logger.warning(
            "Can't get commit id information. Please make sure git is available."
        )
        commit_info = "git is not available while building."

    with open("./python/ms_custom_ops/.commit_id", "w", encoding='utf-8') as f:
        f.write(commit_info)


def get_version():
    """Get version from version.txt or use default."""
    version_path = Path("ms_custom_ops") / "version.txt"
    if version_path.exists():
        return version_path.read_text().strip()

    return "0.1.0"


version = get_version()


def _get_ascend_home_path():
    """Get the Ascend home installation path."""
    if "ASCEND_HOME_PATH" in os.environ:
        return os.environ["ASCEND_HOME_PATH"]

    # Check default paths for compatibility
    defaults = ["/usr/local/Ascend/cann", "/usr/local/Ascend/ascend-toolkit/latest"]
    for path in defaults:
        if os.path.exists(path):
            return path

    return defaults[0]


def _get_ascend_env_path():
    """Locate the set_env.sh file for Ascend environment setup.

    This function searches for 'set_env.sh' in the Ascend installation directory and its parent.
    Returns the full file path if found, otherwise raises a ValueError.

    Returns:
        str: Path to the set_env.sh script.

    Raises:
        ValueError: If set_env.sh cannot be found in expected locations.
    """
    ascend_home = _get_ascend_home_path()
    candidates = [
        os.path.join(ascend_home, "set_env.sh"),
        os.path.join(ascend_home, "..", "set_env.sh"),
    ]

    for candidate in candidates:
        env_script_path = os.path.realpath(candidate)
        if os.path.exists(env_script_path):
            return env_script_path

    raise ValueError(
        f"The file 'set_env.sh' is not found in '{ascend_home}' or its parent directory. "
        "Please make sure environment variable 'ASCEND_HOME_PATH' is set correctly."
    )


def generate_docs():
    """Generate YAML documentation from Markdown sources."""
    logger.info("Generating documentation...")
    doc_generator_script = os.path.join(ROOT_DIR, "scripts", "doc_generator.py")

    if not os.path.exists(doc_generator_script):
        logger.warning(
            "Documentation generator script not found: %s", doc_generator_script
        )
        return

    try:
        # Run the documentation generator
        result = subprocess.run(
            [sys.executable, doc_generator_script],
            cwd=ROOT_DIR,
            capture_output=True,
            text=True,
            check=False,
        )

        if result.returncode == 0:
            logger.info("Documentation generated successfully")
            if result.stdout:
                logger.info("Generator output: %s", result.stdout)
        else:
            logger.warning(
                "Documentation generation failed with exit code %s",
                result.returncode,
            )
            if result.stderr:
                logger.warning("Generator error: %s", result.stderr)
    except Exception as e:  # pylint: disable=W0703
        logger.warning("Failed to run documentation generator: %s", e)


class CustomBuildExt(build_ext):
    """Custom build extension for MS Custom Ops.

    This class extends the standard setuptools build_ext command to
    include custom build steps for MS Custom Ops, such as generating
    documentation and preparing prebuilt dependencies.

    Attributes:
        ROOT_DIR (str): The root directory of the project.
    """

    ROOT_DIR = os.path.abspath(os.path.dirname(__file__))

    def run(self):
        """Override run method to include documentation generation."""
        # Generate documentation before building extensions
        generate_docs()

        # Continue with normal build process
        super().run()

    def build_extension(self, ext):
        if ext.name == "ms_custom_ops":
            self.build_ms_custom_ops(ext)
        else:
            raise ValueError(f"Unknown extension name: {ext.name}")

    def _setup_build_environment(self, ext):
        """Set up build environment variables and paths."""
        self.ext_name = ext.name
        self.so_name = self.ext_name + ".so"
        logger.info("Building %s ...", self.so_name)

        self.build_ops_dir = os.path.join(ROOT_DIR, "build", "ms_custom_ops")
        os.makedirs(self.build_ops_dir, exist_ok=True)

        # 使用全局编译选项
        self.build_type = "Debug" if build_options.debug_mode else "Release"
        self.force_clean = build_options.force_clean

        self.ascend_home_path = _get_ascend_home_path()
        self.env_script_path = _get_ascend_env_path()
        self.build_extension_dir = os.path.join(
            self.build_ops_dir, "kernel_meta", self.ext_name
        )

        self.dst_so_path = self.get_ext_fullpath(ext.name)
        self.dst_dir = os.path.dirname(self.dst_so_path)
        self.package_path = os.path.join(self.dst_dir, package_name)
        os.makedirs(self.package_path, exist_ok=True)

        # Also prepare the Python package directory for generated files
        self.python_package_path = os.path.join(ROOT_DIR, "python", package_name)
        os.makedirs(self.python_package_path, exist_ok=True)

        available_cores = multiprocessing.cpu_count()
        if os.getenv("CMAKE_THREAD_NUM", None):
            self.compile_cores = int(os.getenv("CMAKE_THREAD_NUM"))
        else:
            # Use a conservative default to reduce cc1plus OOM risk on heavy TU builds.
            self.compile_cores = max(1, available_cores // 4)

        logger.info(
            "Available CPU cores: %s, using %s cores for compilation",
            available_cores,
            self.compile_cores,
        )

    def _copy_generated_python_files(self):
        """Copy generated Python files to Python package directory."""
        auto_generate_dir = os.path.join(
            self.build_extension_dir, self.ext_name + "_auto_generate"
        )
        if os.path.exists(auto_generate_dir):
            generated_files = ["gen_ops_def.py", "gen_ops_prim.py"]
            for gen_file in generated_files:
                src_gen_path = os.path.join(auto_generate_dir, gen_file)
                if os.path.exists(src_gen_path):
                    dst_gen_path = os.path.join(self.package_path, gen_file)
                    shutil.copy(src_gen_path, dst_gen_path)
                    replace_cmd = [
                        "sed",
                        "-i",
                        r"s/import ms_custom_ops/from . import ms_custom_ops/g",
                        dst_gen_path,
                    ]
                    try:
                        subprocess.run(
                            replace_cmd, cwd=self.ROOT_DIR, text=True, shell=False, check=False,
                        )
                    except subprocess.CalledProcessError as e:
                        raise RuntimeError(f"Failed to exec command {replace_cmd}: {e}") from e
                    logger.info("Copied %s to %s", gen_file, dst_gen_path)
                else:
                    logger.warning("Generated file not found: %s", src_gen_path)
        else:
            logger.warning("Auto-generate directory not found: %s", auto_generate_dir)

    def _copy_prebuild_files(self):
        """Copy prebuild/ms_kernels_internal files to the package."""
        if "MS_CUSTOM_INTERNAL_KERNEL_HOME" in os.environ:
            prebuild_path = os.environ["MS_CUSTOM_INTERNAL_KERNEL_HOME"]
        else:
            prebuild_path = os.path.join(
                ROOT_DIR, "prebuild", "ms_kernels_internal/ms_kernels_internal"
            )

        if os.path.exists(prebuild_path):
            package_prebuild_path = os.path.join(
                self.package_path, "prebuild", "ms_kernels_internal"
            )
            os.makedirs(package_prebuild_path, exist_ok=True)

            # Walk through the prebuild directory and copy all files
            for root, _, filenames in os.walk(prebuild_path):
                for filename in filenames:
                    src_file_path = os.path.join(root, filename)
                    # Get the relative path from the prebuild/ms_kernels_internal directory
                    rel_path = os.path.relpath(src_file_path, prebuild_path)
                    dst_file_path = os.path.join(package_prebuild_path, rel_path)

                    # Create directory for the destination file
                    dst_dir_path = os.path.dirname(dst_file_path)
                    os.makedirs(dst_dir_path, exist_ok=True)

                    # Copy the file
                    shutil.copy2(src_file_path, dst_file_path)
                    logger.info("Copied %s to %s", src_file_path, dst_file_path)

    def _get_submodule_commit_id(self, submodule_path):
        """Get the current commit id of a git submodule."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=submodule_path,
                capture_output=True,
                text=True,
                check=True,
            )
            return result.stdout.strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.warning("Failed to get commit id for submodule: %s", submodule_path)
            return None

    def _build_cann_recipes_infer(self):
        """Build cann-recipes-infer if not already built (incremental build support).
        
        This function checks:
        1. If the output directory exists and contains files
        2. If the submodule commit id has changed since last build
        
        If output exists and commit id matches, skip the build.
        If commit id changed or no previous build record, rebuild.
        """
        cri_base_dir = os.path.join(ROOT_DIR, "3rdparty/cann-recipes-infer")
        cri_src_dir = os.path.join(cri_base_dir, "ops/ascendc")
        cri_output_dir = os.path.join(cri_src_dir, "output/vendors")
        commit_id_file = os.path.join(cri_src_dir, "output/.build_commit_id")

        # 获取当前 submodule 的 commit id
        current_commit_id = self._get_submodule_commit_id(cri_base_dir)

        # 检查是否需要重新编译
        need_rebuild = True
        if os.path.exists(cri_output_dir) and os.listdir(cri_output_dir):
            if os.path.exists(commit_id_file):
                with open(commit_id_file, 'r', encoding='utf-8') as f:
                    last_commit_id = f.read().strip()
                if current_commit_id and last_commit_id == current_commit_id:
                    logger.info(
                        "cann-recipes-infer output exists and commit id unchanged (%s), skipping build",
                        current_commit_id[:8] if current_commit_id else "unknown"
                    )
                    need_rebuild = False
                else:
                    logger.info(
                        "cann-recipes-infer commit id changed: %s -> %s, rebuilding...",
                        last_commit_id[:8] if last_commit_id else "unknown",
                        current_commit_id[:8] if current_commit_id else "unknown"
                    )
                    # 清理旧的编译输出
                    shutil.rmtree(os.path.join(cri_src_dir, "output"), ignore_errors=True)
                    shutil.rmtree(os.path.join(cri_src_dir, "build"), ignore_errors=True)
            else:
                logger.info(
                    "cann-recipes-infer output exists but no commit id record, skipping build"
                )
                need_rebuild = False

        if not need_rebuild:
            return

        logger.info("Building cann-recipes-infer...")

        # Build cann-recipes-infer using its build.sh
        build_script = os.path.join(cri_src_dir, "build.sh")
        if not os.path.exists(build_script):
            logger.warning(
                "cann-recipes-infer build script not found: %s", build_script
            )
            return

        try:
            # 构建编译命令，支持多种芯片版本
            build_cmd = (
                f"source {self.env_script_path} && "
                f"cd {cri_src_dir} && "
                f"bash build.sh -c 'ascend910b;ascend910_93' --disable-check-compatible"
            )
            result = subprocess.run(
                build_cmd,
                cwd=cri_src_dir,
                text=True,
                shell=True,
                capture_output=False,
                check=False,
            )
            if result.returncode != 0:
                logger.error("cann-recipes-infer build failed with exit code %s", result.returncode)
                raise RuntimeError(
                    f"cann-recipes-infer build failed with exit code {result.returncode}"
                )

            # 运行 .run 文件进行安装
            arch = platform.machine()
            run_file_pattern = os.path.join(cri_src_dir, f"output/CANN-custom_ops--linux.{arch}.run")
            run_files = glob.glob(run_file_pattern)
            if not run_files:
                # 尝试通配符匹配
                run_files = glob.glob(os.path.join(cri_src_dir, "output/*.run"))

            if run_files:
                run_file = run_files[0]
                install_path = os.path.join(cri_src_dir, "output")
                install_cmd = f"bash {run_file} --install-path={install_path}"
                logger.info("Installing cann-recipes-infer: %s", install_cmd)
                result = subprocess.run(
                    install_cmd,
                    cwd=cri_src_dir,
                    text=True,
                    shell=True,
                    capture_output=False,
                    check=False,
                )
                if result.returncode != 0:
                    logger.warning("cann-recipes-infer install returned code %s", result.returncode)
            else:
                logger.warning("No .run file found in cann-recipes-infer output directory")

            # 保存当前 commit id 到文件，用于下次增量编译检测
            if current_commit_id:
                os.makedirs(os.path.dirname(commit_id_file), exist_ok=True)
                with open(commit_id_file, 'w', encoding='utf-8') as f:
                    f.write(current_commit_id)
                logger.info("Saved build commit id: %s", current_commit_id[:8])

            logger.info("cann-recipes-infer built successfully")
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to build cann-recipes-infer: {e}") from e

    def _copy_cann_recipes_infer(self):
        """Copy cann-recipes-infer output to package directory."""
        cri_path = os.path.join(ROOT_DIR, "3rdparty/cann-recipes-infer/ops/ascendc/output/vendors")

        if not os.path.exists(cri_path):
            logger.warning(
                "cann-recipes-infer output not found at %s, skipping copy", cri_path
            )
            return

        dst_path = os.path.join(self.package_path, "3rdparty", "cann-recipes-infer/vendors")
        if os.path.exists(dst_path):
            shutil.rmtree(dst_path)
        shutil.copytree(cri_path, dst_path)
        logger.info("Copied cann-recipes-infer output to %s", dst_path)


    def _run_cmake_build(self):
        """Run CMake configuration and build commands."""
        # 使用实例变量中的强制清理选项
        force_clean_option = "ON" if self.force_clean else "OFF"

        # Combine all cmake commands into one string
        cmake_cmd = (
            f"source {self.env_script_path} && "
            f"cmake -S ./ -B {self.build_ops_dir}"
            f"  -DCMAKE_BUILD_TYPE={self.build_type}"
            f"  -DCMAKE_INSTALL_PREFIX={os.path.join(self.build_ops_dir, 'install')}"
            f"  -DCMAKE_BUILD_PATH={self.build_ops_dir}"
            f"  -DBUILD_EXTENSION_DIR={self.build_extension_dir}"
            f"  -DASCENDC_INSTALL_PATH={self.package_path}"
            f"  -DMS_EXTENSION_NAME={self.ext_name}"
            f"  -DASCEND_CANN_PACKAGE_PATH={self.ascend_home_path}"
            f"  -DCUSTOM_OP_COMPILE_JOBS={self.compile_cores}"
            f"  -DFORCE_CLEAN={force_clean_option} && "
            f"cmake --build {self.build_ops_dir} -j{self.compile_cores}"
        )

        try:
            # Run the combined cmake command
            result = subprocess.run(
                cmake_cmd,
                cwd=self.ROOT_DIR,
                text=True,
                shell=True,
                capture_output=False,
                check=False,
            )
            if result.returncode != 0:
                logger.info("CMake commands failed:")
                logger.info(result.stdout)  # Print standard output
                logger.info(result.stderr)  # Print error output
                raise RuntimeError(
                    f"Combined CMake commands failed with exit code {result.returncode}"
                )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to build {self.so_name}: {e}") from e

    def _copy_built_files(self):
        """Copy the generated .so file to the target directory."""
        src_so_path = os.path.join(self.build_extension_dir, self.so_name)
        if os.path.exists(self.dst_so_path):
            os.remove(self.dst_so_path)
        so_name = os.path.basename(self.dst_so_path)
        shutil.copy(src_so_path, os.path.join(self.package_path, so_name))
        logger.info("Copied %s to %s", so_name, self.dst_so_path)

    def build_ms_custom_ops(self, ext):
        """Build and prepare the MS Custom Ops extension module.

        This method handles the complete build process for custom operations,
        including preparing prebuilt dependencies and compiling the extension.

        Args:
            ext: The extension object being built.

        Returns:
            None

        Raises:
            RuntimeError: If any step of the build process fails.
        """
        # 打印编译选项
        build_options.print_options()

        def _prepare_prebuild():
            bash_cmd = "bash scripts/prepare_prebuild.sh"
            try:
                result = subprocess.run(
                    bash_cmd,
                    cwd=self.ROOT_DIR,
                    text=True,
                    shell=True,
                    capture_output=False,
                    check=False,
                )
                if result.returncode != 0:
                    logger.error("call prepare_prebuild.sh failed:")
                    logger.error(result.stdout)  # Print standard output
                    logger.error(result.stderr)  # Print error output
                    raise RuntimeError(
                        f"call prepare_prebuild.sh failed with exit code {result.returncode}"
                    )
            except subprocess.CalledProcessError as e:
                raise RuntimeError(f"Failed to call prepare_prebuild.sh: {e}") from e

        _prepare_prebuild()
        self._setup_build_environment(ext)
        self._build_cann_recipes_infer()  # Build cann-recipes-infer with incremental support
        self._run_cmake_build()
        self._copy_built_files()
        self._copy_generated_python_files()
        self._copy_prebuild_files()
        self._copy_cann_recipes_infer()


write_commit_id()

package_data = {
    "": ["*.so", "lib/*.so", ".commit_id"],
    "ms_custom_ops": ["gen_ops_def.py", "gen_ops_prim.py"],
}


def _get_ext_modules():
    ext_modules = []
    if os.path.exists(_get_ascend_home_path()):
        # sources are specified in CMakeLists.txt
        ext_modules.append(Extension("ms_custom_ops", sources=[]))
    return ext_modules


setup(
    name=package_name,
    version=version,
    author="MindSpore Team",
    license="Apache 2.0",
    description=("MindSpore Custom Operations for Ascend NPU"),
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://gitee.com/mindspore/ms_custom_ops",
    project_urls={
        "Homepage": "https://gitee.com/mindspore/ms_custom_ops",
    },
    classifiers=[
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: Apache Software License",
        "Intended Audience :: Developers",
        "Intended Audience :: Information Technology",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    packages=find_packages(where="python"),
    package_dir={"": "python"},
    python_requires=">=3.9",
    install_requires=get_requirements(),
    cmdclass={"build_ext": CustomBuildExt},
    ext_modules=_get_ext_modules(),
    include_package_data=True,
    package_data=package_data,
)
