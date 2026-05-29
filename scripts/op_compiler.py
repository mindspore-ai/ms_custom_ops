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
"""setup package for custom compiler tool"""
import argparse
import os
import re
import subprocess
import logging
import shutil

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


SOC_VERSION_MAP = {
    "ascend910a": "ascend910",
    "ascend910proa": "ascend910",
    "ascned910premiuma": "ascend910",
    "ascend910prob": "ascend910",
    "ascend910b": "ascend910b",
    "ascend910b1": "ascend910b",
    "ascend910b2": "ascend910b",
    "ascend910b2c": "ascend910b",
    "ascend910b3": "ascend910b",
    "ascend910b4": "ascend910b",
    "ascend910b4-1": "ascend910b",
    "ascend910_93": "ascend910_93",
    "ascend910_9391": "ascend910_93",
    "ascend910_9392": "ascend910_93",
    "ascend910_9381": "ascend910_93",
    "ascend910_9382": "ascend910_93",
    "ascend910_9372": "ascend910_93",
    "ascend910_9362": "ascend910_93",
    "ascend310p": "ascend310p",
    "ascend310p1": "ascend310p",
    "ascend310p3": "ascend310p",
    "ascend310p5": "ascend310p",
    "ascend310p7": "ascend310p",
    "ascend310p3vir01": "ascend310p",
    "ascend310p3vir02": "ascend310p",
    "ascend310p3vir04": "ascend310p",
    "ascend310p3vir08": "ascend310p",
    "ascend310b": "ascend310b",
    "ascend310b1": "ascend310b",
    "ascend310b2": "ascend310b",
    "ascend310b3": "ascend310b",
    "ascend310b4": "ascend310b",
}


def resolve_ascend_cmake_dir(cann_package_path):
    """Resolve the complete CANN open-project cmake template directory."""
    candidate_dirs = (
        os.path.join(cann_package_path, "tools", "ascend_project", "cmake"),
        os.path.join(cann_package_path, "tools", "op_project_templates", "ascendc", "customize", "cmake"),
    )
    required_files = (
        os.path.join("util", "ascendc_impl_build.py"),
        os.path.join("util", "gen_version_info.sh"),
        "makeself.cmake",
    )
    for candidate in candidate_dirs:
        if all(os.path.isfile(os.path.join(candidate, item)) for item in required_files):
            return candidate

    tools_dir = os.path.join(cann_package_path, "tools")
    for dirpath, _, filenames in os.walk(tools_dir):
        if "ascendc_impl_build.py" in filenames and os.path.basename(dirpath) == "util":
            candidate = os.path.dirname(dirpath)
            if all(os.path.isfile(os.path.join(candidate, item)) for item in required_files):
                return candidate
    raise ValueError(f"Cannot find aclnn cmake template under CANN path: {cann_package_path}")


def prepare_gcc_toolchain_cmake_dir(cann_package_path, build_root, gcc_toolchain):
    """Copy and patch the CANN cmake template with an explicit GCC toolchain."""
    source_cmake_dir = resolve_ascend_cmake_dir(cann_package_path)
    patched_cmake_dir = os.path.join(build_root, "patched_ascend_cmake")
    if os.path.exists(patched_cmake_dir):
        shutil.rmtree(patched_cmake_dir)
    os.makedirs(build_root, exist_ok=True)
    shutil.copytree(source_cmake_dir, patched_cmake_dir, symlinks=False)

    impl_build_path = os.path.join(patched_cmake_dir, "util", "ascendc_impl_build.py")
    if not os.path.isfile(impl_build_path):
        raise ValueError(f"Patched AscendC cmake template is invalid, missing {impl_build_path}")

    with open(impl_build_path, encoding="utf-8") as f:
        content = f.read()

    anchor = 'options.append("-I" + tikcpp_path)\n'
    injected = f'    options.append("--gcc-toolchain={gcc_toolchain}")\n'
    if injected not in content:
        if anchor not in content:
            raise ValueError(f"Failed to patch gcc toolchain, anchor not found in {impl_build_path}")
        content = content.replace(anchor, anchor + injected, 1)

    with open(impl_build_path, "w", encoding="utf-8") as f:
        f.write(content)

    logger.info("AscendC cmake template source: %s", source_cmake_dir)
    logger.info("AscendC cmake template patched: %s", patched_cmake_dir)

    return patched_cmake_dir


def get_config():
    """get config from user"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--common_dirs", type=str, required=True)
    parser.add_argument("--op_dirs", type=str, required=True)
    parser.add_argument("--build_type", type=str, default="Release")
    parser.add_argument("--build_path", type=str, default="")
    parser.add_argument("--soc_version", type=str, default="")
    parser.add_argument("--ascend_cann_package_path", type=str, default="")
    parser.add_argument("--vendor_name", type=str, default="customize")
    parser.add_argument("--install_path", type=str, default="")
    parser.add_argument("-c", "--clear", action="store_true")
    parser.add_argument("-i", "--install", action="store_true")
    parser.add_argument("--force_clean", action="store_true",
                        help="Force clean build directory before compilation (disable incremental build)")
    return parser.parse_args()


class CustomOPCompiler():
    """
    Custom Operator Offline Compilation
    """

    def __init__(self, args):
        self.args = args
        self.aclnn_src_path = self.args.build_path
        self.op_dirs = re.split(r"[;, ]", self.args.op_dirs)
        self.soc_version = self.args.soc_version.replace(",", ";")

    def check_args(self):
        """check config"""
        for op_dir in self.op_dirs:
            if not os.path.isdir(op_dir):
                raise ValueError(
                    f"Config error! op directory [{op_dir}] is not exist, "
                    f"please check your set --op_dirs")

        if self.soc_version != "":
            soc_version_list = re.split(r"[;]", self.soc_version)
            for soc_version in soc_version_list:
                if soc_version.lower() not in SOC_VERSION_MAP:
                    raise ValueError(
                        f"Config error! Unsupported soc version(s): {soc_version}! "
                        f"Please check your set --soc_version and use ';' to separate multiple soc_versions. "
                        f"Supported soc version : {SOC_VERSION_MAP.keys()}.")

        if self.args.ascend_cann_package_path != "":
            if not os.path.isdir(self.args.ascend_cann_package_path):
                raise ValueError(
                    f"Config error! ascend CANN package path [{self.args.ascend_cann_package_path}] is not valid path, "
                    f"please check your set --ascend_cann_package_path")

        if self.args.install or self.args.install_path != "":
            if self.args.install_path == "":
                opp_path = os.environ.get('ASCEND_OPP_PATH')
                if opp_path is None:
                    raise ValueError(
                        "Config error! Can not find install path, please set install path by --install_path")
                self.args.install_path = opp_path

            os.makedirs(self.args.install_path, exist_ok=True)

    def exec_shell_command(self, command, stdout=None):
        """run exec shell"""
        try:
            capture_output = stdout is None
            result = subprocess.run(command,
                                    stdout=subprocess.PIPE if capture_output else stdout,
                                    stderr=subprocess.STDOUT,
                                    shell=False,
                                    text=True,
                                    check=True)
            if capture_output and result.stdout:
                print(result.stdout, end="")
        except FileNotFoundError as e:
            logger.error("Command not found: %s", e)
            raise RuntimeError(f'Command not found: {e}') from e
        except subprocess.CalledProcessError as e:
            output = e.stdout or e.output or ""
            if output:
                print(output, end="")
                tail = "\n".join(output.splitlines()[-200:])
                logger.error("Command output tail:\n%s", tail)
            logger.error("Run %s Command failed with return code %s", command, e.returncode)
            raise RuntimeError(
                f"Run {command} Command failed with return code {e.returncode}"
            ) from e
        return result

    def init_config(self):
        """initialize config"""
        if self.args.ascend_cann_package_path == "":
            self.args.ascend_cann_package_path = os.environ.get('ASCEND_HOME_PATH',
                                                                "/usr/local/Ascend/ascend-toolkit/latest")

    def compile_custom_op(self):
        """compile custom operator"""
        # 构建 build.sh 参数
        build_args = f"-c '{self.soc_version}' --disable-check-compatible"
        if self.args.force_clean:
            build_args += " --clean"
            logger.info("Force clean enabled, will clean build directory before compilation")
        else:
            logger.info("Incremental build enabled, keeping existing build artifacts")

        gcc_toolchain = os.getenv("GCC_TOOLCHAIN")
        if gcc_toolchain:
            build_root = os.path.realpath(os.path.join(self.aclnn_src_path, "../../../build/ascendc"))
            patched_cmake_dir = prepare_gcc_toolchain_cmake_dir(
                self.args.ascend_cann_package_path, build_root, gcc_toolchain
            )
            build_args += f" --ascend_cmake_dir '{patched_cmake_dir}'"
            logger.info("Using patched aclnn cmake dir: %s", patched_cmake_dir)

        if self.args.ascend_cann_package_path != "":
            cann_package_path = self.args.ascend_cann_package_path
            setenv_path = os.path.join(cann_package_path, "bin", "setenv.bash")
            bash_cmd = (
                f"source {setenv_path} > /dev/null 2>&1 && "
                f"export LD_LIBRARY_PATH={cann_package_path}/lib64:$LD_LIBRARY_PATH && "
                f"cd {self.aclnn_src_path} && "
                f"bash build.sh {build_args}"
            )
        else:
            bash_cmd = (
                f"cd {self.aclnn_src_path} && "
                f"bash build.sh {build_args}"
            )
        args = ['bash', '-c', bash_cmd]
        self.exec_shell_command(args)
        logger.info("Custom operator compiled successfully!")

    def install_custom_op(self):
        """install custom run"""
        if self.args.install or self.args.install_path != "":
            logger.info("Install custom opp run in %s", self.args.install_path)
            os.environ['ASCEND_CUSTOM_OPP_PATH'] = self.args.install_path
            run_path = []
            build_out_path = os.path.join(self.aclnn_src_path, "../../../build/ascendc/output")
            for item in os.listdir(build_out_path):
                if item.split('.')[-1] == "run":
                    run_path.append(os.path.join(build_out_path, item))
            if not run_path:
                raise RuntimeError(f"There is no custom run in {build_out_path}")
            self.exec_shell_command(['bash', run_path[0]])
            logger.info("Install custom run opp successfully!")
            logger.info(
                "Please set [source ASCEND_CUSTOM_OPP_PATH=%s/vendors/%s:$ASCEND_CUSTOM_OPP_PATH] to "
                "make the custom operator effective in the current path.",
                self.args.install_path,
                self.args.vendor_name,
            )


    def compile(self):
        """compile op"""
        self.check_args()
        self.init_config()
        self.compile_custom_op()
        self.install_custom_op()


if __name__ == "__main__":
    config = get_config()
    custom_op_compiler = CustomOPCompiler(config)
    custom_op_compiler.compile()
