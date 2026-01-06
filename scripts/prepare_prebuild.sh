#!/bin/bash
set -e

function prepare_ms_kernels_internal()
{
  # ms_kernels_internal
  if [ -d "prebuild/ms_kernels_internal/" ]; then
    rm -rf prebuild/ms_kernels_internal/ms_kernels_internal
  fi

  if [ -z "${MS_CUSTOM_INTERNAL_KERNEL_HOME}" ]; then
    echo "[INFO]:use default internal tar."
    type=aarch64
    uname -a | grep aarch64 &> /dev/null || type=x86_64
    tar -zxf prebuild/ms_kernels_internal/${type}/ms_kernels_internal.tar.gz -C prebuild/ms_kernels_internal
    tar -zxf prebuild/ms_kernels_internal/${type}/ms_kernels_dependency.tar.gz -C prebuild/ms_kernels_internal
    cp prebuild/ms_kernels_internal/ms_kernels_dependency/asdops prebuild/ms_kernels_internal/ms_kernels_internal/ -rf
    rm -rf prebuild/ms_kernels_internal/ms_kernels_dependency
  fi
}

build_cann_recipes_infer()
{
  cd 3rdparty/cann-recipes-infer/ops/ascendc
  echo "ASCEND_HOME_PATH: " ${ASCEND_HOME_PATH}
  if [ -z "$ASCEND_HOME_PATH" ]; then
    echo "build cann-recipes-infer with tools in /usr/local/Ascend/ascend-toolkit/latest"
    bash build.sh -c "ascend910b;ascend910_93" --disable-check-compatible -p "/usr/local/Ascend/ascend-toolkit/latest"
  else
    echo "build cann-recipes-infer with tools in ASCEND_HOME_PATH: " ${ASCEND_HOME_PATH}
    bash build.sh -c "ascend910b;ascend910_93" --disable-check-compatible
  fi

  cur_path=$PWD
  bash "output/CANN-custom_ops--linux.$(arch).run" --install-path=${cur_path}/output/
  cd -
}

prepare_ms_kernels_internal

git submodule update --init

build_cann_recipes_infer
