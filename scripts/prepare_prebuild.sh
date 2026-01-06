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

prepare_ms_kernels_internal

git submodule update --init

# NOTE: cann-recipes-infer 编译已移至 setup.py 中的 _build_cann_recipes_infer 方法
# 支持增量编译：检测到已有编译输出后会自动跳过，使用 FORCE_CLEAN_ASCENDC=on 环境变量可强制重新编译
