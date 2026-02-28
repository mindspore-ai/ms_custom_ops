#!/bin/bash
set -e

usage()
{
  echo "Usage:"
  echo "bash build.sh [-d] [-v] [-p] [-j[n]] [-c] [-h]"
  echo ""
  echo "Options:"
  echo "    -d Debug mode"
  echo "    -v Soc version. (Default: Ascend910B,Ascend310P)"
  echo "    -p The absolute path to the directory of the operator that needs to be compiled, use ',' to split. (Default: all operators)"
  echo "    -j[n] Set the threads when building (Default: quarter available cpus)"
  echo "    -c Force clean build (disable incremental build). By default, incremental build is enabled."
  echo "    -h Help"
  echo ""
  echo "Environment Variables (set by this script):"
  echo "    DEBUG_MODE          'on' for debug build"
  echo "    FORCE_CLEAN         'on' for force clean build"
  echo "    SOC_VERSION         Target SoC version"
  echo "    CMAKE_THREAD_NUM    Number of compilation threads"
  echo "    OP_DIRS             Operator directories to compile"
}

# check and set options
process_options()
{
  # Process the options
  while getopts 'dv:p:j:ch' opt
  do
    case "${opt}" in
      d)
        export DEBUG_MODE="on" ;;
      v)
        export SOC_VERSION="$OPTARG" ;;
      p)
        export OP_DIRS="$OPTARG" ;;
      j)
        export CMAKE_THREAD_NUM=$OPTARG ;;
      c)
        export FORCE_CLEAN="on" ;;
      h)
        usage
        exit 0;;
      *)
        echo "Unknown option ${opt}!"
        usage
        exit 1
    esac
  done
}

process_options "$@"

echo "Start build."

# 增量编译：只有在强制清理模式下才删除 build 和 dist 目录
if [ "${FORCE_CLEAN}" == "on" ]; then
    echo "Force clean mode enabled, removing build and dist directories..."
    rm -rf ./build
    rm -rf ./dist
    python setup.py clean --all
else
    echo "Incremental build mode enabled, keeping existing build artifacts..."
    mkdir -p ./dist
fi

python setup.py bdist_wheel
echo "Finish build."
