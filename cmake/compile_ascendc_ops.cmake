# =============================================================================
# Compile AscendC Ops
# =============================================================================

find_package(Python3 REQUIRED COMPONENTS Interpreter)

if(NOT DEFINED ASCENDC_OP_DIRS)
    message(FATAL_ERROR "ASCENDC_OP_DIRS must be set before including this file")
endif()

if(NOT DEFINED OP_COMPILER_SCRIPT)
    message(FATAL_ERROR "OP_COMPILER_SCRIPT must be set before including this file")
endif()
set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Cmake build type")
set(CMAKE_BUILD_PATH "" CACHE STRING "Cmake build path")
set(ACLNN_SRC_DIR ${CMAKE_SOURCE_DIR}/ops/ascendc/aclnn_src)

# 增量编译选项：设置为 ON 时强制清理并重新编译，默认为 OFF 使用增量编译
option(FORCE_CLEAN "Force clean build directory before compilation" OFF)

if(DEFINED ENV{SOC_VERSION})
    set(SOC_VERSION $ENV{SOC_VERSION})
else()
    set(SOC_VERSION "Ascend910B,Ascend310P" CACHE STRING "SOC version")
endif()
set(VENDOR_NAME "customize" CACHE STRING "Vendor name")
set(ASCENDC_INSTALL_PATH "" CACHE PATH "Install path")
if(NOT ASCENDC_INSTALL_PATH)
    message(FATAL_ERROR "ASCENDC_INSTALL_PATH must be set. Use -DASCENDC_INSTALL_PATH=<path>")
endif()

set(CLEAR OFF CACHE BOOL "Clear build output")
set(INSTALL_OP OFF CACHE BOOL "Install custom op")
if(DEFINED ENV{ASCEND_HOME_PATH})
    set(ASCEND_CANN_PACKAGE_PATH $ENV{ASCEND_HOME_PATH})
    message(STATUS "Using ASCEND_HOME_PATH environment variable: ${ASCEND_HOME_PATH}")
else()
    set(ASCEND_CANN_PACKAGE_PATH /usr/local/Ascend/ascend-toolkit/latest)
endif()

add_custom_target(
    build_custom_op ALL
    COMMAND ${Python3_EXECUTABLE} ${OP_COMPILER_SCRIPT}
        --common_dirs="${ASCENDC_OP_COMMON_DIRS}"
        --op_dirs="${ASCENDC_OP_DIRS}"
        --build_path=${ACLNN_SRC_DIR}
        --build_type=${CMAKE_BUILD_TYPE}
        --soc_version="${SOC_VERSION}"
        --ascend_cann_package_path=${ASCEND_CANN_PACKAGE_PATH}
        --vendor_name=${VENDOR_NAME}
        --install_path=${ASCENDC_INSTALL_PATH}
        $<$<BOOL:${CLEAR}>:-c>
        $<$<BOOL:${INSTALL_OP}>:-i>
        $<$<BOOL:${FORCE_CLEAN}>:--force_clean>
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT "Building custom operator using setup.py"
)
