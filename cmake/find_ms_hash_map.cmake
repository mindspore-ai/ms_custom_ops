# FindMindSporeHashMap.cmake
# Detect MindSpore HashMap type to stay consistent

message(STATUS "=== Start detecting MindSpore HashMap type ===")

# Method 1: locate MindSpore install directory and libraries
execute_process(
    COMMAND python3 -c "import mindspore; print(mindspore.__file__)"
    OUTPUT_VARIABLE MINDSPORE_MODULE_PATH
    ERROR_VARIABLE MINDSPORE_MODULE_ERROR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

message(STATUS "Python module path: ${MINDSPORE_MODULE_PATH}")
message(STATUS "Python error output: ${MINDSPORE_MODULE_ERROR}")

if(MINDSPORE_MODULE_PATH)
    # Derive library path from module path
    string(REPLACE "/__init__.py" "" MINDSPORE_ROOT "${MINDSPORE_MODULE_PATH}")
    string(REPLACE "/mindspore/__init__.py" "" MINDSPORE_ROOT "${MINDSPORE_ROOT}")

    message(STATUS "MindSpore root directory: ${MINDSPORE_ROOT}")

    # Locate MindSpore libraries - corrected path
    file(GLOB MINDSPORE_LIBS "${MINDSPORE_ROOT}/lib/libmindspore_*.so")

    message(STATUS "Discovered libraries: ${MINDSPORE_LIBS}")

    if(MINDSPORE_LIBS)
        # Select the main library to inspect (libmindspore_core.so usually contains core functionality)
        list(FIND MINDSPORE_LIBS "${MINDSPORE_ROOT}/lib/libmindspore_core.so" CORE_LIB_INDEX)
        message(STATUS "libmindspore_core.so index: ${CORE_LIB_INDEX}")

        if(CORE_LIB_INDEX GREATER_EQUAL 0)
            list(GET MINDSPORE_LIBS ${CORE_LIB_INDEX} MINDSPORE_LIB)
        else()
            list(GET MINDSPORE_LIBS 0 MINDSPORE_LIB)
        endif()

        message(STATUS "Selected library: ${MINDSPORE_LIB}")

        # Check whether the MindSpore library contains robin_hood symbols
        message(STATUS "Start checking robin_hood symbol...")

        # Approach 1: run the command via bash
        execute_process(
            COMMAND bash -c "strings '${MINDSPORE_LIB}' | grep -i robin_hood | head -1"
            OUTPUT_VARIABLE ROBIN_HOOD_CHECK
            ERROR_VARIABLE ROBIN_HOOD_CHECK_ERROR
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )

        message(STATUS "robin_hood check result: '${ROBIN_HOOD_CHECK}'")
        message(STATUS "robin_hood check error: '${ROBIN_HOOD_CHECK_ERROR}'")

        # If the bash approach fails, try reading the file directly
        if(NOT ROBIN_HOOD_CHECK)
            message(STATUS "Bash approach failed, trying to read file directly...")
            file(READ ${MINDSPORE_LIB} LIB_CONTENT)
            string(FIND "${LIB_CONTENT}" "robin_hood" ROBIN_HOOD_POS)
            if(ROBIN_HOOD_POS GREATER_EQUAL 0)
                set(ROBIN_HOOD_CHECK "found_in_file")
                message(STATUS "Found robin_hood within the file content")
            endif()
        endif()

        if(ROBIN_HOOD_CHECK)
            message(STATUS "MindSpore uses robin_hood::unordered_map")

            # Check if robin_hood.h is available
            set(ROBIN_HOOD_HEADER "${MINDSPORE_ROOT}/include/third_party/robin_hood_hashing/include/robin_hood.h")
            if(EXISTS "${ROBIN_HOOD_HEADER}")
                message(STATUS "Found robin_hood.h: ${ROBIN_HOOD_HEADER}")
                add_compile_definitions(ENABLE_FAST_HASH_TABLE=1)
                add_compile_definitions(HASHMAP_TYPE="robin_hood")
                # Add robin_hood include path
                set(ROBIN_HOOD_INCLUDE_ROOT "${MINDSPORE_ROOT}/include/third_party/robin_hood_hashing")
                include_directories("${ROBIN_HOOD_INCLUDE_ROOT}")
                message(STATUS "Using fast hash table (robin_hood) for ms_custom_ops to match MindSpore")
            else()
                message(WARNING "robin_hood.h not found, falling back to std::unordered_map")
                add_compile_definitions(HASHMAP_TYPE="std")
                message(STATUS "Using std::unordered_map for ms_custom_ops (robin_hood.h missing)")
            endif()
        else()
            message(STATUS "MindSpore uses std::unordered_map")
            add_compile_definitions(HASHMAP_TYPE="std")
            message(STATUS "Using standard hash table (std::unordered_map) for ms_custom_ops to match MindSpore")
        endif()
    else()
        message(WARNING "MindSpore library not found in ${MINDSPORE_ROOT}/mindspore/lib/")
        set(MINDSPORE_LIB "")
    endif()
else()
    message(WARNING "Could not find MindSpore Python module")
    set(MINDSPORE_LIB "")

    # Method 2: detect via environment variable
    if(DEFINED ENV{MINDSPORE_HASHMAP_TYPE})
        if($ENV{MINDSPORE_HASHMAP_TYPE} STREQUAL "robin_hood")
            message(STATUS "Using robin_hood from environment variable")
            add_compile_definitions(ENABLE_FAST_HASH_TABLE=1)
            add_compile_definitions(HASHMAP_TYPE="robin_hood")
        else()
            message(STATUS "Using std::unordered_map from environment variable")
            add_compile_definitions(HASHMAP_TYPE="std")
        endif()
    else()
        # If the MindSpore library is missing, fall back to defaults
        message(WARNING "MindSpore library not found and env variable unset; using default HashMap implementation")
        add_compile_definitions(HASHMAP_TYPE="std")
    endif()
endif()

message(STATUS "=== MindSpore HashMap detection finished ===")

# Add compile-time log definition (used for conditional compilation)
add_compile_definitions(LOG_HASHMAP_TYPE)
