# =============================================================================
# Collect Source Files from pass Directories
# =============================================================================

file(GLOB_RECURSE BASE_SRC_FILES "${CMAKE_CURRENT_SOURCE_DIR}/*.cc")
set(PASS_SRC_FILES ${BASE_SRC_FILES} PARENT_SCOPE)