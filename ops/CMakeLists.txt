# =============================================================================
# Collect Source Files from Ops Directories
# =============================================================================

add_subdirectory(c_api)
add_subdirectory(ascendc)
add_subdirectory(framework)

set(OPS_SRC_FILES ${C_API_SRC_FILES} ${ASCENDC_SRC_FILES} ${FRAMEWORK_SRC_FILES} PARENT_SCOPE)