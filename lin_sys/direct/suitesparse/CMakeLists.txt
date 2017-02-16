# Add suitesparse directory
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/amd/include
    ${CMAKE_CURRENT_SOURCE_DIR}/ldl/include
)


# AMD source files
set(
    amd_source_files
    amd/include/amd_internal.h
    amd/include/amd.h
    amd/src/amd_1.c
    amd/src/amd_2.c
    amd/src/amd_aat.c
    amd/src/amd_control.c
    amd/src/amd_defaults.c
    amd/src/amd_dump.c
    amd/src/amd_global.c
    amd/src/amd_info.c
    amd/src/amd_order.c
    amd/src/amd_post_tree.c
    amd/src/amd_postorder.c
    amd/src/amd_preprocess.c
    amd/src/amd_valid.c
)

# LDL source files
set(
    ldl_source_files
    ldl/include/ldl.h
    ldl/src/ldl.c
)


# Suitesparse sources
set(
    suitesparse_source_files
    private.c
    private.h
    SuiteSparse_config.c
    SuiteSparse_config.h
    ${amd_source_files}
    ${ldl_source_files}
)
