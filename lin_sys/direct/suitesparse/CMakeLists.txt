# Add suitesparse directory
set(suitesparse_includes
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/amd/include
    ${CMAKE_CURRENT_SOURCE_DIR}/ldl/include
    PARENT_SCOPE
)


# AMD source files
if(NOT DEFINED EMBEDDED)
set(
    amd_sources
    ${CMAKE_CURRENT_SOURCE_DIR}/amd/include/amd_internal.h
    ${CMAKE_CURRENT_SOURCE_DIR}/amd/include/amd.h
    ${CMAKE_CURRENT_SOURCE_DIR}/amd/src/amd_1.c
    ${CMAKE_CURRENT_SOURCE_DIR}/amd/src/amd_2.c
    ${CMAKE_CURRENT_SOURCE_DIR}/amd/src/amd_aat.c
    ${CMAKE_CURRENT_SOURCE_DIR}/amd/src/amd_control.c
    ${CMAKE_CURRENT_SOURCE_DIR}/amd/src/amd_defaults.c
    ${CMAKE_CURRENT_SOURCE_DIR}/amd/src/amd_info.c
    ${CMAKE_CURRENT_SOURCE_DIR}/amd/src/amd_order.c
    ${CMAKE_CURRENT_SOURCE_DIR}/amd/src/amd_post_tree.c
    ${CMAKE_CURRENT_SOURCE_DIR}/amd/src/amd_postorder.c
    ${CMAKE_CURRENT_SOURCE_DIR}/amd/src/amd_preprocess.c
    ${CMAKE_CURRENT_SOURCE_DIR}/amd/src/amd_valid.c
)
endif()

# LDL source files
set(
    ldl_sources
    ${CMAKE_CURRENT_SOURCE_DIR}/ldl/include/ldl.h
    ${CMAKE_CURRENT_SOURCE_DIR}/ldl/src/ldl.c
)


# Suitesparse sources
if(NOT DEFINED EMBEDDED)
    set(
        suitesparse_sources
        ${CMAKE_CURRENT_SOURCE_DIR}/SuiteSparse_config.h
        ${CMAKE_CURRENT_SOURCE_DIR}/SuiteSparse_config.c
        ${amd_sources}
        ${ldl_sources}
        ${CMAKE_CURRENT_SOURCE_DIR}/private.h
        ${CMAKE_CURRENT_SOURCE_DIR}/private.c
        PARENT_SCOPE
    )
else()
  # If embedded is enabled, do not use SuiteSparse_config
  set(
      suitesparse_sources
      ${ldl_sources}
      ${CMAKE_CURRENT_SOURCE_DIR}/private.h
      ${CMAKE_CURRENT_SOURCE_DIR}/private.c
      PARENT_SCOPE
  )
endif()
