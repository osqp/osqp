
# --------------------
# AMD permutation code
#---------------------

if(NOT DEFINED EMBEDDED)
    target_sources(OSQPLIB
        PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/amd/include/amd_internal.h
        ${CMAKE_CURRENT_SOURCE_DIR}/amd/include/amd.h
        ${CMAKE_CURRENT_SOURCE_DIR}/amd/include/SuiteSparse_config.h
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
        ${CMAKE_CURRENT_SOURCE_DIR}/amd/src/SuiteSparse_config.c
    )

    target_include_directories(OSQPLIB
        PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/amd/include
    )

endif()


# --------------------
# OSQP QDLDL interface
#---------------------

target_sources(OSQPLIB
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/qdldl_interface.h
    ${CMAKE_CURRENT_SOURCE_DIR}/qdldl_interface.c
)

target_include_directories(OSQPLIB
    PRIVATE
	"${CMAKE_CURRENT_SOURCE_DIR}"
)

# --------------------
# QDLDL library target
#---------------------

add_subdirectory(qdldl_sources)

# Ignore the extra qdldl targets, we only use the qdldlobject target to get the object file
set_target_properties(qdldlstatic qdldl qdldl_example
    PROPERTIES
    EXCLUDE_FROM_ALL 1
    EXCLUDE_FROM_DEFAULT_BUILD 1
    )

set_source_files_properties($<TARGET_OBJECTS:qdldlobject> PROPERTIES GENERATED 1)


target_sources(OSQPLIB
    PRIVATE
    $<TARGET_OBJECTS:qdldlobject>
    )

# We need "qdldl_types.h" for "qdldl_interface.h"
target_include_directories(OSQPLIB
    PRIVATE
	"${CMAKE_CURRENT_SOURCE_DIR}/qdldl_sources/include"
    )
