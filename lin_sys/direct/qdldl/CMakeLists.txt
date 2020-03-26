
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

# Get and setup the QDLDL library
# ----------------------------------------------
# Allow modifying the git properties for QDLDL
set( QDLDL_GIT_REPO "https://github.com/oxfordcontrol/qdldl.git" CACHE STRING "Git repository to pull the QDLDL sources from" )
set( QDLDL_GIT_TAG  "v0.1.4" CACHE STRING "Git tag in the QDLDL repository to pull" )


message( STATUS "Configuring QDLDL" )

# If the user defines QDLDL_REPO_DIR, then we don't fetch QDLDL and use what they provide
if( DEFINED QDLDL_REPO_DIR )
    message( STATUS " ++ Using existing QDLDL repository at ${QDLDL_REPO_DIR}" )

    set( FETCHCONTENT_SOURCE_DIR_QDLDL ${QDLDL_REPO_DIR} CACHE PATH "QDLDL path" )
else()
    message( STATUS " ++ Using QDLDL sources at git tag ${QDLDL_GIT_TAG} from ${QDLDL_GIT_REPO}" )
endif()

FetchContent_Declare(
  qdldl
  GIT_REPOSITORY ${QDLDL_GIT_REPO}
  GIT_TAG        ${QDLDL_GIT_TAG}
  SOURCE_DIR     ${PROJECT_SOURCE_DIR}/lin_sys/direct/qdldl/qdldl_sources
)

FetchContent_GetProperties( qdldl )

if( NOT qdldl_POPULATED )
    message( STATUS " ++ Populating QDLDL sources" )
    FetchContent_populate( qdldl )
    add_subdirectory( ${qdldl_SOURCE_DIR} ${qdldl_BINARY_DIR} )
else()
    message( STATUS " ++ QDLDL already populated" )
endif()

# --------------------
# Configure the QDLDL library target
# ---------------------

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
    $<TARGET_PROPERTY:qdldlobject,INCLUDE_DIRECTORIES>
    )
