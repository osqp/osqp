# Define where the QDLDL source code is located, and allow advanced users to provide their own copy
set(QDLDL_SRC_LOCATION "${OSQP_ALGEBRA_ROOT}/../ext/qdldl" CACHE STRING "Location of QDLDL source code")
mark_as_advanced(QDLDL_SRC_LOCATION)

# Make QDLDL use the same types as OSQP
set(QDLDL_FLOAT ${OSQP_USE_FLOAT} CACHE BOOL "QDLDL Float type")
set(QDLDL_LONG ${OSQP_USE_LONG} CACHE BOOL "QDLDL Integer type")

# We only want the object library, so turn off the other library products
set(QDLDL_BUILD_STATIC_LIB OFF CACHE BOOL "Build QDLDL static library")
set(QDLDL_BUILD_SHARED_LIB OFF CACHE BOOL "Build QDLDL shared library")

message(STATUS "Configuring QDLDL solver")
list(APPEND CMAKE_MESSAGE_INDENT "  ")

if(NOT EXISTS ${QDLDL_SRC_LOCATION}/README.md)
    message(FATAL_ERROR
            "QDLDL not found in \"${QDLDL_SRC_LOCATION}\".\n \n"
            "Get the QDLDL source code using the following commands and then re-run CMake:\n"
            "    git submodule init"
            "    git submodule update"
            )
endif()

add_subdirectory(${QDLDL_SRC_LOCATION} ${CMAKE_CURRENT_BINARY_DIR}/ext/qdldl EXCLUDE_FROM_ALL)

list(POP_BACK CMAKE_MESSAGE_INDENT)

set_source_files_properties($<TARGET_OBJECTS:qdldlobject> PROPERTIES GENERATED 1)

file(
    GLOB
    AMD_SRC_FILES
    CONFIGURE_DEPENDS
    ${OSQP_ALGEBRA_ROOT}/_common/lin_sys/qdldl/amd/src/*.c
    ${OSQP_ALGEBRA_ROOT}/_common/lin_sys/qdldl/amd/include/*.h
    )

set( LIN_SYS_QDLDL_NON_EMBEDDED_SRC_FILES
     ${AMD_SRC_FILES}
     )

set( LIN_SYS_QDLDL_EMBEDDED_SRC_FILES
     ${OSQP_ALGEBRA_ROOT}/_common/kkt.h
     ${OSQP_ALGEBRA_ROOT}/_common/kkt.c
     ${OSQP_ALGEBRA_ROOT}/_common/lin_sys/qdldl/qdldl_interface.h
     ${OSQP_ALGEBRA_ROOT}/_common/lin_sys/qdldl/qdldl_interface.c
     )

set( LIN_SYS_QDLDL_SRC_FILES
     ${LIN_SYS_QDLDL_EMBEDDED_SRC_FILES}
     ${LIN_SYS_QDLDL_NON_EMBEDDED_SRC_FILES}
     )

set( LIN_SYS_QDLDL_INC_PATHS
     ${qdldl_include}
     ${OSQP_ALGEBRA_ROOT}/_common/
     ${OSQP_ALGEBRA_ROOT}/_common/lin_sys/qdldl/
     ${OSQP_ALGEBRA_ROOT}/_common/lin_sys/qdldl/amd/include
     ${qdldl_SOURCE_DIR}/include
     ${qdldl_BINARY_DIR}/include
     )
