include(FetchContent)

message(STATUS "Fetching/configuring QDLDL solver")
list(APPEND CMAKE_MESSAGE_INDENT "  ")

FetchContent_Declare(
  qdldl
  GIT_REPOSITORY https://github.com/osqp/qdldl.git
  GIT_TAG 07ff30a3eedee4857bbbeb0778f54a86b1ea78f9
  SOURCE_DIR ${OSQP_ALGEBRA_ROOT}/_common/lin_sys/direct/qdldl_sources)
FetchContent_GetProperties(qdldl)

if(NOT qdldl_POPULATED)
  FetchContent_Populate(qdldl)
  # We don't actually want to build anything from here
  add_subdirectory(${qdldl_SOURCE_DIR} ${qdldl_BINARY_DIR} EXCLUDE_FROM_ALL)
endif()

list(POP_BACK CMAKE_MESSAGE_INDENT)

set_source_files_properties($<TARGET_OBJECTS:qdldlobject> PROPERTIES GENERATED 1)

file(
    GLOB
    AMD_SRC_FILES
    CONFIGURE_DEPENDS
    ${OSQP_ALGEBRA_ROOT}/_common/lin_sys/direct/amd/src/*.c
    ${OSQP_ALGEBRA_ROOT}/_common/lin_sys/direct/amd/include/*.h
    )

set( LIN_SYS_QDLDL_NON_EMBEDDED_SRC_FILES
     ${AMD_SRC_FILES}
     )

set( LIN_SYS_QDLDL_EMBEDDED_SRC_FILES
     ${OSQP_ALGEBRA_ROOT}/_common/kkt.h
     ${OSQP_ALGEBRA_ROOT}/_common/kkt.c
     ${OSQP_ALGEBRA_ROOT}/_common/lin_sys/direct/qdldl_interface.h
     ${OSQP_ALGEBRA_ROOT}/_common/lin_sys/direct/qdldl_interface.c
     )

set( LIN_SYS_QDLDL_SRC_FILES
     ${LIN_SYS_QDLDL_EMBEDDED_SRC_FILES}
     ${LIN_SYS_QDLDL_NON_EMBEDDED_SRC_FILES}
     )

set( LIN_SYS_QDLDL_INC_PATHS
     ${qdldl_include}
     ${OSQP_ALGEBRA_ROOT}/_common/
     ${OSQP_ALGEBRA_ROOT}/_common/lin_sys/direct/
     ${OSQP_ALGEBRA_ROOT}/_common/lin_sys/direct/amd/include
     ${OSQP_ALGEBRA_ROOT}/_common/lin_sys/direct/qdldl_sources/include
     )
