include(FetchContent)

message(STATUS "Fetching/configuring QDLDL solver")
list(APPEND CMAKE_MESSAGE_INDENT "  ")

# Corresponds to 0.1.6 release of QDLDL
FetchContent_Declare(
  qdldl
  GIT_REPOSITORY https://github.com/osqp/qdldl.git
  GIT_TAG 29d140419a3bec20d860052d73ba2be927faf5a1
  SOURCE_DIR ${OSQP_ALGEBRA_ROOT}/_common/lin_sys/qdldl/qdldl_sources)
FetchContent_GetProperties(qdldl)

if(NOT qdldl_POPULATED)
  FetchContent_Populate(qdldl)

  # Make QDLDL use the same types as OSQP
  set(QDLDL_FLOAT ${OSQP_USE_FLOAT} CACHE BOOL "QDLDL Float type")
  set(QDLDL_LONG ${OSQP_USE_LONG} CACHE BOOL "QDLDL Integer type")

  # We only want the object library, so turn off the other library products
  set(QDLDL_BUILD_STATIC_LIB OFF CACHE BOOL "Build QDLDL static library")
  set(QDLDL_BUILD_SHARED_LIB OFF CACHE BOOL "Build QDLDL shared library")

  # We don't actually want to build anything from here by default
  add_subdirectory(${qdldl_SOURCE_DIR} ${qdldl_BINARY_DIR} EXCLUDE_FROM_ALL)
endif()

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
     ${OSQP_ALGEBRA_ROOT}/_common/lin_sys/qdldl/qdldl_sources/include
     )
