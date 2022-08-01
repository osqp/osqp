
find_path(GraphBLAS_INCLUDE_DIR
  NAMES GraphBLAS.h
  PATHS
    ${GraphBLAS_INCLUDE_DIRS}
    ${GraphBLAS_ROOT_DIR}/include
  PATH_SUFFIXES GraphBLAS
)

find_library(GraphBLAS_LIBRARY
  NAMES
    libgraphblas.so
  PATHS
    ${GraphBLAS_LIBRARY_DIRS}
    ${GraphBLAS_ROOT_DIR}/lib
    ${GraphBLAS_ROOT_DIR}/lib64
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(GraphBLAS
  FOUND_VAR GraphBLAS_FOUND
  REQUIRED_VARS
    GraphBLAS_LIBRARY
    GraphBLAS_INCLUDE_DIR
)

if(GraphBLAS_FOUND)
  set(GraphBLAS_LIBRARIES ${GraphBLAS_LIBRARY})
  set(GraphBLAS_INCLUDE_DIRS ${GraphBLAS_INCLUDE_DIR})
endif()
