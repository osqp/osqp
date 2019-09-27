# Define the algebra implementation

set(algebra_impl_src
    "${CMAKE_CURRENT_SOURCE_DIR}/algebra_libs.c"
    "${CMAKE_CURRENT_SOURCE_DIR}/cuda_handler.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/cuda_malloc.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/matrix.c"
    "${CMAKE_CURRENT_SOURCE_DIR}/vector.c"
)

set(algebra_impl_headers
    "${CMAKE_CURRENT_SOURCE_DIR}/algebra_types.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/cuda_configure.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/cuda_handler.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/cuda_malloc.h"
    # "${CMAKE_CURRENT_SOURCE_DIR}/cuda_wrapper.h"
)

set(algebra_impl ${algebra_impl_src} ${algebra_impl_headers} PARENT_SCOPE)
