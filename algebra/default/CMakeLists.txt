# Define the algebra implementation

set(algebra_impl_src
    "${CMAKE_CURRENT_SOURCE_DIR}/matrix.c"
    "${CMAKE_CURRENT_SOURCE_DIR}/vector.c"
    "${CMAKE_CURRENT_SOURCE_DIR}/cuda_handler.cu"
)

set(algebra_impl_headers
    "${CMAKE_CURRENT_SOURCE_DIR}/algebra_impl.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/cuda_handler.h"
)

set(algebra_impl ${algebra_impl_src} ${algebra_impl_headers} PARENT_SCOPE)
