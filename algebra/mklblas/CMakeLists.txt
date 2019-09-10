# Define the algebra implementation

#look for INTEL blas
set(BLA_VENDOR Intel10_64lp)
find_package(BLAS REQUIRED)

message("-- BLAS_LINKER_FLAGS: ${BLAS_LINKER_FLAGS}")
message("-- BLAS_LIBRARIES: ${BLAS_LIBRARIES}")

set(algebra_impl_src
    "${CMAKE_CURRENT_SOURCE_DIR}/matrix.c"
    "${CMAKE_CURRENT_SOURCE_DIR}/vector.c"
)

set(algebra_impl_headers
    "${CMAKE_CURRENT_SOURCE_DIR}/algebra_impl.h"
)

add_library(algebra_impl OBJECT ${algebra_impl_src})

target_link_libraries(algebra_impl ${BLAS_LIBRARIES})

# Add parent directory for library handler
target_include_directories(algebra_impl PRIVATE
                           ${algebra_impl_headers}
                           ${PROJECT_SOURCE_DIR}/include
                           ${PROJECT_SOURCE_DIR}/algebra/csc_tools)
