# Define the algebra implementation

#look for INTEL blas
set(BLA_VENDOR Intel10_64lp)
find_package(BLAS REQUIRED)

message("-- Building with BLAS-based algebra.")
message("-- BLAS_LIBRARIES: ${BLAS_LIBRARIES}")

target_sources(OSQPLIB
    PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}/matrix.c"
      "${CMAKE_CURRENT_SOURCE_DIR}/vector.c"
      "${CMAKE_CURRENT_SOURCE_DIR}/algebra_impl.h"
)


target_include_directories(OSQPLIB
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(OSQPLIB ${BLAS_LIBRARIES})
