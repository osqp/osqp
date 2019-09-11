# Define the algebra implementation

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
