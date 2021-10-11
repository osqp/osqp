# Set source files
target_sources(OSQPLIB
    PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/mkl-cg_interface.h
      ${CMAKE_CURRENT_SOURCE_DIR}/mkl-cg_interface.c
)

target_include_directories(OSQPLIB
PRIVATE
${CMAKE_CURRENT_SOURCE_DIR})
