cmake_minimum_required(VERSION 3.18)
project( OSQP_custom_printing )

# Set the CMake options
set( OSQP_BUILD_SHARED_LIB OFF CACHE BOOL "Disable shared library")
set( OSQP_BUILD_DEMO_EXE OFF CACHE BOOL "Disable demo exe")
set( OSQP_CUSTOM_PRINTING "custom_printing.h" CACHE STRING "Header file")


include_directories(${CMAKE_CURRENT_SOURCE_DIR})
add_subdirectory( ../../ ${CMAKE_CURRENT_BINARY_DIR}/osqp )

add_executable( osqp_custom_printing ../../examples/osqp_demo.c custom_printing.c )
target_link_libraries( osqp_custom_printing osqpstatic m )
