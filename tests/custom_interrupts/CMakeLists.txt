cmake_minimum_required(VERSION 3.16)
project( OSQP_custom_interrupts )

# Set the CMake options
set( OSQP_BUILD_SHARED_LIB OFF )
set( OSQP_BUILD_DEMO_EXE OFF )
set( OSQP_CUSTOM_INTERRUPTS "${CMAKE_CURRENT_SOURCE_DIR}/custom_interrupts.h" )

add_subdirectory( ../../ ${CMAKE_CURRENT_BINARY_DIR}/osqp )

add_executable( osqp_custom_interrupts ../../examples/osqp_demo.c custom_interrupts.c )
target_link_libraries( osqp_custom_interrupts osqpstatic m )
