# This file is used to test the CMake installation and library finding logic
# for the package
cmake_minimum_required(VERSION 3.16)
project( OSQP_test_install )

# The test install is up two levels
set( OSQP_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../../install" )

find_package( OSQP )

if( ${OSQP_FOUND} )
    message( "Found OSQP version " ${OSQP_VERSION} )
endif()

add_executable( osqp_static ../../examples/osqp_demo.c)
target_link_libraries( osqp_static osqp::osqpstatic m )

add_executable( osqp_shared ../../examples/osqp_demo.c)
target_link_libraries( osqp_shared osqp::osqp )
