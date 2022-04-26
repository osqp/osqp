# This file is used to test the CMake installation and library finding logic
# for the package
cmake_minimum_required(VERSION 3.16)
project( OSQP_test_install )

set( OSQP_INSTALL_PREFIX  "" CACHE STRING "The location OSQP is installed in" )
set( OSQP_DESIRED_VERSION "" CACHE STRING "The minimum version of OSQP to allow" )


# The test install is up two levels
set( OSQP_ROOT "${OSQP_INSTALL_PREFIX}" )


if( NOT ${OSQP_DESIRED_VERSION} )
    find_package( OSQP )
else()
    find_package( OSQP ${OSQP_DESIRED_VERSION} )
endif()

if( ${OSQP_FOUND} )
    message( "Found OSQP version " ${OSQP_VERSION} )
endif()

add_executable( osqp_static ../../examples/osqp_demo.c)
target_link_libraries( osqp_static osqp::osqpstatic m )

add_executable( osqp_shared ../../examples/osqp_demo.c)
target_link_libraries( osqp_shared osqp::osqp )
