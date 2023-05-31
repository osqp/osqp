# This file is used to test that the embedded codegen files can compile properly into
# a demo program.
cmake_minimum_required(VERSION 3.16)
project( OSQP_test_matrix_codegen )

set( OSQP_BUILD_DIR "" CACHE STRING "The root directory of the OSQP build tree" )
set( OSQP_CODEGEN_DIR "" CACHE STRING "The directory the generated files are in" )

message( STATUS "Configuring OSQP embedded matrix update test" )

file( GLOB OSQP_SOURCES
      CONFIGURE_DEPENDS
      ${OSQP_BUILD_DIR}/codegen_src/src/*.c
      ${OSQP_BUILD_DIR}/codegen_src/inc/public/*.h
      ${OSQP_BUILD_DIR}/codegen_src/inc/private/*.h )

file( GLOB CODEGEN_SOURCES
      CONFIGURE_DEPENDS
      ${OSQP_CODEGEN_DIR}/*workspace.c
      ${OSQP_CODEGEN_DIR}/*.h )


set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O0 -g")


message( STATUS "Found OSQP sources:" )
list(APPEND CMAKE_MESSAGE_INDENT "  ")
foreach( f ${OSQP_SOURCES} )
    message( STATUS ${f} )
endforeach()
list(POP_BACK CMAKE_MESSAGE_INDENT)

message( STATUS "Found codegen workspace sources:" )
list(APPEND CMAKE_MESSAGE_INDENT "  ")
foreach( f ${CODEGEN_SOURCES} )
    message( STATUS ${f} )
endforeach()
list(POP_BACK CMAKE_MESSAGE_INDENT)


add_executable( osqp_codegen_matrix_c mat_codegen_test.c ${OSQP_SOURCES} ${CODEGEN_SOURCES} )
target_include_directories( osqp_codegen_matrix_c
                            PRIVATE
                            ${OSQP_BUILD_DIR}/codegen_src/inc/public
                            ${OSQP_BUILD_DIR}/codegen_src/inc/private
                            ${OSQP_CODEGEN_DIR} )

# Embedded mode 2 needs the sqrt and modf functions from a math library
target_link_libraries( osqp_codegen_matrix_c m )

add_executable( osqp_codegen_matrix_cpp mat_codegen_test.c ${OSQP_SOURCES} ${CODEGEN_SOURCES} )
target_include_directories( osqp_codegen_matrix_cpp
                            PRIVATE
                            ${OSQP_BUILD_DIR}/codegen_src/inc/public
                            ${OSQP_BUILD_DIR}/codegen_src/inc/private
                            ${OSQP_CODEGEN_DIR} )

# Embedded mode 2 needs the sqrt and modf functions from a math library
target_link_libraries( osqp_codegen_matrix_cpp m )
