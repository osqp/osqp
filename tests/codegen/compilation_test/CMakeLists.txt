# This file is used to test that the embedded codegen files can compile properly into
# a demo program with various settings
cmake_minimum_required(VERSION 3.16)
project( OSQP_test_settings_codegen )

set( OSQP_BUILD_DIR "" CACHE STRING "The root directory of the OSQP build tree" )
set( OSQP_TEST_CODEGEN_DIR "" CACHE STRING "The directory the generated files are in" )

message( STATUS "Configuring OSQP embedded settings compilation test" )

file( GLOB OSQP_SOURCES
      CONFIGURE_DEPENDS
      ${OSQP_BUILD_DIR}/codegen_src/src/*.c
      ${OSQP_BUILD_DIR}/codegen_src/inc/public/*.h
      ${OSQP_BUILD_DIR}/codegen_src/inc/private/*.h )

file( GLOB CODEGEN_MODE1_SOURCES
      CONFIGURE_DEPENDS
      ${OSQP_TEST_CODEGEN_DIR}/embedded1/*workspace.c
      ${OSQP_TEST_CODEGEN_DIR}/embedded1/*.h )

file( GLOB CODEGEN_MODE2_SOURCES
      CONFIGURE_DEPENDS
      ${OSQP_TEST_CODEGEN_DIR}/embedded2/*workspace.c
      ${OSQP_TEST_CODEGEN_DIR}/embedded2/*.h )


message( STATUS "Found OSQP sources:" )
list(APPEND CMAKE_MESSAGE_INDENT "  ")
foreach( f ${OSQP_SOURCES} )
    message( STATUS ${f} )
endforeach()
list(POP_BACK CMAKE_MESSAGE_INDENT)

message( STATUS "Found codegen embedded mode 1 workspace sources:" )
list(APPEND CMAKE_MESSAGE_INDENT "  ")
foreach( f ${CODEGEN_MODE1_SOURCES} )
    message( STATUS ${f} )
endforeach()
list(POP_BACK CMAKE_MESSAGE_INDENT)

message( STATUS "Found codegen embedded mode 2 workspace sources:" )
list(APPEND CMAKE_MESSAGE_INDENT "  ")
foreach( f ${CODEGEN_MODE2_SOURCES} )
    message( STATUS ${f} )
endforeach()
list(POP_BACK CMAKE_MESSAGE_INDENT)


set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O0 -g")


add_executable( osqp_codegen_embedded_mode1 embedded_mode1.c ${OSQP_SOURCES} ${CODEGEN_MODE1_SOURCES} )
target_include_directories( osqp_codegen_embedded_mode1
                            PRIVATE
                            ${OSQP_BUILD_DIR}/codegen_src/inc/public
                            ${OSQP_BUILD_DIR}/codegen_src/inc/private
                            ${OSQP_TEST_CODEGEN_DIR}/embedded1 )

set_property(TARGET osqp_codegen_embedded_mode1 PROPERTY C_STANDARD 90)

add_executable( osqp_codegen_embedded_mode2 embedded_mode2.c ${OSQP_SOURCES} ${CODEGEN_MODE2_SOURCES} )
target_include_directories( osqp_codegen_embedded_mode2
                            PRIVATE
                            ${OSQP_BUILD_DIR}/codegen_src/inc/public
                            ${OSQP_BUILD_DIR}/codegen_src/inc/private
                            ${OSQP_TEST_CODEGEN_DIR}/embedded2 )
target_link_libraries( osqp_codegen_embedded_mode2 m )

set_property(TARGET osqp_codegen_embedded_mode2 PROPERTY C_STANDARD 90)
