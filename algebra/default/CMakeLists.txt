include(FetchContent)

message(STATUS "Fetching/configuring QDLDL solver")
list(APPEND CMAKE_MESSAGE_INDENT "  ")

# Corresponds to 0.1.6 release of QDLDL
FetchContent_Declare(
  qdldl
  GIT_REPOSITORY https://github.com/osqp/qdldl.git
  GIT_TAG 29d140419a3bec20d860052d73ba2be927faf5a1
  SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/lin_sys/direct/qdldl_sources)
FetchContent_GetProperties(qdldl)

if(NOT qdldl_POPULATED)
  FetchContent_Populate(qdldl)

  # Make QDLDL use the same types as OSQP
  set(QDLDL_FLOAT ${DFLOAT} CACHE BOOL "QDLDL Float type")
  set(QDLDL_LONG ${DLONG} CACHE BOOL "QDLDL Integer type")

  # We only want the object library, so turn off the other library products
  set(QDLDL_BUILD_STATIC_LIB OFF CACHE BOOL "Build QDLDL static library")
  set(QDLDL_BUILD_SHARED_LIB OFF CACHE BOOL "Build QDLDL shared library")

  # We don't actually want to build anything from here by default
  add_subdirectory(${qdldl_SOURCE_DIR} ${qdldl_BINARY_DIR} EXCLUDE_FROM_ALL)
endif()

list(POP_BACK CMAKE_MESSAGE_INDENT)

set_source_files_properties($<TARGET_OBJECTS:qdldlobject> PROPERTIES GENERATED 1)

get_property(
  qdldl_include
  TARGET qdldlobject
  PROPERTY INCLUDE_DIRECTORIES)

if(NOT EMBEDDED)
  file(
    GLOB
    NON_EMBEDDED_SRC_FILES
    CONFIGURE_DEPENDS
    lin_sys/direct/amd/src/*.c
    lin_sys/direct/amd/include/*.h
    ../_common/csc_utils.h
    ../_common/csc_utils.c)
endif()

target_sources(
  OSQPLIB
  PRIVATE ../_common/csc_math.h
          ../_common/csc_math.c
          ../_common/kkt.h
          ../_common/kkt.c
          algebra_impl.h
          algebra_libs.c
          vector.c
          matrix.c
          ${NON_EMBEDDED_SRC_FILES}
          lin_sys/direct/qdldl_interface.h
          lin_sys/direct/qdldl_interface.c
          $<TARGET_OBJECTS:qdldlobject>)

target_include_directories(OSQPLIB PRIVATE ../_common ${CMAKE_CURRENT_SOURCE_DIR} ${qdldl_include} lin_sys/direct
                                           lin_sys/direct/amd/include)


# Setup the file copying for the code generation target
if( OSQP_CODEGEN )
  set( EMBEDDED_LINALG
       ${CMAKE_CURRENT_SOURCE_DIR}/../_common/csc_math.h
       ${CMAKE_CURRENT_SOURCE_DIR}/../_common/csc_math.c
       ${CMAKE_CURRENT_SOURCE_DIR}/../_common/kkt.h
       ${CMAKE_CURRENT_SOURCE_DIR}/../_common/kkt.c
       ${CMAKE_CURRENT_SOURCE_DIR}/algebra_impl.h
       ${CMAKE_CURRENT_SOURCE_DIR}/algebra_libs.c
       ${CMAKE_CURRENT_SOURCE_DIR}/vector.c
       ${CMAKE_CURRENT_SOURCE_DIR}/matrix.c
       ${CMAKE_CURRENT_SOURCE_DIR}/lin_sys/direct/qdldl_interface.h
       ${CMAKE_CURRENT_SOURCE_DIR}/lin_sys/direct/qdldl_interface.c
       ${qdldl_SOURCE_DIR}/src/qdldl.c
       ${qdldl_SOURCE_DIR}/include/qdldl.h
       ${qdldl_SOURCE_DIR}/include/qdldl_version.h )

  foreach( f ${EMBEDDED_LINALG} )
    get_filename_component( fname ${f} NAME )
    get_filename_component( fext ${f} EXT )

    if( fext STREQUAL ".h" )
      set( dest_file "${EMBEDDED_BUILD_PRIVATE_INC_DIR}/${fname}" )
    elseif( fext STREQUAL ".c" )
      set( dest_file "${EMBEDDED_BUILD_SRC_DIR}/${fname}" )
    endif()

    list( APPEND EMBEDDED_BUILD_LINALG "${dest_file}" )

    add_custom_command(OUTPUT ${dest_file}
                       COMMAND ${CMAKE_COMMAND} -E copy "${f}" "${dest_file}"
                       DEPENDS ${f}
                       COMMENT "Copying ${fname}" )
  endforeach()

    add_custom_command(OUTPUT "${EMBEDDED_BUILD_PRIVATE_INC_DIR}/qdldl_types.h"
                       COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/lin_sys/direct/qdldl_codegen_types.h.in"
                                                        "${EMBEDDED_BUILD_PRIVATE_INC_DIR}/qdldl_types.h"
                       DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/lin_sys/direct/qdldl_codegen_types.h.in"
                       COMMENT "Copying qdldl_types.h" )


  add_custom_target( copy_codegen_linalg DEPENDS ${EMBEDDED_BUILD_LINALG} "${EMBEDDED_BUILD_PRIVATE_INC_DIR}/qdldl_types.h" )
  add_dependencies( copy_codegen_files copy_codegen_linalg )
endif()
