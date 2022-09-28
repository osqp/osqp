# Use the QDLDL solver in this algebra
include(${OSQP_ALGEBRA_ROOT}/_common/lin_sys/qdldl/qdldl.cmake)

if(NOT OSQP_EMBEDDED_MODE)
  set( NON_EMBEDDED_SRC_FILES
       ${LIN_SYS_QDLDL_NON_EMBEDDED_SRC_FILES} )
endif()

target_sources(
  OSQPLIB
  PRIVATE ../_common/csc_math.h
          ../_common/csc_math.c
          ../_common/csc_utils.h
          ../_common/csc_utils.c
          algebra_impl.h
          algebra_libs.c
          vector.c
          matrix.c
          ${NON_EMBEDDED_SRC_FILES}
          ${LIN_SYS_QDLDL_EMBEDDED_SRC_FILES}
          $<TARGET_OBJECTS:qdldlobject> )

target_include_directories(
  OSQPLIB
  PRIVATE ../_common
          ${CMAKE_CURRENT_SOURCE_DIR}
          ${LIN_SYS_QDLDL_INC_PATHS} )


# Setup the file copying for the code generation target
if( OSQP_CODEGEN )
  set( EMBEDDED_LINALG
       ${CMAKE_CURRENT_SOURCE_DIR}/algebra_impl.h
       ${CMAKE_CURRENT_SOURCE_DIR}/algebra_libs.c
       ${CMAKE_CURRENT_SOURCE_DIR}/vector.c
       ${CMAKE_CURRENT_SOURCE_DIR}/matrix.c
       ${OSQP_ALGEBRA_ROOT}/_common/csc_math.h
       ${OSQP_ALGEBRA_ROOT}/_common/csc_math.c
       ${OSQP_ALGEBRA_ROOT}/_common/csc_utils.h
       ${OSQP_ALGEBRA_ROOT}/_common/csc_utils.c
       ${OSQP_ALGEBRA_ROOT}/_common/kkt.h
       ${OSQP_ALGEBRA_ROOT}/_common/kkt.c
       ${OSQP_ALGEBRA_ROOT}/_common/lin_sys/qdldl/qdldl_interface.h
       ${OSQP_ALGEBRA_ROOT}/_common/lin_sys/qdldl/qdldl_interface.c
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
                       COMMAND ${CMAKE_COMMAND} -E copy "${OSQP_ALGEBRA_ROOT}/_common/lin_sys/qdldl/qdldl_codegen_types.h.in"
                                                        "${EMBEDDED_BUILD_PRIVATE_INC_DIR}/qdldl_types.h"
                       DEPENDS "${OSQP_ALGEBRA_ROOT}/_common/lin_sys/qdldl/qdldl_codegen_types.h.in"
                       COMMENT "Copying qdldl_types.h" )


  add_custom_target( copy_codegen_linalg DEPENDS ${EMBEDDED_BUILD_LINALG} "${EMBEDDED_BUILD_PRIVATE_INC_DIR}/qdldl_types.h" )
  add_dependencies( copy_codegen_files copy_codegen_linalg )
endif()
