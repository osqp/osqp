# Add the OSQP headers
set(osqp_headers_private
    "${CMAKE_CURRENT_SOURCE_DIR}/private/version.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/private/algebra_vector.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/private/algebra_matrix.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/private/auxil.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/private/error.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/private/glob_opts.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/private/lin_alg.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/private/printing.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/private/scaling.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/private/timing.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/private/types.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/private/util.h")

set(osqp_headers_public
    "${CMAKE_CURRENT_SOURCE_DIR}/public/osqp.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/public/osqp_api_constants.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/public/osqp_api_functions.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/public/osqp_api_types.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/public/osqp_api_utils.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/public/osqp_export_define.h")

list( APPEND EMBEDDED_PUBLIC_INCS ${osqp_headers_public} )
list( APPEND EMBEDDED_PRIVATE_INCS ${osqp_headers_private} )

# The configure header is handled differentally for code generation (we don't copy it)
list( APPEND osqp_headers_public "${CMAKE_CURRENT_BINARY_DIR}/public/osqp_configure.h")
set( osqp_headers ${osqp_headers_public} PARENT_SCOPE)

# Add more files that should only be in non-embedded code
if(NOT DEFINED OSQP_EMBEDDED_MODE)
  list(APPEND osqp_headers_private
       "${CMAKE_CURRENT_SOURCE_DIR}/private/polish.h")
endif()

# Add the derivative support, if enabled
if(OSQP_ENABLE_DERIVATIVES)
  list(APPEND osqp_headers_private
       "${CMAKE_CURRENT_SOURCE_DIR}/private/derivative.h")
endif()

# Add the ctrl-c handler if enabled
if(OSQP_ENABLE_INTERRUPT)
  list(APPEND osqp_headers_private "${CMAKE_CURRENT_SOURCE_DIR}/private/interrupt.h")
endif()

target_sources(OSQPLIB PUBLIC ${osqp_headers})
target_sources(OSQPLIB PRIVATE ${osqp_headers_private})
target_include_directories(OSQPLIB PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/public")
target_include_directories(OSQPLIB PUBLIC "${CMAKE_CURRENT_BINARY_DIR}/public")
target_include_directories(OSQPLIB PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/private")
target_include_directories(OSQPLIB PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/private")


# Add code generation functionality if enabled
# Added last because this also processes the copying of files needed in the generated code
if(OSQP_CODEGEN)
  list(APPEND osqp_headers "${CMAKE_CURRENT_SOURCE_DIR}/private/codegen.h")

  foreach( f ${EMBEDDED_PUBLIC_INCS} )
    get_filename_component( fname ${f} NAME )

    set( dest_file "${EMBEDDED_BUILD_PUBLIC_INC_DIR}/${fname}" )
    list( APPEND EMBEDDED_BUILD_INCS "${dest_file}" )

    add_custom_command(OUTPUT ${dest_file}
                       COMMAND ${CMAKE_COMMAND} -E copy "${f}" "${dest_file}"
                       DEPENDS ${f}
                       COMMENT "Copying ${fname}" )
  endforeach()

  foreach( f ${EMBEDDED_PRIVATE_INCS} )
    get_filename_component( fname ${f} NAME )

    set( dest_file "${EMBEDDED_BUILD_PRIVATE_INC_DIR}/${fname}" )
    list( APPEND EMBEDDED_BUILD_INCS "${dest_file}" )

    add_custom_command(OUTPUT ${dest_file}
                       COMMAND ${CMAKE_COMMAND} -E copy "${f}" "${dest_file}"
                       DEPENDS ${f}
                       COMMENT "Copying ${fname}" )
  endforeach()

  add_custom_target( copy_codegen_incs DEPENDS ${EMBEDDED_BUILD_INCS} )
  add_dependencies( copy_codegen_files copy_codegen_incs )
endif()
