# Add the OSQP sources
target_sources(OSQPLIB PUBLIC
               "${CMAKE_CURRENT_SOURCE_DIR}/auxil.c"
               "${CMAKE_CURRENT_SOURCE_DIR}/error.c"
               "${CMAKE_CURRENT_SOURCE_DIR}/proj.c"
               "${CMAKE_CURRENT_SOURCE_DIR}/scaling.c"
               "${CMAKE_CURRENT_SOURCE_DIR}/util.c")

# Source files that are needed for embedded code generation
list( APPEND EMBEDDED_SRCS
      "${CMAKE_CURRENT_SOURCE_DIR}/auxil.c"
      "${CMAKE_CURRENT_SOURCE_DIR}/error.c"
      "${CMAKE_CURRENT_SOURCE_DIR}/osqp_api.c"
      "${CMAKE_CURRENT_SOURCE_DIR}/proj.c"
      "${CMAKE_CURRENT_SOURCE_DIR}/scaling.c"
      "${CMAKE_CURRENT_SOURCE_DIR}/util.c" )

# Add more files that should only be in non-embedded code
if(NOT DEFINED EMBEDDED)
  target_sources(OSQPLIB PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/polish.c" "${CMAKE_CURRENT_SOURCE_DIR}/lin_sys.c")
endif()

# Add the ctrl-c handler if enabled
if(OSQP_ENABLE_INTERRUPT)
  target_sources(OSQPLIB PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/ctrlc.c")
endif()

# Add code generation functionality if enabled
# Added last because this also processes the copying of files needed in the generated code
if(OSQP_CODEGEN)
  target_sources(OSQPLIB PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/codegen.c")

  foreach( f ${EMBEDDED_SRCS} )
    get_filename_component( fname ${f} NAME )

    set( dest_file "${EMBEDDED_BUILD_SRC_DIR}/${fname}" )
    list( APPEND EMBEDDED_BUILD_SRCS "${dest_file}" )

    add_custom_command(OUTPUT ${dest_file}
                       COMMAND ${CMAKE_COMMAND} -E copy "${f}" "${dest_file}"
                       DEPENDS ${f}
                       COMMENT "Copying ${fname}" )
  endforeach()

  add_custom_target( copy_codegen_srcs DEPENDS ${EMBEDDED_BUILD_SRCS} )
  add_dependencies( copy_codegen_files copy_codegen_srcs )
endif()
