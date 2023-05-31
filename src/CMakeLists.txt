# Add the OSQP sources
target_sources(OSQPLIB PUBLIC
               "${CMAKE_CURRENT_SOURCE_DIR}/auxil.c"
               "${CMAKE_CURRENT_SOURCE_DIR}/error.c"
               "${CMAKE_CURRENT_SOURCE_DIR}/scaling.c"
               "${CMAKE_CURRENT_SOURCE_DIR}/util.c")

# Source files that are needed for embedded code generation
list( APPEND EMBEDDED_SRCS
      "${CMAKE_CURRENT_SOURCE_DIR}/auxil.c"
      "${CMAKE_CURRENT_SOURCE_DIR}/error.c"
      "${CMAKE_CURRENT_SOURCE_DIR}/osqp_api.c"
      "${CMAKE_CURRENT_SOURCE_DIR}/scaling.c"
      "${CMAKE_CURRENT_SOURCE_DIR}/util.c" )

# Add more files that should only be in non-embedded code
if(NOT DEFINED OSQP_EMBEDDED_MODE)
  target_sources(OSQPLIB PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/polish.c")
endif()

# Add the derivative support, if enabled
if(OSQP_ENABLE_DERIVATIVES)
  target_sources(OSQPLIB PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/derivative.c")
endif()

# Add the ctrl-c handler if enabled and not overriden
if(OSQP_ENABLE_INTERRUPT AND NOT OSQP_CUSTOM_INTERRUPT)
  if(IS_WINDOWS)
    target_sources(OSQPLIB PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/interrupt_windows.c")
  elseif(IS_MAC OR IS_LINUX)
    target_sources(OSQPLIB PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/interrupt_unix.c")
  endif()
endif()

# Add the timing functions if enabled and not overriden
if(OSQP_ENABLE_PROFILING AND NOT OSQP_CUSTOM_TIMING)
  if(IS_WINDOWS)
    target_sources(OSQPLIB PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/timing_windows.c")
  elseif(IS_MAC)
    target_sources(OSQPLIB PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/timing_macos.c")
  else()
    # Assume the other platforms are Linux
    target_sources(OSQPLIB PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/timing_linux.c")
  endif()
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
