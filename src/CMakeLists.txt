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

if(OSQP_PROFILER_ANNOTATIONS)
  target_sources(OSQPLIB PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/profilers.c")

  if(OSQP_ALGEBRA_CUDA)
    message(STATUS "Enabling profiling annotations using NVTX")
    target_sources(OSQPLIB PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/profilers_nvtx.c)
    target_link_libraries(OSQPLIB CUDA::nvtx3)
  elseif(OSQP_PROFILER_ANNOTATIONS STREQUAL "itt")
    message(STATUS "Enabling profiling annotations using ITT API")
    find_package(ITTAPI)

    target_sources(OSQPLIB PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/profilers_itt.c")
    target_link_libraries(OSQPLIB ITTAPI::ittnotify)
  elseif(OSQP_PROFILER_ANNOTATIONS STREQUAL "roctx")
    message(FATAL_ERROR "ROCTX is not supported, no AMDGPU backend in OSQP yet")
    message(STATUS "Enabling profiling annotations using ROCTX")
    find_package(ROCTX)

    target_sources(OSQPLIB PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/profilers_roctx.c")
    target_link_libraries(OSQPLIB ROC::ROCTX)
  elseif(OSQP_PROFILER_ANNOTATIONS STREQUAL "omnitrace")
    message(STATUS "Enabling profiling annotations using omnitrace")
    find_package(omnitrace REQUIRED COMPONENTS user)

    target_sources(OSQPLIB PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/profilers_omnitrace.c")
    target_link_libraries(OSQPLIB omnitrace::omnitrace-user-library)
  else()
    message(FATAL_ERROR "Unknown profiler annotation option: ${OSQP_PROFILER_ANNOTATIONS}")
  endif()
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
