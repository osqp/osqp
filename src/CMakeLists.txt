# Add the OSQP sources
target_sources(
  OSQPLIB
  PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/auxil.c" "${CMAKE_CURRENT_SOURCE_DIR}/error.c"
         "${CMAKE_CURRENT_SOURCE_DIR}/osqp_api.c" "${CMAKE_CURRENT_SOURCE_DIR}/proj.c"
         "${CMAKE_CURRENT_SOURCE_DIR}/scaling.c" "${CMAKE_CURRENT_SOURCE_DIR}/util.c")

# Add more files that should only be in non-embedded code
if(NOT DEFINED EMBEDDED)
  target_sources(OSQPLIB PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/polish.c" "${CMAKE_CURRENT_SOURCE_DIR}/lin_sys.c")

  if(ALGEBRA_DEFAULT)
    target_sources(OSQPLIB PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/codegen.c")
  endif()
endif()

# Add the ctrl-c handler if enabled
if(CTRLC)
  target_sources(OSQPLIB PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/ctrlc.c")
endif()
