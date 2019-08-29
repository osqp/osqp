# Add the OSQP sources
set(
    osqp_src
    "${CMAKE_CURRENT_SOURCE_DIR}/auxil.c"
    "${CMAKE_CURRENT_SOURCE_DIR}/error.c"
    "${CMAKE_CURRENT_SOURCE_DIR}/osqp_api.c"
    "${CMAKE_CURRENT_SOURCE_DIR}/proj.c"
    "${CMAKE_CURRENT_SOURCE_DIR}/scaling.c"
    "${CMAKE_CURRENT_SOURCE_DIR}/util.c"
)

# Add more files that should only be in non-embedded code
if (NOT DEFINED EMBEDDED)
    list(
      APPEND
      osqp_src
      "${CMAKE_CURRENT_SOURCE_DIR}/polish.c"
      "${CMAKE_CURRENT_SOURCE_DIR}/lin_sys.c"
    )
endif()

# Add the ctrl-c handler if enabled
if (CTRLC)
    list(
      APPEND
      osqp_src
      "${CMAKE_CURRENT_SOURCE_DIR}/ctrlc.c"
    )
endif()

# Pass the source list up to the main CMakeLists scope
set(
  osqp_src
  "${osqp_src}"
  PARENT_SCOPE
)
