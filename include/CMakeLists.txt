# Add the OSQP headers
set(
    osqp_headers
    "${CMAKE_CURRENT_SOURCE_DIR}/version.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/auxil.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/constants.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/error.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/glob_opts.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/lin_alg.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/osqp.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/osqp_configure.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/proj.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/scaling.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/types.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/util.h"
)

# Add the KKT update only in normal mode and matrix-updating embedded mode (not mode 1)
if (NOT (EMBEDDED EQUAL 1))
    list(
      APPEND
      osqp_src
      "${CMAKE_CURRENT_SOURCE_DIR}/kkt.h"
    )
endif()

# Add more files that should only be in non-embedded code
if (NOT DEFINED EMBEDDED)
    list(
      APPEND
      osqp_headers
      "${CMAKE_CURRENT_SOURCE_DIR}/cs.h"
      "${CMAKE_CURRENT_SOURCE_DIR}/polish.h"
      "${CMAKE_CURRENT_SOURCE_DIR}/lin_sys.h"
    )
endif()

# Add the ctrl-c handler if enabled
if (CTRLC)
    list(
      APPEND
      osqp_headers
      "${CMAKE_CURRENT_SOURCE_DIR}/ctrlc.h"
    )
endif()

# Pass the header list up to the main CMakeLists scope
set(
  osqp_headers
  "${osqp_headers}"
  PARENT_SCOPE
)
