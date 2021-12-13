# Add the OSQP headers
set(osqp_headers
    "${CMAKE_CURRENT_SOURCE_DIR}/version.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/algebra_vector.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/algebra_matrix.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/auxil.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/csc_type.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/error.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/glob_opts.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/lin_alg.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/proj.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/scaling.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/types.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/util.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/osqp.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/osqp_api_constants.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/osqp_api_functions.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/osqp_api_types.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/osqp_api_utils.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/osqp_configure.h")

# Add more files that should only be in non-embedded code
if(NOT DEFINED EMBEDDED)
  list(APPEND osqp_headers "${CMAKE_CURRENT_SOURCE_DIR}/polish.h" "${CMAKE_CURRENT_SOURCE_DIR}/lin_sys.h")

  if(ALGEBRA_DEFAULT)
    list(APPEND osqp_headers "${CMAKE_CURRENT_SOURCE_DIR}/codegen.h")
  endif()
endif()

# Add the ctrl-c handler if enabled
if(CTRLC)
  list(APPEND osqp_headers "${CMAKE_CURRENT_SOURCE_DIR}/ctrlc.h")
endif()

target_sources(OSQPLIB PUBLIC ${osqp_headers})
target_include_directories(OSQPLIB PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

set(osqp_headers
    "${CMAKE_CURRENT_SOURCE_DIR}/csc_type.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/osqp.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/osqp_api_constants.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/osqp_api_functions.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/osqp_api_types.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/osqp_api_utils.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/osqp_configure.h"
    PARENT_SCOPE)
