# Add the OSQP headers
set(osqp_headers_private
    "${CMAKE_CURRENT_SOURCE_DIR}/private/version.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/private/algebra_vector.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/private/algebra_matrix.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/private/auxil.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/private/error.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/private/glob_opts.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/private/lin_alg.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/private/proj.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/private/scaling.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/private/types.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/private/util.h")

set(osqp_headers
    "${CMAKE_CURRENT_SOURCE_DIR}/public/csc_type.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/public/osqp.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/public/osqp_api_constants.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/public/osqp_api_functions.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/public/osqp_api_types.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/public/osqp_api_utils.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/public/osqp_export_define.h"
    "${CMAKE_CURRENT_BINARY_DIR}/public/osqp_configure.h"
    PARENT_SCOPE)

# Add code generation functionality if enabled
if(OSQP_CODEGEN)
  list(APPEND osqp_headers "${CMAKE_CURRENT_SOURCE_DIR}/private/codegen.h")
endif()

# Add more files that should only be in non-embedded code
if(NOT DEFINED EMBEDDED)
  list(APPEND osqp_headers_private
       "${CMAKE_CURRENT_SOURCE_DIR}/private/polish.h"
       "${CMAKE_CURRENT_SOURCE_DIR}/private/lin_sys.h"
       "${CMAKE_CURRENT_BINARY_DIR}/private/memory_defs.h")
endif()

# Add the ctrl-c handler if enabled
if(CTRLC)
  list(APPEND osqp_headers_private "${CMAKE_CURRENT_SOURCE_DIR}/private/ctrlc.h")
endif()

target_sources(OSQPLIB PUBLIC ${osqp_headers})
target_sources(OSQPLIB PRIVATE ${osqp_headers_private})
target_include_directories(OSQPLIB PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/public")
target_include_directories(OSQPLIB PUBLIC "${CMAKE_CURRENT_BINARY_DIR}/public")
target_include_directories(OSQPLIB PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/private")
target_include_directories(OSQPLIB PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/private")
