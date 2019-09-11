# Add subdirectory for each linear systems solver

if (NOT DEFINED EMBEDDED)

  # Include this directory for library handler
  # NB: Needed by the subfolders
  target_include_directories(OSQPLIB
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR})

endif()

# Add linsys handler if not embedded
if (NOT DEFINED EMBEDDED)
target_sources(OSQPLIB
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/lib_handler.c
    ${CMAKE_CURRENT_SOURCE_DIR}/lib_handler.h)
endif()

# Add the KKT object only in normal mode and
# matrix-updating embedded mode (not mode 1)
if (NOT (EMBEDDED EQUAL 1))
  add_subdirectory(kkt_common)
endif()

# Direct solvers
add_subdirectory(direct)
