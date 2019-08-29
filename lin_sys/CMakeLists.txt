# Add subdirectory for each linear systems solver

if (NOT DEFINED EMBEDDED)

# Include this directory for library handler
# NB Needed for subfolders
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

endif()

# Direct solver
add_subdirectory(direct)

# Indirect solver
# TODO: Add!

# Add linsys handler if not embedded
if (NOT DEFINED EMBEDDED)
set(linsys_lib_handler
    ${CMAKE_CURRENT_SOURCE_DIR}/lib_handler.c
    ${CMAKE_CURRENT_SOURCE_DIR}/lib_handler.h)
endif()

# Combine solvers
# TODO: Add indirect ones
# Add library handler if desktop version
set(linsys_solvers
    ${direct_linsys_solvers}
    ${linsys_lib_handler})

# Combine solvers external libraries
set(linsys_solvers_includes
    ${direct_linsys_solvers_includes})

# Add the KKT object only in normal mode and matrix-updating embedded mode (not mode 1)
if (NOT (EMBEDDED EQUAL 1))

  add_subdirectory(kkt_common)

  set(linsys_solvers ${linsys_solvers} $<TARGET_OBJECTS:kkt_common>)
  set(linsys_solvers_includes ${linsys_solvers_includes} ${kkt_common_includes})

endif()

#pass upwards
set(linsys_solvers ${linsys_solvers} PARENT_SCOPE)
set(linsys_solvers_includes ${linsys_solvers_includes} PARENT_SCOPE)
