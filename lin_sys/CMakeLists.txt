# Add subdirectory for each linear systems solver

# Include this directory for library handler
# NB Needed for subfolders
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# Direct solver
add_subdirectory(direct)

# Indirect solver
# TODO: Add!


# Combine solvers
# TODO: Add indirect ones
set(linsys_solvers
    ${CMAKE_CURRENT_SOURCE_DIR}/lib_handler.c
    ${CMAKE_CURRENT_SOURCE_DIR}/lib_handler.h
    ${direct_linsys_solvers}
    PARENT_SCOPE)

# Combine solvers external libraries
set(linsys_solvers_includes
    ${direct_linsys_solvers_includes}
    PARENT_SCOPE)
