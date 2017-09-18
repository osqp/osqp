# Add subdirectory for each linear systems solver

# Direct solver
add_subdirectory(direct)

# Indirect solver
# TODO: Add!


# Combine solvers
# TODO: Add indirect ones
set(linsys_solvers ${direct_linsys_solvers} PARENT_SCOPE)

# Combine solvers external libraries
set(linsys_solvers_libraries ${direct_linsys_solvers_libraries} PARENT_SCOPE)
