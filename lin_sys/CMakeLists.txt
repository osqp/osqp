# Add subdirectory for each linear systems solver

if (NOT DEFINED EMBEDDED)

  # Include this directory for library handler
  # NB: Needed by the subfolders
  target_include_directories(OSQPLIB
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR})

endif()


if (ALGEBRA_CUDA)

  # CUDA solvers
  add_subdirectory(cuda)

  set(linsys_solvers ${cuda_linsys_solvers})
  set(linsys_solvers_includes ${cuda_linsys_solvers_includes})

  set(linsys_solvers ${linsys_solvers} PARENT_SCOPE)
  set(linsys_solvers_includes ${linsys_solvers_includes} PARENT_SCOPE)

else()

  add_subdirectory(direct)
  add_subdirectory(indirect)

endif()

target_include_directories(OSQPLIB
        PRIVATE
        ${linsys_solvers_includes})

# Pass upwards
set(linsys_solvers ${linsys_solvers} PARENT_SCOPE)
set(linsys_solvers_includes ${linsys_solvers_includes} PARENT_SCOPE)