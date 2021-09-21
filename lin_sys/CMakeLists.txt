# Add subdirectory for each linear systems solver

if (NOT DEFINED EMBEDDED)

# Include this directory for library handler
# NB Needed for subfolders
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

endif()


if (CUDA_SUPPORT)

  # CUDA solvers
  add_subdirectory(cuda)

  set(linsys_solvers ${cuda_linsys_solvers})
  set(linsys_solvers_includes ${cuda_linsys_solvers_includes})

else()

  # Direct solvers
  add_subdirectory(direct)

  # Indirect solvers
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
      ${linsys_lib_handler}
  )

  # Combine solvers external libraries
  set(linsys_solvers_includes ${direct_linsys_solvers_includes})

endif()


# Pass upwards
set(linsys_solvers ${linsys_solvers} PARENT_SCOPE)
set(linsys_solvers_includes ${linsys_solvers_includes} PARENT_SCOPE)
