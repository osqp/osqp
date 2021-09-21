# Algebra implementation
# ----------------------------------------------

# Directory for custom algebra implementation.  Uses
# OSQP provided CSC implementation by default, which
# also serves as a reference implementation

if (NOT DEFINED ALGEBRA)
  if (CUDA_SUPPORT)
    set(ALGEBRA "cuda")
  else()
    set(ALGEBRA "default")
    endif()
endif()

if(IS_ABSOLUTE ${ALGEBRA})
  set(OSQP_ALGEBRA_DIR ${ALGEBRA})
else()
  set(OSQP_ALGEBRA_DIR ${CMAKE_CURRENT_SOURCE_DIR}/${ALGEBRA})
endif()

#make sure that the algebra target actually exists
if(NOT EXISTS ${OSQP_ALGEBRA_DIR})
  message(FATAL_ERROR "Algebra target directory does not exist:" ${OSQP_ALGEBRA_DIR})
endif()

if(${ALGEBRA} STREQUAL "default")
  message(STATUS "Using OSQP default algebra implementation")
else()
  message(STATUS "Using custom algebra implementation in ${OSQP_ALGEBRA_DIR}")
endif()


add_subdirectory(${ALGEBRA})

if (CUDA_SUPPORT)

  # Create object library for algebra
  add_library(osqp_algebra OBJECT ${algebra_impl})
  target_include_directories(osqp_algebra PRIVATE
                             ${algebra_impl_includes}
                             ${osqp_api_includes})

  # Pass upwards for accesibility from CUDA linear system solvers
  set(algebra_impl_includes "${algebra_impl_includes}" PARENT_SCOPE)

else()

  #include the csc_tools support directory
  #and the build-specific algebra implementation

  add_subdirectory(csc_tools)

  # Create object library for algebra
  add_library(osqp_algebra OBJECT ${algebra_impl} ${csc_tools})
  target_include_directories(osqp_algebra PRIVATE
                             ${algebra_impl_includes}
                             ${csc_tools_includes}
                             ${osqp_api_includes})

  #pass the csc tools include directory upwards
  #for accessibility from linear solvers

  set(csc_tools_includes "${csc_tools_includes}" PARENT_SCOPE)

endif()
