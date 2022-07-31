if(NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${ALGEBRA})
  message(FATAL_ERROR "Algebra target directory does not exist:" ${ALGEBRA})
endif()

set(OSQP_ALGEBRA_ROOT ${CMAKE_CURRENT_SOURCE_DIR})

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/${ALGEBRA})
