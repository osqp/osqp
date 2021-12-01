if(NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${ALGEBRA})
  message(FATAL_ERROR "Algebra target directory does not exist:" ${ALGEBRA})
endif()

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/${ALGEBRA})
