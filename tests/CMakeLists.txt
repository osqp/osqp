# ----------------------------------------------
# Python requirements
# ----------------------------------------------
find_package(Python)

if(NOT PYTHON_FOUND)
  message(
    FATAL_ERROR
      "You need python installed to generate unittests. If you do not want to compile the unittests pass -DBUILD_TESTING=OFF to cmake."
  )
endif()

find_python_module(numpy)
if(NOT NUMPY_FOUND)
  message(
    FATAL_ERROR
      "You need numpy python module installed to generate unittests. If you do not want to compile the unittests pass -DBUILD_TESTING=OFF to cmake."
  )
endif()

find_python_module(scipy)
# Check scipy version for sparse.random functionalities
if((NOT SCIPY_FOUND) OR (SCIPY_VERSION VERSION_LESS 0.17.0))
  message(
    FATAL_ERROR
      "You need scipy python module installed to generate unittests. If you do not want to compile the unittests pass -DBUILD_TESTING=OFF to cmake."
  )
endif()

# ----------------------------------------------
# Catch2
# ----------------------------------------------
include(FetchContent)

FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG v2.13.6)
FetchContent_MakeAvailable(Catch2)
# Adds Catch2::Catch2

# ----------------------------------------------
# Test Inclusion
# ----------------------------------------------
# These subfolders populate the variables ${headers} and ${codegen_headers}
# ----------------------------------------------
add_subdirectory(basic_qp)
add_subdirectory(basic_qp2)
add_subdirectory(lin_alg)
add_subdirectory(non_cvx)
add_subdirectory(primal_dual_infeasibility)
add_subdirectory(primal_infeasibility)
add_subdirectory(solve_linsys)
add_subdirectory(unconstrained)
add_subdirectory(update_matrices)

# ----------------------------------------------
# Test Data Generation
# ----------------------------------------------
add_custom_command(
  OUTPUT ${codegen_headers}
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/generate_tests_data.py
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/generate_tests_data.py
  COMMENT "Generating unittests data files using Python")

# ----------------------------------------------
# osqp_tester
# ----------------------------------------------
add_executable(osqp_tester osqp_tester.cpp osqp_tester.h ${headers} ${codegen_headers})
target_include_directories(osqp_tester PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${osqplib_includes})
target_link_libraries(osqp_tester osqpstatic Catch2::Catch2 ${osqplib_link_libs})

add_test(NAME osqp_tester COMMAND osqp_tester)

# ----------------------------------------------
# osqp_tester_custom_memory
# ----------------------------------------------
add_executable(
  osqp_tester_custom_memory
  osqp_tester.cpp osqp_tester.h ${headers} ${codegen_headers} ${CMAKE_CURRENT_SOURCE_DIR}/custom_memory/custom_memory.h
  ${CMAKE_CURRENT_SOURCE_DIR}/custom_memory/custom_memory.c)
target_include_directories(osqp_tester_custom_memory PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${osqplib_includes})
target_link_libraries(osqp_tester_custom_memory osqpstatic Catch2::Catch2 ${osqplib_link_libs})

add_test(NAME osqp_tester_custom_memory COMMAND osqp_tester_custom_memory)

# ----------------------------------------------
# Create a folder to store the test codes
# ----------------------------------------------
file( MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/testcodes )
file( MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/testcodes/embedded1 )
file( MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/testcodes/embedded2 )

add_compile_definitions(CODEGEN_DIR="${CMAKE_CURRENT_BINARY_DIR}/testcodes/")
add_compile_definitions(CODEGEN1_DIR="${CMAKE_CURRENT_BINARY_DIR}/testcodes/embedded1/")
add_compile_definitions(CODEGEN2_DIR="${CMAKE_CURRENT_BINARY_DIR}/testcodes/embedded2/")
