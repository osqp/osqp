# ----------------------------------------------
# Python requirements
# ----------------------------------------------
find_package(Python)

if(NOT PYTHON_FOUND)
  message(
    FATAL_ERROR
      "You need python installed to generate unittests."
  )
endif()

find_python_module(numpy)
if(NOT NUMPY_FOUND)
  message(
    FATAL_ERROR
      "You need numpy python module installed to generate unittests."
  )
endif()

find_python_module(scipy)
# Check scipy version for sparse.random functionalities
if((NOT SCIPY_FOUND) OR (SCIPY_VERSION VERSION_LESS 0.17.0))
  message(
    FATAL_ERROR
      "You need scipy python module installed to generate unittests."
  )
endif()

# ----------------------------------------------
# Define the C++ standard we will use in our test suite
# ----------------------------------------------
set(CMAKE_CXX_STANDARD 14)

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
# These subfolders populate the variable ${OSQP_TESTCASE_SRCS},
# ${OSQP_TESTCASE_GENERATED_SRCS} and ${OSQP_TESTCASE_DIRS}
# ----------------------------------------------
add_subdirectory(basic_lp)
add_subdirectory(basic_qp)
add_subdirectory(basic_qp2)
add_subdirectory(codegen)
add_subdirectory(large_qp)
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
  OUTPUT ${OSQP_TESTCASE_GENERATED_SRCS}
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/generate_tests_data.py
  DEPENDS ${OSQP_TESTCASE_GENERATORS}
          ${CMAKE_CURRENT_SOURCE_DIR}/generate_tests_data.py
  COMMENT "Generating unittests data files using Python")

# ----------------------------------------------
# Linear algebra tester
# ----------------------------------------------
# Make this a separate tester executable to make developing linear algebra
# libraries easier (this will test at the API between OSQP and a linear algebra
# module).
add_executable(lin_alg_tester
               ${OSQP_LINALG_TESTER_SRCS})

target_include_directories(lin_alg_tester PRIVATE
                           ${OSQP_LINALG_TESTER_INC_DIRS})
target_link_libraries(lin_alg_tester osqpstatic Catch2::Catch2 ${osqplib_link_libs})

add_test(NAME lin_alg_tester COMMAND lin_alg_tester)


# ----------------------------------------------
# osqp_tester
# ----------------------------------------------
add_executable(osqp_tester
               osqp_tester.cpp
               osqp_tester.h
               ${OSQP_TESTCASE_SRCS}
               ${OSQP_TESTCASE_GENERATED_SRCS}
               ${CMAKE_CURRENT_SOURCE_DIR}/utils/test_utils.cpp)
target_include_directories(osqp_tester PRIVATE
                           ${OSQP_TESTCASE_DIRS}
                           ${CMAKE_CURRENT_SOURCE_DIR}
                           ${CMAKE_CURRENT_SOURCE_DIR}/utils/
                           ${CMAKE_CURRENT_SOURCE_DIR}/../include/private
                           ${osqplib_includes})
target_link_libraries(osqp_tester osqpstatic Catch2::Catch2 ${osqplib_link_libs})

add_test(NAME osqp_tester COMMAND osqp_tester)

# ----------------------------------------------
# osqp_tester_custom_memory (only use with builtin algebra)
# ----------------------------------------------
if(OSQP_ALGEBRA_BUILTIN)
  add_executable(osqp_tester_custom_memory
    osqp_tester.cpp
    osqp_tester.h
    ${OSQP_TESTCASE_SRCS}
    ${OSQP_TESTCASE_GENERATED_SRCS}
    ${CMAKE_CURRENT_SOURCE_DIR}/utils/test_utils.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/custom_memory/custom_memory.h
    ${CMAKE_CURRENT_SOURCE_DIR}/custom_memory/custom_memory.c)
  target_include_directories(osqp_tester_custom_memory PRIVATE
                             ${OSQP_TESTCASE_DIRS}
                             ${CMAKE_CURRENT_SOURCE_DIR}
                             ${CMAKE_CURRENT_SOURCE_DIR}/utils/
                             ${CMAKE_CURRENT_SOURCE_DIR}/../include/private
                             ${osqplib_includes})
  target_link_libraries(osqp_tester_custom_memory osqpstatic Catch2::Catch2 ${osqplib_link_libs})

  add_test(NAME osqp_tester_custom_memory COMMAND osqp_tester_custom_memory)
endif()

# ----------------------------------------------
# Create a folder to store the test codes
# ----------------------------------------------
file( MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/testcodes )
file( MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/testcodes/embedded1 )
file( MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/testcodes/embedded2 )

add_compile_definitions(CODEGEN_DIR="${CMAKE_CURRENT_BINARY_DIR}/testcodes/")
add_compile_definitions(CODEGEN1_DIR="${CMAKE_CURRENT_BINARY_DIR}/testcodes/embedded1/")
add_compile_definitions(CODEGEN2_DIR="${CMAKE_CURRENT_BINARY_DIR}/testcodes/embedded2/")

option(OSQP_GENERATE_COVERAGE_REPORT "Generate an HTML coverage report")
mark_as_advanced(OSQP_GENERATE_COVERAGE_REPORT)

if(OSQP_COVERAGE_CHECK AND OSQP_GENERATE_COVERAGE_REPORT)
  find_program(LCOV lcov REQUIRED)
  find_program(GENHTML genhtml REQUIRED)

  set(COVERAGE_REPORT_FILE ${CMAKE_BINARY_DIR}/coverage.info)
  set(COVERAGE_EXCLUDES
      "'${CMAKE_SOURCE_DIR}/tests/*'"
      "'${CMAKE_SOURCE_DIR}/algebra/_common/lin_sys/qdldl/amd/*'"
      "'${CMAKE_SOURCE_DIR}/examples/*'"
      "'${CMAKE_SOURCE_DIR}/build*/*'"
      "'/usr/include/*'")

  add_custom_target(coverage_report
    # Gather data
    COMMAND ${LCOV} --directory ${CMAKE_BINARY_DIR} --capture --output-file ${COVERAGE_REPORT_FILE}
    # Remove data
    COMMAND ${LCOV} --remove ${COVERAGE_REPORT_FILE} ${COVERAGE_EXCLUDES} -o ${COVERAGE_REPORT_FILE}
    # Generate report
    COMMAND ${GENHTML} --demangle-cpp -o coverage ${COVERAGE_REPORT_FILE}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

  set(OSQP_LINALG_BINARY_DIR ${CMAKE_BINARY_DIR}/CMakeFiles/OSQPLIB.dir/algebra)
  set(COVERAGE_LINALG_REPORT_FILE ${CMAKE_BINARY_DIR}/coverage.info)

  add_custom_target(coverage_report_linalg
    # Gather data
    COMMAND ${LCOV} --directory ${OSQP_LINALG_BINARY_DIR} --capture --output-file ${COVERAGE_LINALG_REPORT_FILE}
    # Remove data
    COMMAND ${LCOV} --remove ${COVERAGE_LINALG_REPORT_FILE} ${COVERAGE_EXCLUDES} -o ${COVERAGE_LINALG_REPORT_FILE}
    # Generate report
    COMMAND ${GENHTML} --demangle-cpp -o coverage_linalg ${COVERAGE_LINALG_REPORT_FILE}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
endif()
