# ----------------------------------------------
# lin_alg_tester
# ----------------------------------------------

# Ensure the testcase sources and dependencies are generated properly
get_directory_property(OSQP_TESTCASE_GENERATED_SRCS DIRECTORY ${PROJECT_SOURCE_DIR}/tests DEFINITION OSQP_TESTCASE_GENERATED_SRCS)

set(OSQP_TESTCASE_GENERATED_SRCS
    ${OSQP_TESTCASE_GENERATED_SRCS}
    ${CMAKE_CURRENT_SOURCE_DIR}/lin_alg_data.cpp
    PARENT_SCOPE)

get_directory_property(OSQP_TESTCASE_GENERATORS DIRECTORY ${PROJECT_SOURCE_DIR}/tests DEFINITION OSQP_TESTCASE_GENERATORS)

set(OSQP_TESTCASE_GENERATORS
    ${OSQP_TESTCASE_GENERATORS}
    ${CMAKE_CURRENT_SOURCE_DIR}/generate_problem.py
    PARENT_SCOPE)

set(OSQP_LINALG_TESTER_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/test_mat_vec.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_matrix.cpp

    # Vector tests
    ${CMAKE_CURRENT_SOURCE_DIR}/testcases/test_vector_creation.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/testcases/test_vector_comparisons.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/testcases/test_vector_math_ops.cpp

    # Test data
    ${CMAKE_CURRENT_SOURCE_DIR}/lin_alg_tester.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/lin_alg_data.cpp

    # Other utilities that are needed
    ${CMAKE_CURRENT_SOURCE_DIR}/../utils/test_utils.cpp
    PARENT_SCOPE)

set(OSQP_LINALG_TESTER_INC_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include/private/
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../
    ${CMAKE_CURRENT_SOURCE_DIR}/../utils/
    ${osqplib_includes}
    PARENT_SCOPE)
