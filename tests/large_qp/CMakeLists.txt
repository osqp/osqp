get_directory_property(OSQP_TESTCASE_SRCS DIRECTORY ${PROJECT_SOURCE_DIR}/tests DEFINITION OSQP_TESTCASE_SRCS)

# The data was actually generated from a codegen example, so we have no Python to generate the code
set(OSQP_TESTCASE_SRCS
    ${OSQP_TESTCASE_SRCS}
    ${CMAKE_CURRENT_SOURCE_DIR}/test_large_qp.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/large_qp_data.c
    PARENT_SCOPE)


get_directory_property(OSQP_TESTCASE_DIRS DIRECTORY ${PROJECT_SOURCE_DIR}/tests DEFINITION OSQP_TESTCASE_DIRS)

set(OSQP_TESTCASE_DIRS
    ${OSQP_TESTCASE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}
    PARENT_SCOPE)
