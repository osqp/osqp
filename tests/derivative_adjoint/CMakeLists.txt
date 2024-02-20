# The derivatives are only implemented for built-in algebra currently
if( OSQP_ALGEBRA_BUILTIN )
    get_directory_property(OSQP_TESTCASE_SRCS DIRECTORY ${PROJECT_SOURCE_DIR}/tests DEFINITION OSQP_TESTCASE_SRCS)

    # The data was actually generated from a codegen example, so we have no Python to generate the code
    set(OSQP_TESTCASE_SRCS
        ${OSQP_TESTCASE_SRCS}
        ${CMAKE_CURRENT_SOURCE_DIR}/test_derivative_adjoint.cpp
        PARENT_SCOPE)

        get_directory_property(OSQP_TESTCASE_GENERATED_SRCS DIRECTORY ${PROJECT_SOURCE_DIR}/tests DEFINITION OSQP_TESTCASE_GENERATED_SRCS)

    set(OSQP_TESTCASE_GENERATED_SRCS
        ${OSQP_TESTCASE_GENERATED_SRCS}
        ${CMAKE_CURRENT_SOURCE_DIR}/derivative_adjoint_data.cpp
        PARENT_SCOPE)

    get_directory_property(OSQP_TESTCASE_GENERATORS DIRECTORY ${PROJECT_SOURCE_DIR}/tests DEFINITION OSQP_TESTCASE_GENERATORS)

    set(OSQP_TESTCASE_GENERATORS
        ${OSQP_TESTCASE_GENERATORS}
        ${CMAKE_CURRENT_SOURCE_DIR}/generate_problem.py
        PARENT_SCOPE)

    get_directory_property(OSQP_TESTCASE_DIRS DIRECTORY ${PROJECT_SOURCE_DIR}/tests DEFINITION OSQP_TESTCASE_DIRS)

    set(OSQP_TESTCASE_DIRS
        ${OSQP_TESTCASE_DIRS}
        ${CMAKE_CURRENT_SOURCE_DIR}
        PARENT_SCOPE)
endif()