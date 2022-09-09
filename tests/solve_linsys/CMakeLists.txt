get_directory_property(headers DIRECTORY ${PROJECT_SOURCE_DIR}/tests DEFINITION headers)

set(headers
    ${headers} ${CMAKE_CURRENT_SOURCE_DIR}/test_solve_linsys.h
    PARENT_SCOPE)

get_directory_property(OSQP_TESTCASE_GENERATORS DIRECTORY ${PROJECT_SOURCE_DIR}/tests DEFINITION OSQP_TESTCASE_GENERATORS)

set(OSQP_TESTCASE_GENERATORS
    ${OSQP_TESTCASE_GENERATORS}
    ${CMAKE_CURRENT_SOURCE_DIR}/generate_problem.py
    PARENT_SCOPE)

get_directory_property(codegen_headers DIRECTORY ${PROJECT_SOURCE_DIR}/tests DEFINITION codegen_headers)

set(codegen_headers
    ${codegen_headers} ${CMAKE_CURRENT_SOURCE_DIR}/data.h
    PARENT_SCOPE)
