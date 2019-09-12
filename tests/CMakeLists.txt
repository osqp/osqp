# Add custom command to generate tests
# ----------------------------------------------
find_package(PythonInterp)
if(NOT PYTHONINTERP_FOUND)
    message( FATAL_ERROR "You need python installed to generate unittests. If you do not want to compile the unittests pass -DUNITTESTS=OFF to cmake." )
endif()

INCLUDE(FindPythonModule)
find_python_module(numpy)
IF(NOT NUMPY_FOUND)
    message( FATAL_ERROR "You need numpy python module installed to generate unittests. If you do not want to compile the unittests pass -DUNITTESTS=OFF to cmake." )
ENDIF()

find_python_module(scipy)
# Check scipy version for sparse.random functionalities
IF((NOT SCIPY_FOUND) OR (SCIPY_VERSION VERSION_LESS 0.17.0))
    message( FATAL_ERROR "You need scipy python module installed to generate unittests. If you do not want to compile the unittests pass -DUNITTESTS=OFF to cmake." )
ENDIF()

find_python_module(__future__)
IF(NOT __FUTURE___FOUND)
    message( FATAL_ERROR "You need future python module installed to generate unittests. If you do not want to compile the unittests pass -DUNITTESTS=OFF to cmake." )
ENDIF()

# Add subdirectories of all the tests
#------------------------------------------
add_subdirectory(basic_qp)
add_subdirectory(basic_qp2)
add_subdirectory(lin_alg)
add_subdirectory(non_cvx)
add_subdirectory(primal_dual_infeasibility)
add_subdirectory(primal_infeasibility)
add_subdirectory(solve_linsys)
add_subdirectory(unconstrained)
add_subdirectory(update_matrices)

# Add test_headers and codegen_test_headers
#------------------------------------------
set(test_headers ${headers})
set(codegen_test_headers ${codegen_headers})

# Generating tests.stamp so that the test data are not always generated
# set(data_timestamp ${CMAKE_CURRENT_SOURCE_DIR}/tests_data.stamp)
add_custom_command(
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMAND ${PYTHON_EXECUTABLE} generate_tests_data.py
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/generate_tests_data.py
    OUTPUT ${codegen_test_headers}
    COMMENT "Generating unittests data files using Python"
)

add_executable(osqp_tester
            ${CMAKE_CURRENT_SOURCE_DIR}/osqp_tester.c
            ${CMAKE_CURRENT_SOURCE_DIR}/osqp_tester.h
            ${CMAKE_CURRENT_SOURCE_DIR}/minunit.h
            ${test_headers}
            ${codegen_test_headers})
target_include_directories(osqp_tester PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(osqp_tester osqpstatic)

#since osqp_tester exercises internal OSQP functions,
#extract the include directories for OSQPLIB and add
#them to the include paths for osqp_tester
get_property(osqpinternalincludes TARGET OSQPLIB PROPERTY INCLUDE_DIRECTORIES)
target_include_directories(osqp_tester
                           PRIVATE "${osqpinternalincludes}")


# Add custom memory target
add_executable(osqp_tester_custom_memory
               EXCLUDE_FROM_ALL
               ${CMAKE_CURRENT_SOURCE_DIR}/osqp_tester.c
               ${CMAKE_CURRENT_SOURCE_DIR}/osqp_tester.h
               ${CMAKE_CURRENT_SOURCE_DIR}/minunit.h
               ${codegen_test_headers}
   ${CMAKE_CURRENT_SOURCE_DIR}/custom_memory/custom_memory.c
   ${CMAKE_CURRENT_SOURCE_DIR}/custom_memory/custom_memory.h
)
target_link_libraries(osqp_tester_custom_memory osqpstatic)
target_include_directories(osqp_tester_custom_memory
                           PRIVATE "${osqpinternalincludes}")

# Add testing
include(CTest)
enable_testing()
add_test(NAME tester COMMAND $<TARGET_FILE:osqp_tester>)
