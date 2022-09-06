# ----------------------------------------------
# lin_alg_tester
# ----------------------------------------------

# Make this a separate tester executable to make developing linear algebra
# libraries easier (this will test at the API between OSQP and a linear algebra
# module).
add_executable(lin_alg_tester
               lin_alg_tester.cpp
               lin_alg_data.cpp
               test_mat_vec.cpp
               test_matrix.cpp
               test_vector.cpp
               ${CMAKE_CURRENT_SOURCE_DIR}/../utils/test_utils.cpp)

target_include_directories(lin_alg_tester PRIVATE
                           ${CMAKE_CURRENT_SOURCE_DIR}/../../include/private/
                           ${CMAKE_CURRENT_SOURCE_DIR}
                           ${CMAKE_CURRENT_SOURCE_DIR}/../
                           ${CMAKE_CURRENT_SOURCE_DIR}/../utils/
                           ${osqplib_includes})
target_link_libraries(lin_alg_tester osqpstatic Catch2::Catch2 ${osqplib_link_libs})

add_test(NAME lin_alg_tester COMMAND lin_alg_tester)
