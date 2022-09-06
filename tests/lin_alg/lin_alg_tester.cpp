/* Linear Algebra Tester Module */

#define CATCH_CONFIG_RUNNER
#include <catch2/catch.hpp>

#include "test_lin_alg.h"

int main( int argc, char* argv[] ) {
    // Setup the algebra
    osqp_algebra_init_libs(0);

    int result = Catch::Session().run( argc, argv );

    // Cleanup the algebra
    osqp_algebra_free_libs();

    return result;
}

/*
TEST_CASE( "test_solve_linsys", "[multi-file:2]" ) {
    osqp_algebra_init_libs(0);
    SECTION( "test_solveKKT" ) {
        test_solveKKT();
    }
    osqp_algebra_free_libs();
}
*/

// void test_quad_form_upper_triang() {

//   OSQPFloat val;
//   lin_alg_sols_data *data = generate_problem_lin_alg_sols_data();
//   OSQPMatrix* P  = OSQPMatrix_new_from_csc(data->test_qpform_Pu, 1); //triu;
//   OSQPVectorf* x = OSQPVectorf_new(data->test_qpform_x, data->test_mat_vec_n);

//   // Compute quadratic form
//   val = OSQPMatrix_quad_form(P, x);

//   mu_assert(
//     "Linear algebra tests: error in computing quadratic form using upper triangular matrix!",
//     (c_absval(val - data->test_qpform_value) < TESTS_TOL));

//   // cleanup
//   OSQPMatrix_free(P);
//   OSQPVectorf_free(x);
//   clean_problem_lin_alg_sols_data(data);
// }
