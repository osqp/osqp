/* Linear Algebra Tester Module */

#define CATCH_CONFIG_RUNNER
#include <catch2/catch.hpp>

#include "test_lin_alg.h"

#include <cstdlib>
#include <iostream>

int main( int argc, char* argv[] ) {
    Catch::Session session;

    int result = 0;
    int deviceNum = 0;

    /*
     * Select the device to run the test on.
     */

    // Start by looking for the environment variable
    if(const char* deviceEnv = std::getenv("OSQP_TEST_DEVICE_NUM")) {
        deviceNum = std::atoi(deviceEnv);
    }

    using namespace Catch::clara;

    // Next use a command line option (if it exists)
    auto cli = session.cli() | Opt(deviceNum, "device")["--device"]("Compute device (overrides the OSQP_TEST_DEVICE_NUM environment variable)");
    session.cli(cli);

    result = session.applyCommandLine( argc, argv );

    if( result != 0 ) {
        // Indicates a command line error
        std::cout << "Error configuring command line for test driver" << std::endl;
        return result;
    }

    /*
     * Configure and run the OSQP test suite
     */

    // Setup the algebra
    result = osqp_algebra_init_libs(deviceNum);

    if( result != 0) {
        std::cout << "Error configuring linear algebra library" << std::endl;
        return result;
    }

    result = session.run( argc, argv );

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
