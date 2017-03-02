#include <stdio.h>
#include "osqp.h"
#include "cs.h"
#include "util.h"
#include "minunit.h"
#include "lin_sys.h"


#include "solve_linsys/solve_linsys.h"



static char * test_solveKKT(){
    c_int exitflag=0;
    Priv * p; // Private structure to form KKT factorization
    OSQPSettings *settings = (OSQPSettings *)c_malloc(sizeof(OSQPSettings)); // Settings

    solve_linsys_sols_data *  data = generate_problem_solve_linsys_sols_data();

    // Form and factorize KKT matrix
    settings->rho = data->test_solve_KKT_rho;
    settings->sigma = data->test_solve_KKT_sigma;
    p = init_priv(data->test_solve_KKT_Pu, data->test_solve_KKT_A, settings, 0);

    // Solve  KKT x = b via LDL given factorization
    solve_lin_sys(settings, p, data->test_solve_KKT_rhs);

    mu_assert("Linear systems solve tests: error in forming and solving KKT system!", vec_norm2_diff(data->test_solve_KKT_rhs, data->test_solve_KKT_x, data->test_solve_KKT_m + data->test_solve_KKT_n) < TESTS_TOL);


    // Cleanup
    free_priv(p);
    c_free(settings);
    clean_problem_solve_linsys_sols_data(data);

    return 0;
}


static char * test_solve_linsys()
{
    mu_run_test(test_solveKKT);

    return 0;
}
