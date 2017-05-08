#include <stdio.h>
#include "osqp.h"
#include "cs.h"
#include "util.h"
#include "minunit.h"
#include "lin_sys.h"


#include "solve_linsys/data.h"



static char * test_solveKKT(){
    c_int exitflag=0;
    Priv * p; // Private structure to form KKT factorization
    OSQPSettings *settings = (OSQPSettings *)c_malloc(sizeof(OSQPSettings)); // Settings

    solve_linsys_sols_data *  data = generate_problem_solve_linsys_sols_data();

    // Form and factorize KKT matrix
    settings->rho = data->test_solve_KKT_rho;
    settings->sigma = data->test_solve_KKT_sigma;
    p = init_priv(data->test_solve_KKT_Pu, data->test_solve_KKT_A, settings, 0);

    // Debug print KKT and LDL
    // print_csc_mat(data->test_solve_KKT_KKT, "KKTpy");
    // print_csc_mat(p->KKT, "KKT");
    // c_float * KKTdnspy = csc_to_dns(data->test_solve_KKT_KKT);
    // c_float * KKTdns = csc_to_dns(p->KKT);
    // print_dns_matrix(KKTdnspy, data->test_solve_KKT_KKT->m, data->test_solve_KKT_KKT->n, "KKTdnspy");
    // print_dns_matrix(KKTdns, p->KKT->m, p->KKT->n, "KKTdns");


    // Solve  KKT x = b via LDL given factorization
    solve_lin_sys(settings, p, data->test_solve_KKT_rhs);

    mu_assert("Linear systems solve tests: error in forming and solving KKT system!",
              vec_norm_inf_diff(data->test_solve_KKT_rhs, data->test_solve_KKT_x,
                                data->test_solve_KKT_m + data->test_solve_KKT_n) < TESTS_TOL);


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
