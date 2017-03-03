#include <stdio.h>
#include "osqp.h"
#include "cs.h"
#include "util.h"
#include "minunit.h"
#include "lin_sys.h"


#include "update_matrices/data.h"



static char * test_form_KKT(){
    update_matrices_sols_data *  data;
    c_float rho, sigma;
    c_int * PtoKKT, * AtoKKT, * Pdiag_idx, Pdiag_n;
    csc * Ptriu, *Ptriu_new;
    csc * KKT;

    // Load problem data
    data = generate_problem_update_matrices_sols_data();

    // Define rho and sigma to form KKT
    rho = data->test_form_KKT_rho;
    sigma = data->test_form_KKT_sigma;


    // Allocate vectors of indeces
    PtoKKT = c_malloc((data->test_form_KKT_Pu->p[data->test_form_KKT_Pu->n]) *
                    sizeof(c_int));
    AtoKKT = c_malloc((data->test_form_KKT_A->p[data->test_form_KKT_A->n]) *
                    sizeof(c_int));

    // Form upper triangular part in P
    Ptriu = csc_to_triu(data->test_form_KKT_P);

    // Form KKT matrix storing the index vectors
    KKT = form_KKT(Ptriu, data->test_form_KKT_A, sigma, 1./rho, PtoKKT, AtoKKT, &Pdiag_idx, &Pdiag_n);

    // Assert if KKT matrix is the same as predicted one
    mu_assert("Update matrices: error in forming KKT matrix!", is_eq_csc(KKT, data->test_form_KKT_KKTu, TESTS_TOL));

    // Update KKT matrix with new P and new A
    update_KKT_P(KKT, data->test_form_KKT_Pu_new, PtoKKT, sigma, Pdiag_idx, Pdiag_n);
    update_KKT_A(KKT, data->test_form_KKT_A_new, AtoKKT);


    // Assert if KKT matrix is the same as predicted one
    mu_assert("Update matrices: error in updating KKT matrix!", is_eq_csc(KKT, data->test_form_KKT_KKTu_new, TESTS_TOL));


    // Cleanup
    clean_problem_update_matrices_sols_data(data);
    c_free(Pdiag_idx);
    csc_spfree(Ptriu);
    csc_spfree(KKT);
    c_free(AtoKKT);
    c_free(PtoKKT);
    return 0;
}

static char * test_update(){
    update_matrices_sols_data *  data;
    OSQPData * problem;
    OSQPWorkspace * work;
    OSQPSettings * settings;

    // Update matrix P
    c_float *Px_new;
    c_int *Px_new_idx;
    c_int P_new_n;

    // Update matrix A
    c_float *Ax_new;
    c_int *Ax_new_idx;
    c_int A_new_n;


    // Load problem data
    data = generate_problem_update_matrices_sols_data();

    // Generate first problem data
    problem = c_malloc(sizeof(OSQPData));
    problem->P = data->test_solve_P;
    problem->q = data->test_solve_q;
    problem->A = data->test_solve_A;
    problem->l = data->test_solve_l;
    problem->u = data->test_solve_u;
    problem->n = data->test_solve_P->n;
    problem->m = data->test_solve_A->m;


    // Define Solver settings as default
    // Problem settings
    settings = (OSQPSettings *)c_malloc(sizeof(OSQPSettings));
    set_default_settings(settings);
    settings->max_iter = 1000;
    settings->alpha = 1.6;
    settings->verbose = 0;

    // Setup workspace
    work = osqp_setup(problem, settings);

    // Setup correct
    mu_assert("Update matrices: original problem, setup error!", work != OSQP_NULL);

    // Solve Problem
    osqp_solve(work);

    // Compare solver statuses
    mu_assert("Update matrices: original problem, error in solver status!",
              work->info->status_val == data->test_solve_status );

    // Compare primal solutions
    mu_assert("Update matrices: original problem, error in primal solution!",
              vec_norm2_diff(work->solution->x, data->test_solve_x, data->n) < TESTS_TOL);

    // Compare dual solutions
    mu_assert("Update matrices: original problem, error in dual solution!",
              vec_norm2_diff(work->solution->y, data->test_solve_y, data->m) < TESTS_TOL);



    // Update P

    osqp_update_P(work, c_float *Px_new, c_int *Px_new_idx, c_int P_new_n)




    osqp_cleanup(work);

    // Cleanup problems
    clean_problem_update_matrices_sols_data(data);
    c_free(problem);

    return 0;
}



static char * test_update_matrices()
{
    mu_run_test(test_form_KKT);
    mu_run_test(test_update);

    return 0;
}
