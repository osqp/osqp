#include "osqp.h"     // OSQP API
#include "cs.h"       // CSC data structure
#include "util.h"     // Utilities for testing
#include "minunit.h"  // Basic testing script header

#include "basic_qp/data.h"


static char * test_basic_qp_solve()
{
    // Problem settings
    OSQPSettings * settings = (OSQPSettings *)c_malloc(sizeof(OSQPSettings));

    // Structures
    OSQPWorkspace * work;  // Workspace
    OSQPData * data;  // Data
    basic_qp_sols_data *  sols_data;

    // Populate data
    data = generate_problem_basic_qp();
    sols_data = generate_problem_basic_qp_sols_data();


    // Define Solver settings as default
    set_default_settings(settings);
    settings->max_iter = 2000;
    settings->alpha = 1.6;
    settings->polish = 1;
    settings->auto_rho = 0;
    settings->scaling = 0;
    settings->verbose = 1;
    settings->warm_start = 0;

    // Setup workspace
    work = osqp_setup(data, settings);

    // Setup correct
    mu_assert("Basic QP test solve: Setup error!", work != OSQP_NULL);

    // Solve Problem
    osqp_solve(work);

    // Compare solver statuses
    mu_assert("Basic QP test solve: Error in solver status!",
              work->info->status_val == sols_data->status_test );

    // Compare primal solutions
    mu_assert("Basic QP test solve: Error in primal solution!",
              vec_norm_inf_diff(work->solution->x, sols_data->x_test, data->n) < TESTS_TOL);

    // Compare dual solutions
    mu_assert("Basic QP test solve: Error in dual solution!",
              vec_norm_inf_diff(work->solution->y, sols_data->y_test, data->m) < TESTS_TOL);


    // Compare objective values
    mu_assert("Basic QP test solve: Error in objective value!",
              c_absval(work->info->obj_val - sols_data->obj_value_test) < TESTS_TOL);

    // Try to set wrong settings
    mu_assert("Basic QP test solve: Wrong value of rho not caught!",
              osqp_update_rho(work, -0.1) == 1);

    mu_assert("Basic QP test solve: Wrong value of max_iter not caught!",
              osqp_update_max_iter(work, -1) == 1);

    mu_assert("Basic QP test solve: Wrong value of eps_abs not caught!",
              osqp_update_eps_abs(work, 0.) == 1);

    mu_assert("Basic QP test solve: Wrong value of eps_rel not caught!",
              osqp_update_eps_rel(work, 0.) == 1);

    mu_assert("Basic QP test solve: Wrong value of eps_prim_inf not caught!",
              osqp_update_eps_prim_inf(work, -0.1) == 1);

    mu_assert("Basic QP test solve: Wrong value of eps_dual_inf not caught!",
              osqp_update_eps_dual_inf(work, -0.1)== 1);

    mu_assert("Basic QP test solve: Wrong value of alpha not caught!",
              osqp_update_alpha(work, 2.0) == 1);

    mu_assert("Basic QP test solve: Wrong value of warm_start not caught!",
              osqp_update_warm_start(work, -1) == 1);

    mu_assert("Basic QP test solve: Wrong value of scaled_termination not caught!",
              osqp_update_scaled_termination(work, 2) == 1);

    mu_assert("Basic QP test solve: Wrong value of early_terminate not caught!",
              osqp_update_early_terminate(work, 5) == 1);

    mu_assert("Basic QP test solve: Wrong value of delta not caught!",
              osqp_update_delta(work, 0.) == 1);

    mu_assert("Basic QP test solve: Wrong value of polish not caught!",
              osqp_update_polish(work, 2) == 1);

    mu_assert("Basic QP test solve: Wrong value of pol_refine_iter not caught!",
              osqp_update_pol_refine_iter(work, -1) == 1);

    mu_assert("Basic QP test solve: Wrong value of verbose not caught!",
              osqp_update_verbose(work, 2) == 1);


    // Clean workspace
    osqp_cleanup(work);


    // Cleanup data
    clean_problem_basic_qp(data);
    clean_problem_basic_qp_sols_data(sols_data);

    // Cleanup
    c_free(settings);

    return 0;
}


static char * test_basic_qp_solve_pardiso()
{
    // Problem settings
    OSQPSettings * settings = (OSQPSettings *)c_malloc(sizeof(OSQPSettings));

    // Structures
    OSQPWorkspace * work;  // Workspace
    OSQPData * data;  // Data
    basic_qp_sols_data *  sols_data;

    // Populate data
    data = generate_problem_basic_qp();
    sols_data = generate_problem_basic_qp_sols_data();


    // Define Solver settings as default
    set_default_settings(settings);
    settings->max_iter = 2000;
    settings->alpha = 1.6;
    settings->polish = 1;
    settings->auto_rho = 0;
    settings->scaling = 0;
    settings->verbose = 1;
    settings->warm_start = 0;
    settings->linsys_solver = PARDISO_SOLVER;

    // Setup workspace
    work = osqp_setup(data, settings);

    // Setup correct
    mu_assert("Basic QP test solve: Setup error!", work != OSQP_NULL);

    // Solve Problem
    osqp_solve(work);

    // Compare solver statuses
    mu_assert("Basic QP test solve: Error in solver status!",
              work->info->status_val == sols_data->status_test );

    // Compare primal solutions
    mu_assert("Basic QP test solve: Error in primal solution!",
              vec_norm_inf_diff(work->solution->x, sols_data->x_test, data->n) < TESTS_TOL);

    // Compare dual solutions
    mu_assert("Basic QP test solve: Error in dual solution!",
              vec_norm_inf_diff(work->solution->y, sols_data->y_test, data->m) < TESTS_TOL);


    // Compare objective values
    mu_assert("Basic QP test solve: Error in objective value!",
              c_absval(work->info->obj_val - sols_data->obj_value_test) < TESTS_TOL);

    // Clean workspace
    osqp_cleanup(work);


    // Cleanup data
    clean_problem_basic_qp(data);
    clean_problem_basic_qp_sols_data(sols_data);

    // Cleanup
    c_free(settings);

    return 0;
}


static char * test_basic_qp_update()
{
    // Problem settings
    OSQPSettings * settings = (OSQPSettings *)c_malloc(sizeof(OSQPSettings));

    // Structures
    OSQPWorkspace * work;  // Workspace
    OSQPData * data;  // Data
    basic_qp_sols_data *  sols_data;

    // Populate data
    data = generate_problem_basic_qp();
    sols_data = generate_problem_basic_qp_sols_data();


    // Define Solver settings as default
    set_default_settings(settings);
    settings->max_iter = 200;
    settings->alpha = 1.6;
    settings->polish = 1;
    settings->scaling = 0;
    settings->verbose = 1;
    settings->warm_start = 0;

    // Setup workspace
    work = osqp_setup(data, settings);

    // Setup correct
    mu_assert("Basic QP test update: Setup error!", work != OSQP_NULL);


    // ====================================================================
    //  Update data
    // ====================================================================

    // Update linear cost
    osqp_update_lin_cost(work, sols_data->q_new);
    mu_assert("Basic QP test update: Error in updating linear cost!",
              vec_norm_inf_diff(work->data->q, sols_data->q_new, data->n) < TESTS_TOL);

    // UPDATE BOUND
    // Try to update with non-consistent values
    mu_assert("Basic QP test update: Error in bounds update ordering not caught!",
              osqp_update_bounds(work, sols_data->u_new, sols_data->l_new) == 1);

    // Now update with correct values
    mu_assert("Basic QP test update: Error in bounds update ordering!",
              osqp_update_bounds(work, sols_data->l_new, sols_data->u_new) == 0);

    mu_assert("Basic QP test update: Error in bounds update, lower bound!",
              vec_norm_inf_diff(work->data->l, sols_data->l_new, data->m) < TESTS_TOL);

    mu_assert("Basic QP test update: Error in bounds update, upper bound!",
              vec_norm_inf_diff(work->data->u, sols_data->u_new, data->m) < TESTS_TOL);

    // Return original values
    osqp_update_bounds(work, data->l, data->u);



    // UPDATE LOWER BOUND
    // Try to update with non-consistent values
    mu_assert("Basic QP test update: Error in lower bound update ordering not caught!",
              osqp_update_lower_bound(work, sols_data->u_new) == 1);

    // Now update with correct values
    mu_assert("Basic QP test update: Error in lower bound update ordering!",
              osqp_update_lower_bound(work, sols_data->l_new) == 0);

    mu_assert("Basic QP test update: Error in updating lower bound!",
              vec_norm_inf_diff(work->data->l, sols_data->l_new, data->m) < TESTS_TOL);

    // Return original values
    osqp_update_lower_bound(work, data->l);



    // UPDATE UPPER BOUND
    // Try to update with non-consistent values
    mu_assert("Basic QP test update: Error in upper bound update: ordering not caught!",
              osqp_update_upper_bound(work, sols_data->l_new) == 1);

    // Now update with correct values
    mu_assert("Basic QP test update: Error in upper bound update: ordering!",
              osqp_update_upper_bound(work, sols_data->u_new) == 0);

    mu_assert("Basic QP test update: Error in updating upper bound!",
            vec_norm_inf_diff(work->data->u, sols_data->u_new, data->m) < TESTS_TOL);


    // Clean workspace
    osqp_cleanup(work);


    // Cleanup data
    clean_problem_basic_qp(data);
    clean_problem_basic_qp_sols_data(sols_data);

    // Cleanup
    c_free(settings);

    return 0;
}



static char * test_basic_qp_early_terminate()
{
    // Problem settings
    OSQPSettings * settings = (OSQPSettings *)c_malloc(sizeof(OSQPSettings));

    // Structures
    OSQPWorkspace * work;  // Workspace
    OSQPData * data;  // Data
    basic_qp_sols_data *  sols_data;

    // Populate data
    data = generate_problem_basic_qp();
    sols_data = generate_problem_basic_qp_sols_data();


    // Define Solver settings as default
    set_default_settings(settings);
    settings->max_iter = 200;
    settings->alpha = 1.6;
    settings->polish = 0;
    settings->scaling = 0;
    settings->verbose = 1;
    settings->early_terminate = 0;
    settings->warm_start = 0;

    // Setup workspace
    work = osqp_setup(data, settings);

    // Setup correct
    mu_assert("Basic QP test solve: Setup error!", work != OSQP_NULL);

    // Solve Problem
    osqp_solve(work);

    // Check if iter == max_iter
    mu_assert("Basic QP test early terminate: Error in number of iterations taken!",
              work->info->iter == work->settings->max_iter );

    // Compare solver statuses
    mu_assert("Basic QP test early terminate: Error in solver status!",
              work->info->status_val == sols_data->status_test );

    // Compare primal solutions
    mu_assert("Basic QP test early terminate: Error in primal solution!",
              vec_norm_inf_diff(work->solution->x, sols_data->x_test, data->n) < TESTS_TOL);

    // Compare dual solutions
    print_vec(work->solution->y, data->m, "y_sol");
    print_vec(sols_data->y_test, data->m, "y_test");
    mu_assert("Basic QP test early terminate: Error in dual solution!",
              vec_norm_inf_diff(work->solution->y, sols_data->y_test, data->m) < TESTS_TOL);

    // Compare objective values
    mu_assert("Basic QP test early terminate: Error in objective value!",
              c_absval(work->info->obj_val - sols_data->obj_value_test) < TESTS_TOL);

    // Clean workspace
    osqp_cleanup(work);


    // Cleanup data
    clean_problem_basic_qp(data);
    clean_problem_basic_qp_sols_data(sols_data);

    // Cleanup
    c_free(settings);

    return 0;
}



static char * test_basic_qp_update_rho()
{
    // Problem settings
    OSQPSettings * settings = (OSQPSettings *)c_malloc(sizeof(OSQPSettings));

    // Structures
    OSQPWorkspace * work;  // Workspace
    OSQPData * data;  // Data
    basic_qp_sols_data *  sols_data;

    // Exitflag
    c_int exitflag;

    // rho to use
    c_float rho;

    // Define number of iterations to compare
    c_int n_iter_new_solver, n_iter_update_rho;

    // Populate data
    data = generate_problem_basic_qp();
    sols_data = generate_problem_basic_qp_sols_data();


    // Define Solver settings as default
    rho = 0.7;
    set_default_settings(settings);
    settings->rho = rho;
    settings->eps_abs = 1e-05;
    settings->eps_rel = 1e-05;
    settings->early_terminate_interval = 1;

    // Setup workspace
    work = osqp_setup(data, settings);

    // Setup correct
    mu_assert("Update rho test solve: Setup error!", work != OSQP_NULL);

    // Solve Problem
    osqp_solve(work);

    // Store number of iterations
    n_iter_new_solver = work->info->iter;

    // Compare solver statuses
    mu_assert("Update rho test solve: Error in solver status!",
              work->info->status_val == sols_data->status_test );

    // Compare primal solutions
    mu_assert("Update rho test solve: Error in primal solution!",
              vec_norm_inf_diff(work->solution->x, sols_data->x_test, data->n) < TESTS_TOL);

    // Compare dual solutions
    mu_assert("Update rho test solve: Error in dual solution!",
              vec_norm_inf_diff(work->solution->y, sols_data->y_test, data->m) < TESTS_TOL);

    // Compare objective values
    mu_assert("Update rho test solve: Error in objective value!",
              c_absval(work->info->obj_val - sols_data->obj_value_test) < TESTS_TOL);

    // Clean workspace
    osqp_cleanup(work);



    // Create new problem with different rho and update it
    set_default_settings(settings);
    settings->rho = 0.1;
    settings->early_terminate_interval = 1;
    settings->eps_abs = 1e-05;
    settings->eps_rel = 1e-05;

    // Setup workspace
    work = osqp_setup(data, settings);

    // Setup correct
    mu_assert("Update rho test update: Setup error!", work != OSQP_NULL);

    // Update rho
    exitflag = osqp_update_rho(work, rho);
    mu_assert("Update rho test update: Error update rho!", exitflag == 0);

    // Solve Problem
    osqp_solve(work);

    // Compare solver statuses
    mu_assert("Update rho test update: Error in solver status!",
              work->info->status_val == sols_data->status_test );

    // Compare primal solutions
    mu_assert("Update rho test update: Error in primal solution!",
              vec_norm_inf_diff(work->solution->x, sols_data->x_test, data->n) < TESTS_TOL);

    // Compare dual solutions
    mu_assert("Update rho test update: Error in dual solution!",
              vec_norm_inf_diff(work->solution->y, sols_data->y_test, data->m) < TESTS_TOL);

    // Compare objective values
    mu_assert("Update rho test update: Error in objective value!",
              c_absval(work->info->obj_val - sols_data->obj_value_test) < TESTS_TOL);

    // Get number of iterations
    n_iter_update_rho = work->info->iter;

    // Assert same number of iterations
    mu_assert("Update rho test update: Error in number of iterations!",
              n_iter_new_solver == n_iter_update_rho);

    // Cleanup solver
    osqp_cleanup(work);

    // Cleanup data
    clean_problem_basic_qp(data);
    clean_problem_basic_qp_sols_data(sols_data);

    // Cleanup
    c_free(settings);

    return 0;
}

static char * test_basic_qp()
{

    mu_run_test(test_basic_qp_solve);
    mu_run_test(test_basic_qp_solve_pardiso);
    mu_run_test(test_basic_qp_update);
    mu_run_test(test_basic_qp_early_terminate);
    mu_run_test(test_basic_qp_update_rho);

    return 0;
}
