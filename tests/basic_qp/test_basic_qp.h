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
              vec_norm2_diff(work->solution->x, sols_data->x_test, data->n) < TESTS_TOL);

    // Compare dual solutions
    mu_assert("Basic QP test solve: Error in dual solution!",
              vec_norm2_diff(work->solution->y, sols_data->y_test, data->m) < TESTS_TOL);


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
              vec_norm2_diff(work->data->q, sols_data->q_new, data->n) < TESTS_TOL);

    // UPDATE BOUNDS
    mu_assert("Basic QP test update: Error in bounds update ordering!",
              osqp_update_bounds(work, sols_data->l_new, sols_data->u_new) == 0);

    mu_assert("Basic QP test update: Error in bounds update, lower bound!",
              vec_norm2_diff(work->data->l, sols_data->l_new, data->m) < TESTS_TOL);

    mu_assert("Basic QP test update: Error in bounds update, upper bound!",
              vec_norm2_diff(work->data->u, sols_data->u_new, data->m) < TESTS_TOL);

    // Update lower bound
    mu_assert("Basic QP test update: Error in lower bound update. ordering!",
              osqp_update_lower_bound(work, sols_data->l_new) == 0);

    mu_assert("Basic QP test update: Error in updating lower bound!",
              vec_norm2_diff(work->data->l, sols_data->l_new, data->m) < TESTS_TOL);

    // Update upper bound
    mu_assert("Basic QP test update: Error in upper bound update: ordering!",
              osqp_update_upper_bound(work, sols_data->u_new) == 0);

    mu_assert("Basic QP test update: Error in updating upper bound!",
            vec_norm2_diff(work->data->u, sols_data->u_new, data->m) < TESTS_TOL);


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
    settings->polish = 1;
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
    mu_assert("Basic QP test solve: Error in number of iterations taken!",
              work->info->iter == work->settings->max_iter );

    // Compare solver statuses
    mu_assert("Basic QP test solve: Error in solver status!",
              work->info->status_val == sols_data->status_test );

    // Compare primal solutions
    mu_assert("Basic QP test solve: Error in primal solution!",
              vec_norm2_diff(work->solution->x, sols_data->x_test, data->n) < TESTS_TOL);

    // Compare dual solutions
    mu_assert("Basic QP test solve: Error in dual solution!",
              vec_norm2_diff(work->solution->y, sols_data->y_test, data->m) < TESTS_TOL);

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



static char * test_basic_qp()
{

    mu_run_test(test_basic_qp_solve);
    mu_run_test(test_basic_qp_update);
    mu_run_test(test_basic_qp_early_terminate);

    return 0;
}
