#include "osqp.h"     // OSQP API
#include "minunit.h"  // Basic testing script header


#include "basic_qp2/data.h"


static char * test_basic_qp2_solve()
{
    /* local variables */
    c_int exitflag = 0;  // No errors

    // Problem settings
    OSQPSettings * settings = (OSQPSettings *)c_malloc(sizeof(OSQPSettings));

    // Structures
    OSQPWorkspace * work;  // Workspace
    OSQPData * data;  // Data
    basic_qp2_sols_data *  sols_data;


    // Populate data
    data = generate_problem_basic_qp2();
    sols_data = generate_problem_basic_qp2_sols_data();


    // Define Solver settings as default
    set_default_settings(settings);
    settings->alpha = 1.6;
    settings->auto_rho = 0;
    settings->rho = 0.1;
    settings->polish = 1;
    settings->verbose = 1;

    // Setup workspace
    work = osqp_setup(data, settings);

    // Setup correct
    mu_assert("Basic QP 2 test solve: Setup error!", work != OSQP_NULL);

    // Solve Problem first time
    osqp_solve(work);

    // Compare solver statuses
    mu_assert("Basic QP 2 test solve: Error in solver status!",
              work->info->status_val == sols_data->status_test );

    // Compare primal solutions
    mu_assert("Basic QP 2 test solve: Error in primal solution!",
              vec_norm_inf_diff(work->solution->x, sols_data->x_test, data->n)/vec_norm_inf(sols_data->x_test_new, data->n) < TESTS_TOL);


    // Compare dual solutions
    mu_assert("Basic QP 2 test solve: Error in dual solution!",
              vec_norm_inf_diff(work->solution->y, sols_data->y_test, data->m)/vec_norm_inf(sols_data->y_test_new, data->m) < TESTS_TOL);


    // Compare objective values
    mu_assert("Basic QP 2 test solve: Error in objective value!",
              c_absval(work->info->obj_val - sols_data->obj_value_test) < TESTS_TOL);


    // Clean workspace
    osqp_cleanup(work);

    // Cleanup settings and data
    c_free(settings);
    clean_problem_basic_qp2(data);
    clean_problem_basic_qp2_sols_data(sols_data);

    return 0;
}


static char * test_basic_qp2_update()
{
    /* local variables */
    c_int exitflag = 0;  // No errors

    // Problem settings
    OSQPSettings * settings = (OSQPSettings *)c_malloc(sizeof(OSQPSettings));

    // Structures
    OSQPWorkspace * work;  // Workspace
    OSQPData * data;  // Data
    basic_qp2_sols_data *  sols_data;


    // Populate data
    data = generate_problem_basic_qp2();
    sols_data = generate_problem_basic_qp2_sols_data();


    // Define Solver settings as default
    set_default_settings(settings);
    settings->alpha = 1.6;
    // settings->eps_abs = 1e-08;
    // settings->eps_rel = 1e-08;
    settings->warm_start = 1;
    settings->auto_rho = 0;
    settings->polish = 1;
    settings->verbose = 1;

    // Setup workspace
    work = osqp_setup(data, settings);

    // Setup correct
    mu_assert("Basic QP 2 test update: Setup error!", work != OSQP_NULL);


    // Modify linear cost and upper bound
    osqp_update_lin_cost(work, sols_data->q_new);
    osqp_update_upper_bound(work, sols_data->u_new);

    // Solve Problem second time(with different data now)
    osqp_solve(work);

    // Compare solver statuses
    mu_assert("Basic QP 2 test update: Error in solver status!",
              work->info->status_val == sols_data->status_test_new );

    // Compare primal solutions
    mu_assert("Basic QP 2 test update: Error in primal solution!",
              vec_norm_inf_diff(work->solution->x, sols_data->x_test_new, data->n)/vec_norm_inf(sols_data->x_test_new, data->n) < TESTS_TOL);

    // Compare dual solutions
    print_vec(sols_data->y_test_new, data->m, "y_test");
    print_vec(work->solution->y, data->m, "y_osqp");
    c_print("Vec norm diff = %.5e\n",  vec_norm_inf_diff(work->solution->y, sols_data->y_test_new, data->m));
    c_print("Vec norm diff normalized = %.5e\n",  vec_norm_inf_diff(work->solution->y, sols_data->y_test_new, data->m)/vec_norm_inf(sols_data->y_test_new, data->m));

    mu_assert("Basic QP 2 test update: Error in dual solution!",
              vec_norm_inf_diff(work->solution->y, sols_data->y_test_new, data->m)/vec_norm_inf(sols_data->y_test_new, data->m) < TESTS_TOL);


    // Compare objective values
    mu_assert("Basic QP 2 test update: Error in objective value!",
              c_absval(work->info->obj_val - sols_data->obj_value_test_new) < TESTS_TOL);

    // Clean workspace
    osqp_cleanup(work);

    // Cleanup settings and data
    c_free(settings);
    clean_problem_basic_qp2(data);
    clean_problem_basic_qp2_sols_data(sols_data);

    return 0;
}



static char * test_basic_qp2()
{

    mu_run_test(test_basic_qp2_solve);
    mu_run_test(test_basic_qp2_update);

    return 0;
}
