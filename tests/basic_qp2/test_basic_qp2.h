#include "osqp.h"        // OSQP API
#include "osqp_tester.h" // Basic testing script header


#include "basic_qp2/data.h"


void test_basic_qp2_solve()
{
  c_int exitflag;

  // Problem settings
  OSQPSettings *settings = (OSQPSettings *)c_malloc(sizeof(OSQPSettings));

  // Structures
  OSQPWorkspace *work; // Workspace
  OSQPData *data;      // Data
  basic_qp2_sols_data *sols_data;


  // Populate data
  data = generate_problem_basic_qp2();
  sols_data = generate_problem_basic_qp2_sols_data();


  // Define Solver settings as default
  osqp_set_default_settings(settings);
  settings->alpha   = 1.6;
  settings->rho     = 0.1;
  settings->polish  = 1;
  settings->verbose = 1;

  // Setup workspace
  exitflag = osqp_setup(&work, data, settings);

  // Setup correct
  mu_assert("Basic QP 2 test solve: Setup error!", exitflag == 0);

  // Solve Problem first time
  osqp_solve(work);

  // Compare solver statuses
  mu_assert("Basic QP 2 test solve: Error in solver status!",
            work->info->status_val == sols_data->status_test);

  // Compare primal solutions
  mu_assert("Basic QP 2 test solve: Error in primal solution!",
            vec_norm_inf_diff(work->solution->x, sols_data->x_test,
                              data->n) /
            vec_norm_inf(sols_data->x_test_new, data->n) < TESTS_TOL);


  // Compare dual solutions
  mu_assert("Basic QP 2 test solve: Error in dual solution!",
            vec_norm_inf_diff(work->solution->y, sols_data->y_test,
                              data->m) /
            vec_norm_inf(sols_data->y_test_new, data->m) < TESTS_TOL);

  // Compare objective values
  mu_assert("Basic QP 2 test solve: Error in objective value!",
            c_absval(work->info->obj_val - sols_data->obj_value_test)/(c_absval(sols_data->obj_value_test)) < TESTS_TOL);

  // Clean workspace
  osqp_cleanup(work);

  // Cleanup settings and data
  c_free(settings);
  clean_problem_basic_qp2(data);
  clean_problem_basic_qp2_sols_data(sols_data);
}

#ifdef ENABLE_MKL_PARDISO
void test_basic_qp2_solve_pardiso()
{
  c_int exitflag;

  // Problem settings
  OSQPSettings *settings = (OSQPSettings *)c_malloc(sizeof(OSQPSettings));

  // Structures
  OSQPWorkspace *work; // Workspace
  OSQPData *data;      // Data
  basic_qp2_sols_data *sols_data;


  // Populate data
  data = generate_problem_basic_qp2();
  sols_data = generate_problem_basic_qp2_sols_data();


  // Define Solver settings as default
  osqp_set_default_settings(settings);
  settings->alpha         = 1.6;
  settings->rho           = 0.1;
  settings->polish        = 1;
  settings->verbose       = 1;
  settings->linsys_solver = MKL_PARDISO_SOLVER;

  // Setup workspace
  exitflag = osqp_setup(&work, data, settings);

  // Setup correct
  mu_assert("Basic QP 2 test solve: Setup error!", exitflag == 0);

  // Solve Problem first time
  osqp_solve(work);

  // Compare solver statuses
  mu_assert("Basic QP 2 test solve: Error in solver status!",
            work->info->status_val == sols_data->status_test);

  // Compare primal solutions
  mu_assert("Basic QP 2 test solve: Error in primal solution!",
            vec_norm_inf_diff(work->solution->x, sols_data->x_test,
                              data->n) /
            vec_norm_inf(sols_data->x_test_new, data->n) < TESTS_TOL);


  // Compare dual solutions
  mu_assert("Basic QP 2 test solve: Error in dual solution!",
            vec_norm_inf_diff(work->solution->y, sols_data->y_test,
                              data->m) /
            vec_norm_inf(sols_data->y_test_new, data->m) < TESTS_TOL);


  // Compare objective values
  mu_assert("Basic QP 2 test solve: Error in objective value!",
            c_absval(work->info->obj_val - sols_data->obj_value_test) <
            TESTS_TOL);


  // Clean workspace
  osqp_cleanup(work);

  // Cleanup settings and data
  c_free(settings);
  clean_problem_basic_qp2(data);
  clean_problem_basic_qp2_sols_data(sols_data);
}
#endif

void test_basic_qp2_update()
{
  c_int exitflag;

  // Problem settings
  OSQPSettings *settings = (OSQPSettings *)c_malloc(sizeof(OSQPSettings));

  // Structures
  OSQPWorkspace *work; // Workspace
  OSQPData *data;      // Data
  basic_qp2_sols_data *sols_data;


  // Populate data
  data = generate_problem_basic_qp2();
  sols_data = generate_problem_basic_qp2_sols_data();


  // Define Solver settings as default
  osqp_set_default_settings(settings);
  settings->alpha = 1.6;

  settings->warm_start = 1;
  settings->polish     = 1;
  settings->verbose    = 1;

  // Setup workspace
  exitflag = osqp_setup(&work, data, settings);

  // Setup correct
  mu_assert("Basic QP 2 test update: Setup error!", exitflag == 0);


  // Modify linear cost and upper bound
  osqp_update_lin_cost(work, sols_data->q_new);
  osqp_update_upper_bound(work, sols_data->u_new);

  // Solve Problem second time(with different data now)
  osqp_solve(work);

  // Compare solver statuses
  mu_assert("Basic QP 2 test update: Error in solver status!",
            work->info->status_val == sols_data->status_test_new);

  // Compare primal solutions
  mu_assert("Basic QP 2 test update: Error in primal solution!",
            vec_norm_inf_diff(work->solution->x, sols_data->x_test_new,
                              data->n) /
            vec_norm_inf(sols_data->x_test_new, data->n) < TESTS_TOL);

  // Compare dual solutions
  mu_assert("Basic QP 2 test update: Error in dual solution!",
            vec_norm_inf_diff(work->solution->y, sols_data->y_test_new,
                              data->m) /
            vec_norm_inf(sols_data->y_test_new, data->m) < TESTS_TOL);


  // Compare objective values
  mu_assert("Basic QP 2 test update: Error in objective value!",
            c_absval(
              work->info->obj_val - sols_data->obj_value_test_new)/(c_absval(sols_data->obj_value_test_new)) < TESTS_TOL);

  // Clean workspace
  osqp_cleanup(work);

  // Cleanup settings and data
  c_free(settings);
  clean_problem_basic_qp2(data);
  clean_problem_basic_qp2_sols_data(sols_data);
}
