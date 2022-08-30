#include "osqp.h"    // OSQP API
#include "osqp_tester.h" // Basic testing script header


#include "basic_qp2/data.h"


void test_basic_qp2_solve()
{
  OSQPInt exitflag;

  // Problem settings
  OSQPSettings_ptr settings{(OSQPSettings *)c_malloc(sizeof(OSQPSettings))};

  // Structures
  OSQPSolver*    tmpSolver = nullptr;
  OSQPSolver_ptr solver{nullptr};   // Wrap solver inside memory management

  // Populate data
  basic_qp2_problem_ptr data{generate_problem_basic_qp2()};
  basic_qp2_sols_data_ptr sols_data{generate_problem_basic_qp2_sols_data()};

  // Define Solver settings as default
  osqp_set_default_settings(settings.get());
  settings->alpha   = 1.6;
  settings->rho     = 0.1;
  settings->verbose = 1;
  settings->eps_abs = 1e-5;
  settings->eps_rel = 1e-5;

  /* Test with and without polishing */
  settings->polishing = GENERATE(0, 1);
  settings->polish_refine_iter = 4;

  /* Test all possible linear system solvers in this test case */
  settings->linsys_solver = GENERATE(filter(&isLinsysSupported, values({OSQP_DIRECT_SOLVER, OSQP_INDIRECT_SOLVER})));

  CAPTURE(settings->linsys_solver, settings->polishing);

  // Setup workspace
  exitflag = osqp_setup(&tmpSolver, data->P, data->q,
                        data->A, data->l, data->u,
                        data->m, data->n, settings.get());
  solver.reset(tmpSolver);

  // Setup correct
  mu_assert("Basic QP 2 test solve: Setup error!", exitflag == 0);

  // Solve Problem first time
  osqp_solve(solver.get());

  // Compare solver statuses
  mu_assert("Basic QP 2 test solve: Error in solver status!",
            solver->info->status_val == sols_data->status_test);

  // Compare primal solutions
  mu_assert("Basic QP 2 test solve: Error in primal solution!",
            vec_norm_inf_diff(solver->solution->x, sols_data->x_test,
                              data->n) /
            vec_norm_inf(sols_data->x_test_new, data->n) < TESTS_TOL);


  // Compare dual solutions
  mu_assert("Basic QP 2 test solve: Error in dual solution!",
            vec_norm_inf_diff(solver->solution->y, sols_data->y_test,
                              data->m) /
            vec_norm_inf(sols_data->y_test_new, data->m) < TESTS_TOL);

  // Compare objective values
  mu_assert("Basic QP 2 test solve: Error in objective value!",
            c_absval(solver->info->obj_val - sols_data->obj_value_test)/(c_absval(sols_data->obj_value_test)) < TESTS_TOL);
}

void test_basic_qp2_update()
{
  OSQPInt exitflag;

  // Problem settings
  OSQPSettings_ptr settings{(OSQPSettings *)c_malloc(sizeof(OSQPSettings))};

  // Structures
  OSQPSolver*    tmpSolver = nullptr;
  OSQPSolver_ptr solver{nullptr};   // Wrap solver inside memory management

  // Populate data
  basic_qp2_problem_ptr data{generate_problem_basic_qp2()};
  basic_qp2_sols_data_ptr sols_data{generate_problem_basic_qp2_sols_data()};

  // Define Solver settings as default
  osqp_set_default_settings(settings.get());
  settings->alpha         = 1.6;
  settings->warm_starting = 1;
  settings->polishing     = 1;
  settings->verbose       = 1;

  /* Test all possible linear system solvers in this test case */
  settings->linsys_solver = GENERATE(filter(&isLinsysSupported, values({OSQP_DIRECT_SOLVER, OSQP_INDIRECT_SOLVER})));

  // Setup workspace
  exitflag = osqp_setup(&tmpSolver, data->P, data->q,
                        data->A, data->l, data->u,
                        data->m, data->n, settings.get());
  solver.reset(tmpSolver);

  // Setup correct
  mu_assert("Basic QP 2 test update: Setup error!", exitflag == 0);


  // Modify linear cost and upper bound
  osqp_update_data_vec(solver.get(), sols_data->q_new, NULL, NULL);
  osqp_update_data_vec(solver.get(), NULL, NULL, sols_data->u_new);

  // Solve Problem second time(with different data now)
  osqp_solve(solver.get());

  // Compare solver statuses
  mu_assert("Basic QP 2 test update: Error in solver status!",
            solver->info->status_val == sols_data->status_test_new);

  // Compare primal solutions
  mu_assert("Basic QP 2 test update: Error in primal solution!",
            vec_norm_inf_diff(solver->solution->x, sols_data->x_test_new,
                              data->n) /
            vec_norm_inf(sols_data->x_test_new, data->n) < TESTS_TOL);

  // Compare dual solutions
  mu_assert("Basic QP 2 test update: Error in dual solution!",
            vec_norm_inf_diff(solver->solution->y, sols_data->y_test_new,
                              data->m) /
            vec_norm_inf(sols_data->y_test_new, data->m) < TESTS_TOL);


  // Compare objective values
  mu_assert("Basic QP 2 test update: Error in objective value!",
            c_absval(
              solver->info->obj_val - sols_data->obj_value_test_new)/(c_absval(sols_data->obj_value_test_new)) < TESTS_TOL);
}
