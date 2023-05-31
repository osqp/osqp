#include <catch2/catch.hpp>

#include "osqp_api.h"    /* OSQP API wrapper (public + some private) */
#include "osqp_tester.h" /* Tester helpers */
#include "test_utils.h"  /* Testing Helper functions */

#include "basic_qp2_data.h"


TEST_CASE_METHOD(basic_qp2_test_fixture, "Basic QP2 solve", "[solve],[qp]")
{
  OSQPInt exitflag;

  // Need slightly tighter tolerances on this problem to pass the tests
  settings->eps_abs = 1e-6;
  settings->eps_rel = 1e-6;

  /* Test with and without polishing */
  OSQPInt polish;
  OSQPInt expectedPolishStatus;

  std::tie( polish, expectedPolishStatus ) =
      GENERATE( table<OSQPInt, OSQPInt>(
          { /* first is polish enabled, second is expected status */
            std::make_tuple( 0, OSQP_POLISH_NOT_PERFORMED ),
            std::make_tuple( 1, OSQP_POLISH_SUCCESS ) } ) );

  settings->polishing = polish;
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

  // Check polishing status
  mu_assert("Basic QP 2 test solve: Error in polish status!",
            solver->info->status_polish == expectedPolishStatus);
}

TEST_CASE_METHOD(basic_qp2_test_fixture, "Basic QP2: Update vectors", "[solve],[qp],[update]")
{
  OSQPInt exitflag;

  // Define Solver settings as default
  osqp_set_default_settings(settings.get());
  settings->warm_starting = 1;
  settings->polishing     = 1;

  /* Test all possible linear system solvers in this test case */
  settings->linsys_solver = GENERATE(filter(&isLinsysSupported, values({OSQP_DIRECT_SOLVER, OSQP_INDIRECT_SOLVER})));

  CAPTURE(settings->linsys_solver);

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

  // Check polishing status
  mu_assert("Basic QP 2 test solve: Error in polish status!",
            solver->info->status_polish == OSQP_POLISH_SUCCESS);
}
