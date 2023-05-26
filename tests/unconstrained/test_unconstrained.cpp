#include <catch2/catch.hpp>

#include "osqp_api.h"    /* OSQP API wrapper (public + some private) */
#include "osqp_tester.h" /* Tester helpers */
#include "test_utils.h"  /* Testing Helper functions */

#include "unconstrained_data.h"


TEST_CASE_METHOD(unconstrained_test_fixture, "Unconstrained solve", "[unconstrained],[solve]")
{
  OSQPInt exitflag;

  /* Test with and without polishing */
  OSQPInt polish;
  OSQPInt expectedPolishStatus;

  std::tie( polish, expectedPolishStatus ) =
      GENERATE( table<OSQPInt, OSQPInt>(
          { /* first is polish enabled, second is expected status */
            std::make_tuple( 0, OSQP_POLISH_NOT_PERFORMED ),
            std::make_tuple( 1, OSQP_POLISH_NO_ACTIVE_SET_FOUND ) } ) );

  settings->polishing = polish;
  settings->polish_refine_iter = 4;

  /* Test all possible linear system solvers in this test case */
  settings->linsys_solver = GENERATE(filter(&isLinsysSupported, values({OSQP_DIRECT_SOLVER, OSQP_INDIRECT_SOLVER})));

  CAPTURE(settings->linsys_solver, settings->polishing);

  // Setup solver
  exitflag = osqp_setup(&tmpSolver, data->P, data->q,
                        data->A, data->l, data->u,
                        data->m, data->n, settings.get());
  solver.reset(tmpSolver);

  // Setup correct
  mu_assert("Unconstrained test solve: Setup error!", exitflag == 0);

  // Solve Problem first time
  osqp_solve(solver.get());

  // Compare solver statuses
  mu_assert("Unconstrained test solve: Error in solver status!",
            solver->info->status_val == sols_data->status_test);

  // Compare primal solutions
  mu_assert("Unconstrained test solve: Error in primal solution!",
            vec_norm_inf_diff(solver->solution->x, sols_data->x_test,
                              data->n) < TESTS_TOL);

  // Compare objective values
  mu_assert("Unconstrained test solve: Error in objective value!",
            c_absval(solver->info->obj_val - sols_data->obj_value_test) <
            TESTS_TOL);

  // Check polishing status
  mu_assert("Unconstrained test solve: Error in polish status!",
            solver->info->status_polish == expectedPolishStatus);
}
