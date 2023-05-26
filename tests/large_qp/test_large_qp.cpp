#include <catch2/catch.hpp>

#include "osqp_api.h"    /* OSQP API wrapper (public + some private) */
#include "osqp_tester.h" /* Tester helpers */
#include "test_utils.h"  /* Testing Helper functions */

#include "large_qp_data.h"


TEST_CASE_METHOD(OSQPTestFixture, "Large QP solve", "[solve],[qp]")
{
  OSQPInt exitflag;

  /* Test all possible linear system solvers in this test case */
  settings->linsys_solver = GENERATE(filter(&isLinsysSupported, values({OSQP_DIRECT_SOLVER, OSQP_INDIRECT_SOLVER})));

  CAPTURE(settings->linsys_solver, settings->polishing);

  // Setup workspace
  exitflag = osqp_setup(&tmpSolver, &prob1_data_P_csc, prob1_data_q_val,
                        &prob1_data_A_csc, prob1_data_l_val, prob1_data_u_val,
                        prob1_data_m, prob1_data_n, settings.get());
  solver.reset(tmpSolver);

  // Setup correct
  mu_assert("Large QP test solve: Setup error!", exitflag == 0);

  // Solve Problem first time
  osqp_solve(solver.get());

  // Compare solver statuses
  mu_assert("Large QP test solve: Error in solver status!",
            solver->info->status_val == OSQP_SOLVED);

  // Compare objective values
  mu_assert("Large QP test solve: Error in objective value!",
            c_absval(solver->info->obj_val - prob1_obj_val)/(c_absval(prob1_obj_val)) < TESTS_TOL);
}
