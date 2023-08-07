#include <catch2/catch.hpp>

#include "osqp_api.h"    /* OSQP API wrapper (public + some private) */
#include "osqp_tester.h" /* Tester helpers */
#include "test_utils.h"  /* Testing Helper functions */

#include "basic_qp_data.h"


TEST_CASE_METHOD(basic_qp_test_fixture, "Basic QP: Solve and get solution later", "[solve][qp]")
{
  OSQPInt exitflag;

  // Test-specific options
  settings->allocate_solution = 0;
  settings->polishing         = 1;
  settings->scaling           = 0;
  settings->warm_starting     = 0;

  /* Test all possible linear system solvers in this test case */
  settings->linsys_solver = GENERATE(filter(&isLinsysSupported, values({OSQP_DIRECT_SOLVER, OSQP_INDIRECT_SOLVER})));

  CAPTURE(settings->linsys_solver);

  // Setup solver
  exitflag = osqp_setup(&tmpSolver, data->P, data->q,
                        data->A, data->l, data->u,
                        data->m, data->n, settings.get());
  solver.reset(tmpSolver);

  // Setup correct
  mu_assert("Basic QP test solve: Setup error!", exitflag == 0);
  mu_assert("Basic QP test solve: Solution improperly allocated", !(solver->solution));

  // Solve Problem
  osqp_solve(solver.get());

  // Compare solver statuses
  mu_assert("Basic QP test solve: Error in solver status!",
      solver->info->status_val == sols_data->status_test);

  // Get the solution
  OSQPSolution_ptr solution{(OSQPSolution*)c_calloc(1, sizeof(OSQPSolution))};
  solution->x             = (OSQPFloat*) c_calloc(1, data->n * sizeof(OSQPFloat));
  solution->y             = (OSQPFloat*) c_calloc(1, data->m * sizeof(OSQPFloat));
  solution->prim_inf_cert = (OSQPFloat*) c_calloc(1, data->m * sizeof(OSQPFloat));
  solution->dual_inf_cert = (OSQPFloat*) c_calloc(1, data->n * sizeof(OSQPFloat));

  osqp_get_solution(solver.get(), solution.get());

  // Compare primal solutions
  mu_assert("Basic QP test solve: Error in primal solution!",
      vec_norm_inf_diff(solution->x, sols_data->x_test,
            data->n) < TESTS_TOL);

  // Compare dual solutions
  mu_assert("Basic QP test solve: Error in dual solution!",
      vec_norm_inf_diff(solution->y, sols_data->y_test,
            data->m) < TESTS_TOL);

  // Compare objective values
  mu_assert("Basic QP test solve: Error in objective value!",
      c_absval(solver->info->obj_val - sols_data->obj_value_test) <
      TESTS_TOL);
}
