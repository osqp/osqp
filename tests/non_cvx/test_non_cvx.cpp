#include <catch2/catch.hpp>

#include "osqp_api.h"    /* OSQP API wrapper (public + some private) */
#include "osqp_tester.h" /* Tester helpers */
#include "test_utils.h"  /* Testing Helper functions */

#include "non_cvx_data.h"

#ifndef OSQP_ALGEBRA_CUDA
TEST_CASE_METHOD(non_cvx_test_fixture, "Nonconvex: Setup detection", "[nonconvex],[setup]")
{
  OSQPInt exitflag;

  // Test-specific solver settings
  settings->adaptive_rho = 0;

  // Direct linear solvers detect the nonconvexity at the setup phase
  settings->linsys_solver = OSQP_DIRECT_SOLVER;

  SECTION("Nonconvex test setup: (P + sigma I) negative eigenvalue") {
    settings->sigma = 1e-6;

    // Setup workspace
    exitflag = osqp_setup(&tmpSolver, data->P, data->q,
                          data->A, data->l, data->u,
                          data->m, data->n, settings.get());
    solver.reset(tmpSolver);

    // Setup should fail due to (P + sigma I) having a negative eigenvalue
    mu_assert("Nonconvex test setup: Setup should have failed!",
              exitflag == OSQP_NONCVX_ERROR);
  }

  SECTION("Nonconvex test setup: (P + sigma I) no negative eigenvalues") {
    settings->sigma = sols_data->sigma_new;

    // Setup workspace
    exitflag = osqp_setup(&tmpSolver, data->P, data->q,
                          data->A, data->l, data->u,
                          data->m, data->n, settings.get());
    solver.reset(tmpSolver);

    // Setup should not fail due to (P + sigma I) having negative eigenvalues
    mu_assert("Nonconvex test setup: Setup should not have failed!",
              exitflag == OSQP_NO_ERROR);
  }
}
#endif

TEST_CASE_METHOD(non_cvx_test_fixture, "Nonconvex: Solve", "[nonconvex],[solve]")
{
  OSQPInt exitflag;

  // Test-specific solver settings
  settings->adaptive_rho = 0;
  settings->sigma = sols_data->sigma_new;

  // Setup workspace
  exitflag = osqp_setup(&tmpSolver, data->P, data->q,
                        data->A, data->l, data->u,
                        data->m, data->n, settings.get());
  solver.reset(tmpSolver);

  // Setup should work because (P + sigma I) is positive definite
  mu_assert("Nonconvex test solve: Setup error!",
             exitflag == OSQP_NO_ERROR);

  // Solve Problem first time
  osqp_solve(solver.get());

  // Compare solver statuses
  mu_assert("Nonconvex test solve: Error in solver status!",
            solver->info->status_val == OSQP_NON_CVX);

  // Compare objective values
  mu_assert("Nonconvex test solve: Error in objective value!",
            solver->info->obj_val == OSQP_NAN);
}
