#include <catch2/catch.hpp>

#include "osqp_api.h"    /* OSQP API wrapper (public + some private) */
#include "osqp_tester.h" /* Tester helpers */
#include "test_utils.h"  /* Testing Helper functions */

#include "basic_lp_data.h"


TEST_CASE( "Basic LP", "[solve][lp]" )
{
  OSQPInt exitflag;

  // Problem settings
  OSQPSettings_ptr settings{(OSQPSettings *)c_malloc(sizeof(OSQPSettings))};

  // Structures
  OSQPSolver*    tmpSolver = nullptr;
  OSQPSolver_ptr solver{nullptr};   // Wrap solver inside memory management

  // Populate data
  basic_lp_problem_ptr data{generate_problem_basic_lp()};
  basic_lp_sols_data_ptr sols_data{generate_problem_basic_lp_sols_data()};

  // Define Solver settings as default
  osqp_set_default_settings(settings.get());
  settings->max_iter = 2000;
  settings->scaling  = 1;
  settings->verbose  = 1;
  settings->eps_abs  = 1e-5;
  settings->eps_rel  = 1e-5;

  /* Test with and without polishing */
  settings->polishing = GENERATE(0, 1);
  settings->polish_refine_iter = 4;

  /* TODO: MKL CG is failing this test, so test with default linear algebra only */
#ifndef OSQP_ALGEBRA_MKL
  /* Test all possible linear system solvers in this test case */
  settings->linsys_solver = GENERATE(filter(&isLinsysSupported, values({OSQP_DIRECT_SOLVER, OSQP_INDIRECT_SOLVER})));
#endif

  CAPTURE(settings->linsys_solver, settings->polishing);

  // Setup solver
  exitflag = osqp_setup(&tmpSolver, data->P, data->q,
                        data->A, data->l, data->u,
                        data->m, data->n, settings.get());
  solver.reset(tmpSolver);

  // Setup correct
  mu_assert("Basic LP test solve: Setup error!", exitflag == 0);

  // Solve Problem
  osqp_solve(solver.get());

  // Compare solver statuses
  mu_assert("Basic LP test solve: Error in solver status!",
      solver->info->status_val == sols_data->status_test);

  // Compare objective values
  mu_assert("Basic LP test solve: Error in objective value!",
      c_absval(solver->info->obj_val - sols_data->obj_value_test) <
      TESTS_TOL);

  // Compare primal solutions
  mu_assert("Basic LP test solve: Error in primal solution!",
      vec_norm_inf_diff(solver->solution->x, sols_data->x_test,
            data->n) < TESTS_TOL);

  // Compare dual solutions
  mu_assert("Basic LP test solve: Error in dual solution!",
      vec_norm_inf_diff(solver->solution->y, sols_data->y_test,
            data->m) < TESTS_TOL);
}
