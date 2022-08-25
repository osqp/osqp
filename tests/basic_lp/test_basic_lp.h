#include "osqp.h"    // OSQP API
#include "util.h"    // Utilities for testing
#include "osqp_tester.h" // Basic testing script header

#include "basic_lp/data.h"


void test_basic_lp_solve()
{
  c_int exitflag;

  // Problem settings
  OSQPSettings_ptr settings{(OSQPSettings *)c_malloc(sizeof(OSQPSettings))};

  // Structures
  OSQPSolver     *tmpSolver = nullptr;
  OSQPSolver_ptr solver{nullptr};   // Wrap solver inside memory management

  // Populate data
  basic_lp_problem_ptr data{generate_problem_basic_lp()};
  basic_lp_sols_data_ptr sols_data{generate_problem_basic_lp_sols_data()};

  // Define Solver settings as default
  osqp_set_default_settings(settings.get());
  settings->max_iter      = 2000;
  settings->polishing     = 1;
  settings->scaling       = 1;
  settings->verbose       = 1;

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

#ifdef OSQP_ALGEBRA_MKL
void test_basic_lp_solve_pardiso()
{
  c_int exitflag;

  // Problem settings
  OSQPSettings_ptr settings{(OSQPSettings *)c_malloc(sizeof(OSQPSettings))};

  // Structures
  OSQPSolver     *tmpSolver = nullptr;
  OSQPSolver_ptr solver{nullptr};   // Wrap solver inside memory management

  // Populate data
  basic_lp_problem_ptr data{generate_problem_basic_lp()};
  basic_lp_sols_data_ptr sols_data{generate_problem_basic_lp_sols_data()};

  // Define Solver settings as default
  osqp_set_default_settings(settings.get());
  settings->max_iter      = 2000;
  settings->polishing     = 1;
  settings->verbose       = 1;
  settings->linsys_solver = OSQP_DIRECT_SOLVER;

  // Setup solver
  exitflag = osqp_setup(&tmpSolver, data->P, data->q,
                        data->A, data->l, data->u,
                        data->m, data->n, settings.get());
  solver.reset(tmpSolver);

  // Setup correct
  mu_assert("Basic LP test solve Pardiso: Setup error!", exitflag == 0);

  // Solve Problem
  osqp_solve(solver.get());

  // Compare solver statuses
  mu_assert("Basic LP test solve Pardiso: Error in solver status!",
            solver->info->status_val == sols_data->status_test);

  // Compare primal solutions
  mu_assert("Basic LP test solve Pardiso: Error in primal solution!",
            vec_norm_inf_diff(solver->solution->x, sols_data->x_test,
                              data->n) < TESTS_TOL);

  // Compare dual solutions
  mu_assert("Basic LP test solve Pardiso: Error in dual solution!",
            vec_norm_inf_diff(solver->solution->y, sols_data->y_test,
                              data->m) < TESTS_TOL);

  // Compare objective values
  mu_assert("Basic LP test solve Pardiso: Error in objective value!",
            c_absval(solver->info->obj_val - sols_data->obj_value_test) <
            TESTS_TOL);
}
#endif
