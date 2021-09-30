#include "osqp.h"    // OSQP API
#include "osqp_tester.h" // Basic testing script header


#include "unconstrained/data.h"


void test_unconstrained_solve()
{
  c_int exitflag;

  // Problem settings
  OSQPSettings *settings = (OSQPSettings *)c_malloc(sizeof(OSQPSettings));

  // Structures
  OSQPSolver *solver; // Solver
  OSQPTestData *data;      // Data
  unconstrained_sols_data *sols_data;


  // Populate data
  data = generate_problem_unconstrained();
  sols_data = generate_problem_unconstrained_sols_data();


  // Define Solver settings as default
  osqp_set_default_settings(settings);
  settings->eps_abs = 1e-05;
  settings->eps_rel = 1e-05;
  settings->verbose = 1;

  // Setup solver
  exitflag = osqp_setup(&solver, data->P, data->q,
                        data->A, data->l, data->u,
                        data->m, data->n, settings);

  // Setup correct
  mu_assert("Unconstrained test solve: Setup error!", exitflag == 0);

  // Solve Problem first time
  osqp_solve(solver);

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

  // Clean solver
  osqp_cleanup(solver);

  // Cleanup settings and data
  c_free(settings);
  clean_problem_unconstrained(data);
  clean_problem_unconstrained_sols_data(sols_data);
}