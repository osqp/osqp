#include "osqp.h"    // OSQP API
#include "util.h"    // Utilities for testing
#include "minunit.h" // Basic testing script header

#include "primal_infeasibility/data.h"


static const char* test_primal_infeasible_qp_solve()
{
  c_int exitflag;

  // Problem settings
  OSQPSettings *settings = (OSQPSettings *)c_malloc(sizeof(OSQPSettings));

  // Structures
  OSQPSolver *solver; // Workspace
  OSQPTestData *data;      // Data
  primal_infeasibility_sols_data *sols_data;

  // Populate data
  data      = generate_problem_primal_infeasibility();
  sols_data = generate_problem_primal_infeasibility_sols_data();


  // Define Solver settings as default
  osqp_set_default_settings(settings);
  settings->max_iter   = 10000;
  settings->alpha      = 1.6;
  settings->polish     = 1;
  settings->scaling    = 0;
  settings->verbose    = 1;
  settings->warm_start = 0;

  // Setup workspace
  exitflag = osqp_setup(&solver, data->P, data->q,
                        data->A, data->l, data->u,
                        data->m, data->n, settings);

  // Setup correct
  mu_assert("Primal infeasible QP test solve: Setup error!", exitflag == 0);

  // Solve Problem
  osqp_solve(solver);

  // Compare solver statuses
  mu_assert("Primal infeasible QP test solve: Error in solver status!",
            solver->info->status_val == sols_data->status_test);


  // Clean workspace
  osqp_cleanup(solver);


  // Cleanup data
  clean_problem_primal_infeasibility(data);
  clean_problem_primal_infeasibility_sols_data(sols_data);

  // Cleanup
  c_free(settings);

  return 0;
}

static const char* test_primal_infeasibility()
{
  mu_run_test(test_primal_infeasible_qp_solve);


  return 0;
}
