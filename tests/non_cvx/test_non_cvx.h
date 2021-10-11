#include "osqp.h"    // OSQP API
#include "osqp_tester.h" // Basic testing script header


#include "non_cvx/data.h"


void test_non_cvx_solve()
{
  c_int exitflag;

  // Problem settings
  OSQPSettings *settings = (OSQPSettings *)c_malloc(sizeof(OSQPSettings));

  // Structures
  OSQPSolver *solver; // Workspace
  OSQPTestData *data;      // Data
  non_cvx_sols_data *sols_data;


  // Populate data
  data = generate_problem_non_cvx();
  sols_data = generate_problem_non_cvx_sols_data();


  // Define Solver settings as default
  osqp_set_default_settings(settings);
  settings->verbose = 1;
  settings->adaptive_rho = 0;
  settings->sigma = 1e-6;

#ifndef CUDA_SUPPORT 
  // Setup workspace
  exitflag = osqp_setup(&solver, data->P, data->q,
                        data->A, data->l, data->u,
                        data->m, data->n, settings);

  // Setup should fail due to (P + sigma I) having a negative eigenvalue
  mu_assert("Non Convex test solve: Setup should have failed!",
            exitflag == OSQP_NONCVX_ERROR);

  osqp_cleanup(solver);
#endif /* ifndef CUDA_SUPPORT */

  // Update Solver settings
  settings->sigma = sols_data->sigma_new;

  // Setup workspace again
  exitflag = osqp_setup(&solver, data->P, data->q,
                        data->A, data->l, data->u,
                        data->m, data->n, settings);

  // Setup should work this time because (P + sigma I) is positive definite
  mu_assert("Non Convex test solve: Setup error!", exitflag == 0);

  // Solve Problem first time
  osqp_solve(solver);

  // Compare solver statuses
  mu_assert("Non Convex test solve: Error in solver status!",
            solver->info->status_val == OSQP_NON_CVX);

  // Compare objective values
  mu_assert("Non Convex test solve: Error in objective value!",
            solver->info->obj_val == OSQP_NAN);

  // Clean solver
  osqp_cleanup(solver);

  // Cleanup settings and data
  c_free(settings);
  clean_problem_non_cvx(data);
  clean_problem_non_cvx_sols_data(sols_data);
}