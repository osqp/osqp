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

#ifndef ALGEBRA_CUDA
  if (settings->linsys_solver == OSQP_DIRECT_SOLVER) {
      // Setup workspace
      exitflag = osqp_setup(&solver, data->P, data->q,
                            data->A, data->l, data->u,
                            data->m, data->n, settings);

      // Setup should fail due to (P + sigma I) having a negative eigenvalue
      mu_assert("Non Convex test solve: Setup should have failed!",
                exitflag == OSQP_NONCVX_ERROR);

      osqp_cleanup(solver);
  }
#endif

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

#ifdef OSQP_CODEGEN
void test_non_cvx_codegen()
{
  c_int exitflag;

  // Problem settings
  OSQPSettings *settings = (OSQPSettings *)c_malloc(sizeof(OSQPSettings));

  // Codegen defines
  OSQPCodegenDefines *defines = (OSQPCodegenDefines *)c_malloc(sizeof(OSQPCodegenDefines));

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

  // Define codegen settings
  defines->embedded_mode = 1;    // vector update
  defines->float_type = 1;       // floats
  defines->printing_enable = 0;  // no printing
  defines->profiling_enable = 0; // no timing
  defines->interrupt_enable = 0; // no interrupts

  // Setup workspace
  exitflag = osqp_setup(&solver, data->P, data->q,
                        data->A, data->l, data->u,
                        data->m, data->n, settings);

  // Setup should fail due to (P + sigma I) having a negative eigenvalue
  mu_assert("Non Convex codegen: Setup should have failed!",
            exitflag == OSQP_NONCVX_ERROR);

  exitflag = osqp_codegen(solver, "./", "noncvx_", defines);

  // Codegen should fail due to (P + sigma I) having a negative eigenvalue
  mu_assert("Non Convex codegen: codegen type 1 should have failed!",
            exitflag == OSQP_NONCVX_ERROR);

  defines->embedded_mode = 2;    // matrix update

  exitflag = osqp_codegen(solver, "./", "noncvx_", defines);

  // Codegen should fail due to (P + sigma I) having a negative eigenvalue
  mu_assert("Non Convex codegen: codegen type 2 should have failed!",
            exitflag == OSQP_NONCVX_ERROR);


  osqp_cleanup(solver);

  // Update Solver settings
  settings->sigma = sols_data->sigma_new;

  // Setup workspace again
  exitflag = osqp_setup(&solver, data->P, data->q,
                        data->A, data->l, data->u,
                        data->m, data->n, settings);

  // Setup should work this time because (P + sigma I) is positive definite
  mu_assert("Non Convex codegen: Setup error!", exitflag == 0);

  exitflag = osqp_codegen(solver, "./", "noncvx_", defines);

  // Codegen should work since (P + sigma I) is positive definite
  mu_assert("Non Convex codegen: codegen should have worked!",
            exitflag == 0);

  // Clean solver
  osqp_cleanup(solver);

  // Cleanup settings and data
  c_free(settings);
  c_free(defines);
  clean_problem_non_cvx(data);
  clean_problem_non_cvx_sols_data(sols_data);
}
#endif
