#include "osqp.h"
#include "util.h"
#include "osqp_tester.h"

#include "primal_dual_infeasibility/data.h"


void test_optimal()
{
  OSQPInt exitflag;

  // Structures
  OSQPSolver*   solver;   // Solver
  OSQPTestData* problem;  // Problem data
  OSQPSettings* settings; // Settings

  primal_dual_infeasibility_sols_data* data;

  // Load problem data
  data = generate_problem_primal_dual_infeasibility_sols_data();

  // Populate problem data
  problem    = (OSQPTestData*) c_malloc(sizeof(OSQPTestData));
  problem->P = data->P;
  problem->q = data->q;
  problem->A = data->A12;
  problem->l = data->l;
  problem->u = data->u1;
  problem->n = data->P->n;
  problem->m = data->A12->m;

  // Define Solver settings as default
  settings = (OSQPSettings *)c_malloc(sizeof(OSQPSettings));
  osqp_set_default_settings(settings);
  settings->max_iter  = 2000;
  settings->alpha     = 1.6;
  settings->polishing = 1;
  settings->scaling   = 0;
  settings->verbose   = 1;

  // Setup workspace
  exitflag = osqp_setup(&solver,problem->P,problem->q,
                        problem->A,problem->l,problem->u,
                        problem->m,problem->n, settings);

  // Setup correct
  mu_assert("Primal dual infeasibility test 1: Setup error!", exitflag == 0);

  // Solve Problem
  osqp_solve(solver);

  // Compare solver statuses
  mu_assert("Primal dual infeasibility test 1: Error in solver status!",
            solver->info->status_val == OSQP_SOLVED);

  // Compare primal solutions
  mu_assert("Primal dual infeasibility test 1: Error in primal solution!",
            vec_norm_inf_diff(solver->solution->x, data->x1,
                              problem->n) < TESTS_TOL);

  // Compare dual solutions
  mu_assert("Primal dual infeasibility test 1: Error in dual solution!",
            vec_norm_inf_diff(solver->solution->y, data->y1,
                              problem->m) < TESTS_TOL);


  // Compare objective values
  mu_assert("Primal dual infeasibility test 1: Error in objective value!",
            c_absval(solver->info->obj_val - data->obj_value1) < TESTS_TOL);


  // Cleanup
  osqp_cleanup(solver);
  clean_problem_primal_dual_infeasibility_sols_data(data);
  c_free(problem);
  c_free(settings);
}

void test_prim_infeas()
{
  OSQPInt exitflag;

  // Structures
  OSQPSolver *solver;    // Workspace
  OSQPTestData *problem;      // Problem data
  OSQPSettings *settings; // Settings
  primal_dual_infeasibility_sols_data *data;

  // Load problem data
  data = generate_problem_primal_dual_infeasibility_sols_data();

  // Populate problem data
  problem    = (OSQPTestData*) c_malloc(sizeof(OSQPTestData));
  problem->P = data->P;
  problem->q = data->q;
  problem->A = data->A12;
  problem->l = data->l;
  problem->u = data->u2;
  problem->n = data->P->n;
  problem->m = data->A12->m;

  // Define Solver settings as default
  settings = (OSQPSettings *)c_malloc(sizeof(OSQPSettings));
  osqp_set_default_settings(settings);
  settings->max_iter  = 2000;
  settings->alpha     = 1.6;
  settings->polishing = 0;
  settings->scaling   = 0;
  settings->verbose   = 1;

  // Setup workspace
  exitflag = osqp_setup(&solver,problem->P,problem->q,
                        problem->A,problem->l,problem->u,
                        problem->m,problem->n, settings);

  // Setup correct
  mu_assert("Primal dual infeasibility test 2: Setup error!", exitflag == 0);

  // Solve Problem
  osqp_solve(solver);

  // Compare solver statuses
  mu_assert("Primal dual infeasibility test 2: Error in solver status!",
            solver->info->status_val == OSQP_PRIMAL_INFEASIBLE);

  // Cleanup
  osqp_cleanup(solver);
  clean_problem_primal_dual_infeasibility_sols_data(data);
  c_free(problem);
  c_free(settings);
}

void test_dual_infeas()
{
  OSQPInt exitflag;

  // Structures
  OSQPSolver *solver;    // Solver
  OSQPTestData *problem;      // Problem data
  OSQPSettings *settings; // Settings
  primal_dual_infeasibility_sols_data *data;

  // Load problem data
  data = generate_problem_primal_dual_infeasibility_sols_data();

  // Populate problem data
  problem    = (OSQPTestData*) c_malloc(sizeof(OSQPTestData));
  problem->P = data->P;
  problem->q = data->q;
  problem->A = data->A34;
  problem->l = data->l;
  problem->u = data->u3;
  problem->n = data->P->n;
  problem->m = data->A34->m;

  // Define Solver settings as default
  settings = (OSQPSettings *)c_malloc(sizeof(OSQPSettings));
  osqp_set_default_settings(settings);
  settings->max_iter  = 2000;
  settings->alpha     = 1.6;
  settings->polishing = 0;
  settings->scaling   = 0;
  settings->verbose   = 1;

  // Setup solver
  exitflag = osqp_setup(&solver,problem->P,problem->q,
                        problem->A,problem->l,problem->u,
                        problem->m,problem->n, settings);

  // Setup correct
  mu_assert("Primal dual infeasibility test 3: Setup error!", exitflag == 0);

  // Solve Problem
  osqp_solve(solver);

  // Compare solver statuses
  mu_assert("Primal dual infeasibility test 3: Error in solver status!",
            solver->info->status_val == OSQP_DUAL_INFEASIBLE);

  // Cleanup
  osqp_cleanup(solver);
  clean_problem_primal_dual_infeasibility_sols_data(data);
  c_free(problem);
  c_free(settings);
}

void test_primal_dual_infeas()
{
  OSQPInt exitflag;

  // Structures
  OSQPSolver *solver;    // Solver
  OSQPTestData *problem;      // Problem data
  OSQPSettings *settings; // Settings
  primal_dual_infeasibility_sols_data *data;

  // Load problem data
  data = generate_problem_primal_dual_infeasibility_sols_data();

  // Populate problem data
  problem    = (OSQPTestData*) c_malloc(sizeof(OSQPTestData));
  problem->P = data->P;
  problem->q = data->q;
  problem->A = data->A34;
  problem->l = data->l;
  problem->u = data->u4;
  problem->n = data->P->n;
  problem->m = data->A34->m;

  // Define Solver settings as default
  settings = (OSQPSettings *)c_malloc(sizeof(OSQPSettings));
  osqp_set_default_settings(settings);
  settings->max_iter  = 2000;
  settings->alpha     = 1.6;
  settings->polishing = 0;
  settings->scaling   = 0;
  settings->verbose   = 1;

  // Setup Solver
  exitflag = osqp_setup(&solver,problem->P,problem->q,
                        problem->A,problem->l,problem->u,
                        problem->m,problem->n, settings);

  // Setup correct
  mu_assert("Primal dual infeasibility test 4: Setup error!", exitflag == 0);

  // Solve Problem
  osqp_solve(solver);

  // Compare solver statuses
  mu_assert("Primal dual infeasibility test 4: Error in solver status!",
            ((solver->info->status_val == OSQP_PRIMAL_INFEASIBLE) ||
             (solver->info->status_val == OSQP_DUAL_INFEASIBLE)));

  // Cleanup
  osqp_cleanup(solver);
  clean_problem_primal_dual_infeasibility_sols_data(data);
  c_free(problem);
  c_free(settings);
}
