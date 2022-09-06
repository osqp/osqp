#include <catch2/catch.hpp>
#include <time.h>

#include "osqp_api.h"    /* OSQP API wrapper (public + some private) */
#include "osqp_tester.h" /* Tester helpers */
#include "test_utils.h"  /* Testing Helper functions */

#include "primal_dual_infeasibility_data.h"


TEST_CASE("Feasible problem", "[solve]")
{
  OSQPInt exitflag;

  // Structures
  OSQPSolver*      tmpSolver;   // Solver
  OSQPSolver_ptr   solver{};
  OSQPSettings_ptr settings{(OSQPSettings *)c_malloc(sizeof(OSQPSettings))};

  primal_dual_infeasibility_sols_data_ptr data{generate_problem_primal_dual_infeasibility_sols_data()};

  // Define Solver settings as default
  osqp_set_default_settings(settings.get());
  settings->max_iter  = 2000;
  settings->alpha     = 1.6;
  settings->polishing = 1;
  settings->scaling   = 0;
  settings->verbose   = 1;

  // Setup workspace
  exitflag = osqp_setup(&tmpSolver,   data->P,    data->q,
                        data->A12,    data->l,    data->u1,
                        data->A12->m, data->P->n, settings.get());
  solver.reset(tmpSolver);

  // Setup correct
  mu_assert("Primal dual infeasibility test 1: Setup error!",
            exitflag == OSQP_NO_ERROR);

  // Solve Problem
  osqp_solve(solver.get());

  // Compare solver statuses
  mu_assert("Primal dual infeasibility test 1: Error in solver status!",
            solver->info->status_val == OSQP_SOLVED);

  // Compare primal solutions
  mu_assert("Primal dual infeasibility test 1: Error in primal solution!",
            vec_norm_inf_diff(solver->solution->x, data->x1,
                              data->P->n) < TESTS_TOL);

  // Compare dual solutions
  mu_assert("Primal dual infeasibility test 1: Error in dual solution!",
            vec_norm_inf_diff(solver->solution->y, data->y1,
                              data->A12->m) < TESTS_TOL);

  // Compare objective values
  mu_assert("Primal dual infeasibility test 1: Error in objective value!",
            c_absval(solver->info->obj_val - data->obj_value1) < TESTS_TOL);
}

TEST_CASE("Primal infeasible problem", "[solve],[infeasible]")
{
  OSQPInt exitflag;

// Structures
  OSQPSolver*      tmpSolver;   // Solver
  OSQPSolver_ptr   solver{};
  OSQPSettings_ptr settings{(OSQPSettings *)c_malloc(sizeof(OSQPSettings))};

  primal_dual_infeasibility_sols_data_ptr data{generate_problem_primal_dual_infeasibility_sols_data()};

  // Define Solver settings as default
  osqp_set_default_settings(settings.get());
  settings->max_iter  = 2000;
  settings->alpha     = 1.6;
  settings->polishing = 0;
  settings->scaling   = 0;
  settings->verbose   = 1;

  // Setup workspace
  exitflag = osqp_setup(&tmpSolver,   data->P,    data->q,
                        data->A12,    data->l,    data->u2,
                        data->A12->m, data->P->n, settings.get());
  solver.reset(tmpSolver);

  // Setup correct
  mu_assert("Primal dual infeasibility test 2: Setup error!",
            exitflag == OSQP_NO_ERROR);

  // Solve Problem
  osqp_solve(solver.get());

  // Compare solver statuses
  mu_assert("Primal dual infeasibility test 2: Error in solver status!",
            solver->info->status_val == OSQP_PRIMAL_INFEASIBLE);
}

TEST_CASE("Dual infeasible problem", "[solve],[infeasible]")
{
  OSQPInt exitflag;

  // Structures
  OSQPSolver*      tmpSolver;   // Solver
  OSQPSolver_ptr   solver{};
  OSQPSettings_ptr settings{(OSQPSettings *)c_malloc(sizeof(OSQPSettings))};

  primal_dual_infeasibility_sols_data_ptr data{generate_problem_primal_dual_infeasibility_sols_data()};

  // Define Solver settings as default
  osqp_set_default_settings(settings.get());
  settings->max_iter  = 2000;
  settings->alpha     = 1.6;
  settings->polishing = 0;
  settings->scaling   = 0;
  settings->verbose   = 1;

  // Setup solver
  exitflag = osqp_setup(&tmpSolver,   data->P,    data->q,
                        data->A34,    data->l,    data->u3,
                        data->A34->m, data->P->n, settings.get());
  solver.reset(tmpSolver);

  // Setup correct
  mu_assert("Primal dual infeasibility test 3: Setup error!",
            exitflag == OSQP_NO_ERROR);

  // Solve Problem
  osqp_solve(solver.get());

  // Compare solver statuses
  mu_assert("Primal dual infeasibility test 3: Error in solver status!",
            solver->info->status_val == OSQP_DUAL_INFEASIBLE);
}

TEST_CASE("Primal and dual infeasible problem", "[solve],[infeasible]")
{
  OSQPInt exitflag;

// Structures
  OSQPSolver*      tmpSolver;   // Solver
  OSQPSolver_ptr   solver{};
  OSQPSettings_ptr settings{(OSQPSettings *)c_malloc(sizeof(OSQPSettings))};

  primal_dual_infeasibility_sols_data_ptr data{generate_problem_primal_dual_infeasibility_sols_data()};

  // Define Solver settings as default
  osqp_set_default_settings(settings.get());
  settings->max_iter  = 2000;
  settings->alpha     = 1.6;
  settings->polishing = 0;
  settings->scaling   = 0;
  settings->verbose   = 1;

  // Setup Solver
  exitflag = osqp_setup(&tmpSolver,   data->P,    data->q,
                        data->A34,    data->l,    data->u4,
                        data->A34->m, data->P->n, settings.get());
  solver.reset(tmpSolver);

  // Setup correct
  mu_assert("Primal dual infeasibility test 4: Setup error!",
            exitflag == OSQP_NO_ERROR);

  // Solve Problem
  osqp_solve(solver.get());

  // Compare solver statuses
  mu_assert("Primal dual infeasibility test 4: Error in solver status!",
            ((solver->info->status_val == OSQP_PRIMAL_INFEASIBLE) ||
             (solver->info->status_val == OSQP_DUAL_INFEASIBLE)));
}
