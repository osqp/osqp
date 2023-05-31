#include <catch2/catch.hpp>

#include "osqp_api.h"    /* OSQP API wrapper (public + some private) */
#include "osqp_tester.h" /* Tester helpers */
#include "test_utils.h"  /* Testing Helper functions */

TEST_CASE( "Demo solve", "[optimization],[QP]" ) {
  /* Load problem data */
  OSQPFloat P_x[3] = { 4.0, 1.0, 2.0, };
  OSQPInt   P_nnz  = 3;
  OSQPInt   P_i[3] = { 0, 0, 1, };
  OSQPInt   P_p[3] = { 0, 1, 3, };
  OSQPFloat q[2]   = { 1.0, 1.0, };
  OSQPFloat A_x[4] = { 1.0, 1.0, 1.0, 1.0, };
  OSQPInt   A_nnz  = 4;
  OSQPInt   A_i[4] = { 0, 1, 0, 2, };
  OSQPInt   A_p[3] = { 0, 2, 4, };
  OSQPFloat l[3]   = { 1.0, 0.0, 0.0, };
  OSQPFloat u[3]   = { 1.0, 0.7, 0.7, };
  OSQPInt   n      = 2;
  OSQPInt   m      = 3;

  /* Exitflag */
  OSQPInt exitflag;

  /* Workspace, settings, matrices */
  OSQPSolver*      tmpSolver = nullptr;
  OSQPSolver_ptr   solver{nullptr};
  OSQPSettings_ptr settings{(OSQPSettings *)malloc(sizeof(OSQPSettings))};

  OSQPCscMatrix_ptr P{(OSQPCscMatrix*)malloc(sizeof(OSQPCscMatrix))};
  OSQPCscMatrix_ptr A{(OSQPCscMatrix*)malloc(sizeof(OSQPCscMatrix))};

  /* Populate matrices */
  csc_set_data(P.get(), n, n, P_nnz, P_x, P_i, P_p);
  csc_set_data(A.get(), m, n, A_nnz, A_x, A_i, A_p);

  /* Set default settings */
  if (settings.get())
    osqp_set_default_settings(settings.get());

  /* Setup workspace */
  exitflag = osqp_setup(&tmpSolver, P.get(), q, A.get(), l, u, m, n, settings.get());
  solver.reset(tmpSolver);

  /* Setup correct */
  mu_assert("Demo test solve: Setup error!", exitflag == 0);

  /* Solve Problem */
  osqp_solve(solver.get());

  /* Compare solver statuses */
  mu_assert("Demo test solve: Error in solver status!",
	          solver->info->status_val == OSQP_SOLVED);
}
