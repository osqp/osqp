#include "osqp.h"    /* OSQP API */
#include "minunit.h" /* Basic testing script header */

static const char* test_demo_solve()
{
  /* Load problem data */
  c_float P_x[3] = { 4.0, 1.0, 2.0, };
  c_int   P_nnz  = 3;
  c_int   P_i[3] = { 0, 0, 1, };
  c_int   P_p[3] = { 0, 1, 3, };
  c_float q[2]   = { 1.0, 1.0, };
  c_float A_x[4] = { 1.0, 1.0, 1.0, 1.0, };
  c_int   A_nnz  = 4;
  c_int   A_i[4] = { 0, 1, 0, 2, };
  c_int   A_p[3] = { 0, 2, 4, };
  c_float l[3]   = { 1.0, 0.0, 0.0, };
  c_float u[3]   = { 1.0, 0.7, 0.7, };
  c_int n = 2;
  c_int m = 3;

    /* Exitflag */
  c_int exitflag;

  /* Workspace, settings, matrices */
  OSQPWorkspace *work;
  OSQPSettings *settings;
  csc *P, *A;

  /* Populate matrices */
  P = csc_matrix(n, n, P_nnz, P_x, P_i, P_p);
  A = csc_matrix(m, n, A_nnz, A_x, A_i, A_p);

  /* Set default settings */
  settings = (OSQPSettings *)c_malloc(sizeof(OSQPSettings));
  if (settings) osqp_set_default_settings(settings);

  /* Setup workspace */
  exitflag = osqp_setup(&work, P, q, A, l, u, m, n, settings);

  /* Setup correct */
  mu_assert("Demo test solve: Setup error!", exitflag == 0);

  /* Solve Problem */
  osqp_solve(work);

  /* Compare solver statuses */
  mu_assert("Demo test solve: Error in solver status!",
	          work->info->status_val == OSQP_SOLVED);

  /* Clean workspace */
  osqp_cleanup(work);
  c_free(A);
  c_free(P);
  c_free(settings);

  return 0;
}


static const char* test_demo()
{
  mu_run_test(test_demo_solve);

  return 0;
}
