#include "osqp.h"
#include <stdlib.h>
#include <stdio.h>

int main(void) {

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

  /* Solver, settings, matrices */
  OSQPSolver   *solver   = NULL;
  OSQPSettings *settings = NULL;
  csc *P = malloc(sizeof(csc));
  csc *A = malloc(sizeof(csc));

  /* Populate matrices */
  csc_set_data(A, m, n, A_nnz, A_x, A_i, A_p);
  csc_set_data(P, n, n, P_nnz, P_x, P_i, P_p);

  /* Set default settings */
  settings = (OSQPSettings *)malloc(sizeof(OSQPSettings));
  if (settings) {
    osqp_set_default_settings(settings);
    settings->polishing = 1;

    //settings->linsys_solver = OSQP_DIRECT_SOLVER;
    //settings->linsys_solver = OSQP_INDIRECT_SOLVER;
  }

  c_int cap = osqp_capabilities();

  printf("This OSQP library supports:\n");
  if(cap & OSQP_CAPABILITIY_DIRECT_SOLVER) {
    printf("    A direct linear algebra solver\n");
  }
  if(cap & OSQP_CAPABILITIY_INDIRECT_SOLVER) {
    printf("    An indirect linear algebra solver\n");
  }
  if(cap & OSQP_CAPABILITIY_CODEGEN) {
    printf("    Code generation\n");
  }
  printf("\n");

  /* Setup solver */
  exitflag = osqp_setup(&solver, P, q, A, l, u, m, n, settings);

  /* Solve problem */
  if (!exitflag) exitflag = osqp_solve(solver);

  /* Cleanup */
  osqp_cleanup(solver);
  if (A) free(A);
  if (P) free(P);
  if (settings) free(settings);

  return (int)exitflag;
}
