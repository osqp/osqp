#include "osqp.h"
#include <stdlib.h>
#include <stdio.h>

int main(int argc, char *argv[]) {

  char vecDirPath[255];
  char matDirPath[255];

  switch(argc) {
    case 3:
      sprintf(matDirPath, "%s", argv[2]);
      /* Fall through to set the vector path */

    case 2:
      sprintf(vecDirPath, "%s", argv[1]) ;
      break;

    default:
      sprintf(vecDirPath, "");
      sprintf(matDirPath, "");
  }

  printf("OSQP code generation demo program.\n\n");
  printf("This demo will write a sample solver to two different sets of files,\n");
  printf("one supporting only vector updates and the other supporting updates of\n");
  printf("both vectors and matrices.\n\n");
  printf("The directory to generate the vector workspace files into can be given\n");
  printf("as the first argument, and the directory to generate the matrix workspace\n");
  printf("can be given as the second argument. If neither arguments are given, then\n");
  printf("the workspaces are exported into the current directory.\n\n");

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
  OSQPCscMatrix* P = malloc(sizeof(OSQPCscMatrix));
  OSQPCscMatrix* A = malloc(sizeof(OSQPCscMatrix));

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

  /* Setup solver */
  exitflag = osqp_setup(&solver, P, q, A, l, u, m, n, settings);

  /* Test codegen */
  OSQPCodegenDefines *defs = (OSQPCodegenDefines *)malloc(sizeof(OSQPCodegenDefines));

  defs->float_type = 0;        /* Use doubles */
  defs->printing_enable = 0;   /* Don't enable printing */
  defs->profiling_enable = 0;  /* Don't enable profiling */
  defs->interrupt_enable = 0;  /* Don't enable interrupts */

  /* Sample with vector update only */
  defs->embedded_mode = 1;
  osqp_codegen(solver, vecDirPath, "mpc_vec_", defs);

  /* Sample with both vector and matrix updates */
  defs->embedded_mode = 2;
  osqp_codegen(solver, matDirPath, "mpc_mat_", defs);

  /* Solve problem */
  if (!exitflag) exitflag = osqp_solve(solver);

  /* Cleanup */
  osqp_cleanup(solver);
  if (A) free(A);
  if (P) free(P);
  if (settings) free(settings);

  return (int)exitflag;
}
