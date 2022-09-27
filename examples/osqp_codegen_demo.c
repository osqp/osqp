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
  OSQPInt   n = 2;
  OSQPInt   m = 3;

  /* Exitflag */
  OSQPInt exitflag = 0;

  /* Solver, settings, matrices */
  OSQPSolver*    solver   = NULL;
  OSQPSettings*  settings = NULL;
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

  if(exitflag) {
    printf( "  OSQP errored during setup: %s\n", osqp_error_message(exitflag) );
    return exitflag;
  }

  /* Test codegen */
  OSQPCodegenDefines *defs = (OSQPCodegenDefines *)calloc(1, sizeof(OSQPCodegenDefines));

  /* Get the default codegen options */
  osqp_set_default_codegen_defines( defs );

  /* Sample with vector update only */
  defs->embedded_mode = 1;
  exitflag = osqp_codegen(solver, vecDirPath, "mpc_vec_", defs);

  if(exitflag) {
    printf( "  OSQP errored during vector code genreation: %s\n", osqp_error_message(exitflag) );
    return exitflag;
  }

  /* Sample with both vector and matrix updates */
  defs->embedded_mode = 2;
  exitflag = osqp_codegen(solver, matDirPath, "mpc_mat_", defs);

  if(exitflag) {
    printf( "  OSQP errored during matrix code genreation: %s\n", osqp_error_message(exitflag) );
    return exitflag;
  }

  /* Solve problem */
  exitflag = osqp_solve(solver);

  if(exitflag) {
    printf( "  OSQP errored during solve: %s\n", osqp_error_message(exitflag) );
    return exitflag;
  }

  /* Cleanup */
  osqp_cleanup(solver);
  if (A) free(A);
  if (P) free(P);
  if (settings) free(settings);

  return (int)exitflag;
}
