#include "osqp.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

/* Sample problems */
#include "problems/cvxqp2_s.h"
#include "problems/hs21.h"
#include "problems/hs35.h"
#include "problems/qpcblend.h"
#include "problems/qptest.h"
#include "problems/largeqp.h"
#include "problems/primalc1.h"

#define NAME_BUF_LENGTH 20

#define PREMADE_PROBLEM(name) printf("Using problem data: " #name "\n"); \
                              qp_n = name ## _data_n;                    \
                              qp_m = name ## _data_m;                    \
                              qp_P = & name ## _data_P_csc;              \
                              qp_q = name ## _data_q_val;                \
                              qp_A = & name ## _data_A_csc;              \
                              qp_l = name ## _data_l_val;                \
                              qp_u = name ## _data_u_val;                \
                              dynamic_matrices = 0;                      \
                              dynamic_settings = 0;                      \
                              settings = & name ## _settings;            \
                              settings->verbose = 1;

int main(int argc, char *argv[]) {

  /* Problem data for a simple problem */
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
  OSQPInt exitflag;
  OSQPInt dynamic_matrices = 0;
  OSQPInt dynamic_settings = 0;

  /* Solver, settings, matrices */
  OSQPSolver*   solver   = NULL;
  OSQPSettings* settings = NULL;

  OSQPInt qp_n = 0;
  OSQPInt qp_m = 0;
  OSQPFloat* qp_q = NULL;
  OSQPFloat* qp_l = NULL;
  OSQPFloat* qp_u = NULL;
  OSQPCscMatrix* qp_P = NULL;
  OSQPCscMatrix* qp_A = NULL;

  // Extract a problem name
  char problem_name[NAME_BUF_LENGTH];

  if( argc == 2) {
    snprintf(problem_name, NAME_BUF_LENGTH, "%s", argv[1]);
  }

  if( strcmp(problem_name, "cvxqp2_s") == 0 ) {
    /*
     * CVXQP2_S problem from the Maros Mesaros problem set
     */
    PREMADE_PROBLEM(cvxqp2_s);
  }
  else if( strcmp(problem_name, "qpcblend") == 0 ) {
    /*
     * QPCBLEND problem from the Maros Mesaros problem set
     */
    PREMADE_PROBLEM(qpcblend);
  }
  else if( strcmp(problem_name, "largeqp") == 0 ) {
    PREMADE_PROBLEM(largeqp);

    // This problem uses dynamic settings
    dynamic_settings = 1;
  }
  else if( strcmp(problem_name, "qptest") == 0 ) {
    /*
     * QPTEST problem from the Maros Mesaros problem set
     */
    PREMADE_PROBLEM(qptest);
  }
  else if( strcmp(problem_name, "hs21") == 0 ) {
    /*
     * HS21 problem from the Maros Mesaros problem set
     */
    PREMADE_PROBLEM(hs21);
  }
  else if( strcmp(problem_name, "hs35") == 0 ) {
    /*
     * HS35 problem from the Maros Mesaros problem set
     */
    PREMADE_PROBLEM(hs35);
  }
  else if( strcmp(problem_name, "primalc1") == 0 ) {
    /*
     * PRIMALC1 problem from the Maros Mesaros problem set
     */
    PREMADE_PROBLEM(primalc1);
    settings->scaling = 1;
    settings->restart_enable = 0;
  }
  else {
    printf("Using problem data: default\n");
    // Problem dimensions
    qp_n = n;
    qp_m = m;

    // Cost function
    qp_P = malloc(sizeof(OSQPCscMatrix));
    qp_q = q;

    // Constraints
    qp_A = malloc(sizeof(OSQPCscMatrix));
    qp_l = l;
    qp_u = u;

    dynamic_matrices = 1;

    /* Populate matrices */
    csc_set_data(qp_A, m, n, A_nnz, A_x, A_i, A_p);
    csc_set_data(qp_P, n, n, P_nnz, P_x, P_i, P_p);
  }

  /* Set default settings */
  if( dynamic_settings || !settings ) {
    settings = (OSQPSettings *)malloc(sizeof(OSQPSettings));
    dynamic_settings = 1;

    if (settings) {
      osqp_set_default_settings(settings);
      settings->polishing = 1;
      settings->scaling = 1;

      //settings->linsys_solver = OSQP_DIRECT_SOLVER;
      //settings->linsys_solver = OSQP_INDIRECT_SOLVER;
    }
  }

  settings->scaled_termination = 0;

  settings->restart_enable = 1;
  //settings->adaptive_rho = OSQP_ADAPTIVE_RHO_UPDATE_TIME;
  settings->adaptive_rho = OSQP_ADAPTIVE_RHO_UPDATE_DISABLED;
  settings->rho = 1.0;

  OSQPInt cap = osqp_capabilities();

  printf("This OSQP library supports:\n");
  if(cap & OSQP_CAPABILITY_DIRECT_SOLVER) {
    printf("    A direct linear algebra solver\n");
  }
  if(cap & OSQP_CAPABILITY_INDIRECT_SOLVER) {
    printf("    An indirect linear algebra solver\n");
  }
  if(cap & OSQP_CAPABILITY_CODEGEN) {
    printf("    Code generation\n");
  }
  if(cap & OSQP_CAPABILITY_DERIVATIVES) {
    printf("    Derivatives calculation\n");
  }
  printf("\n");

  /* Setup solver */
  exitflag = osqp_setup(&solver, qp_P, qp_q, qp_A, qp_l, qp_u, qp_m, qp_n, settings);

  /* Solve problem */
  if (!exitflag) exitflag = osqp_solve(solver);

  /* Cleanup */
  osqp_cleanup(solver);

  if( dynamic_matrices ) {
    if (qp_A) free(qp_A);
    if (qp_P) free(qp_P);
  }

  if (settings && dynamic_settings)
    free(settings);

  return (int)exitflag;
}
