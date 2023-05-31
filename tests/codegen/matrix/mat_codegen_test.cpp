#include "osqp.h"
#include <stdio.h>
#include <mpc_mat_workspace.h>

int main(void) {

  /* Problem data to use to update the solver */
  OSQPFloat l[3] = { 0.0, -1.0, -1.0, };
  OSQPFloat u[3] = { 3.0, 2.0, 2.0, };

  OSQPFloat P_x[2] = { 4.5, 2.5, };
  OSQPInt   P_new_idx[2] = { 0, 2 };

  OSQPFloat A_x[2] = { 2.0, 3.0, };
  OSQPInt   A_new_idx[2] = {0, 2};

  /* Exitflag */
  OSQPInt exitflag;

  printf( "Embedded test program for matrix updates.\n" );

  /* Solve problem */
  exitflag = osqp_solve( &mpc_mat_solver );

  if( exitflag > 0 ) {
    printf( "  OSQP errored: %s\n", osqp_error_message(exitflag) );
    return (int)exitflag;
  } else {
    printf( "  Solved workspace with no error.\n" );
  }

  /* Update vectors in the workspace */
  printf( "  Updating l and u vector.\n" );
  exitflag = osqp_update_data_vec( &mpc_mat_solver, NULL, l, u );

  if( exitflag > 0 ) {
    printf( "  OSQP errored: %s\n", osqp_error_message(exitflag) );
    return (int)exitflag;
  } else {
    printf( "  Updated workspace vectors with no error.\n" );
  }

  /* Solve the updated problem */
  exitflag = osqp_solve( &mpc_mat_solver );

  if( exitflag > 0 ) {
    printf( "  OSQP errored: %s\n", osqp_error_message(exitflag) );
    return (int)exitflag;
  } else {
    printf( "  Solved updated workspace with no error.\n" );
  }

    /* Update matrices in the workspace */
  printf( "  Updating P and A matrices.\n" );
  exitflag = osqp_update_data_mat( &mpc_mat_solver, P_x, P_new_idx, 2, A_x, A_new_idx, 2 );

  if( exitflag > 0 ) {
    printf( "  OSQP errored: %s\n", osqp_error_message(exitflag) );
    return (int)exitflag;
  } else {
    printf( "  Updated workspace matrices with no error.\n" );
  }

  /* Solve the updated problem */
  exitflag = osqp_solve( &mpc_mat_solver );

  if( exitflag > 0 ) {
    printf( "  OSQP errored: %s\n", osqp_error_message(exitflag) );
    return (int)exitflag;
  } else {
    printf( "  Solved updated workspace with no error.\n" );
  }

  return (int)exitflag;
}
