#include "osqp.h"
#include <stdio.h>
#include <mpc_vec_workspace.h>

int main(void) {

  /* Problem data to use to update the solver */
  OSQPFloat l[3] = { 0.0, -1.0, -1.0, };
  OSQPFloat u[3] = { 3.0, 2.0, 2.0, };

  /* Exitflag */
  OSQPInt exitflag;

  printf( "Embedded test program for vector updates.\n" );

  /* Solve problem */
  exitflag = osqp_solve( &mpc_vec_solver );

  if( exitflag > 0 ) {
    printf( "  OSQP errored: %s\n", osqp_error_message(exitflag) );
    return (int)exitflag;
  } else {
    printf( "  Solved workspace with no error.\n" );
  }

  /* Update vectors in the workspace */
  printf( "  Updating l and u vector.\n" );
  exitflag = osqp_update_data_vec( &mpc_vec_solver, NULL, l, u );

  if( exitflag > 0 ) {
    printf( "  OSQP errored: %s\n", osqp_error_message(exitflag) );
    return (int)exitflag;
  } else {
    printf( "  Updated workspace vectors with no error.\n" );
  }

  /* Solve the updated problem */
  exitflag = osqp_solve( &mpc_vec_solver );

  if( exitflag > 0 ) {
    printf( "  OSQP errored: %s\n", osqp_error_message(exitflag) );
    return (int)exitflag;
  } else {
    printf( "  Solved updated workspace with no error.\n" );
  }


  return (int)exitflag;
}
