/*
 * Test file to compile the generated workspaces from the test suite.
 */

#include <stdio.h>
#include "osqp.h"

#include "rho_is_vec_0_embedded_1_workspace.h"
#include "rho_is_vec_1_embedded_1_workspace.h"

#include "scaling_0_embedded_1_workspace.h"
#include "scaling_1_embedded_1_workspace.h"

int main() {
  c_int exitflag;

  printf( "Embedded test program for embedded mode 1 settings.\n");

  /*
   * rho_is_vec = 0
   */
  exitflag = osqp_solve( &rho_is_vec_0_embedded_1_solver );

  if( exitflag > 0 ) {
    printf( "  OSQP errored on rho_is_vec_0: %s\n", osqp_error_message(exitflag));
    return (int)exitflag;
  } else {
    printf( "  Solved rho_is_vec_0 with no error.\n" );
  }


  /*
   * rho_is_vec = 1
   */
  exitflag = osqp_solve( &rho_is_vec_1_embedded_1_solver );

  if( exitflag > 0 ) {
    printf( "  OSQP errored on rho_is_vec_1: %s\n", osqp_error_message(exitflag));
    return (int)exitflag;
  } else {
    printf( "  Solved rho_is_vec_1 with no error.\n" );
  }


  /*
   * scaling = 0
   */
  exitflag = osqp_solve( &scaling_0_embedded_1_solver );

  if( exitflag > 0 ) {
    printf( "  OSQP errored on scaling_0: %s\n", osqp_error_message(exitflag));
    return (int)exitflag;
  } else {
    printf( "  Solved scaling_0 with no error.\n" );
  }


  /*
   * scaling = 1
   */
  exitflag = osqp_solve( &scaling_1_embedded_1_solver );

  if( exitflag > 0 ) {
    printf( "  OSQP errored on scaling_1: %s\n", osqp_error_message(exitflag));
    return (int)exitflag;
  } else {
    printf( "  Solved scaling_1 with no error.\n" );
  }
}
