/*
 * This file contains example functions for implementing custom timing
 * routines. The function definitions are located in "custom_timing.h".
 */

#include "custom_tictoc.h"


/*
 * This is a custom osqp_tic function that initiailizes a counter
 * variable to 0.
 */
void osqp_tic(OSQPTimer *timer)
{
  timer->cnt = 0;

  printf( "OSQP timing (tic)\n");
}



/*
 * This is a custom osqp_toc function that simply will count the number of times
 * the timer object has been queried.
 */
c_float osqp_toc(OSQPTimer *timer)
{
  timer->cnt++;
  printf( "OSQP timing (toc): %d calls\n", timer->cnt );

  return timer->cnt;
}
