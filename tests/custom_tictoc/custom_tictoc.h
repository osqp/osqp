/*
 * Header file demonstrating definitions for custom functions for the tictoc
 * timing functions. This header file is included in OSQP using the
 * -DOSQP_CUSTOM_TICTOC flag.
 */
#ifndef CUSTOM_TICTOC_H_
#define CUSTOM_TICTOC_H_


# ifdef __cplusplus
extern "C" {
# endif /* ifdef __cplusplus */

#include "types.h"
#include "glob_opts.h"

void osqp_tic(OSQPTimer *timer);
c_float osqp_toc(OSQPTimer *timer);

struct OSQP_TIMER {
  int cnt;
};


# ifdef __cplusplus
}
# endif /* ifdef __cplusplus */

#endif
