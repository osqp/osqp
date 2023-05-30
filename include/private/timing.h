#ifndef TIMING_H_
#define TIMING_H_

#include "osqp_configure.h"
#include "types.h"

/**
 * Timer Methods
 */

#ifdef __cplusplus
extern "C" {
#endif

#ifdef OSQP_ENABLE_PROFILING

/**
 * Create a new timer.
 * @return the timer
 */
OSQPTimer* OSQPTimer_new();

/**
 * Free an existing timer.
 * @param t Timer object to destroy
 */
void OSQPTimer_free(OSQPTimer* t);

/**
 * Start timer
 * @param t Timer object
 */
void osqp_tic(OSQPTimer* t);

/**
 * Report time
 * @param  t Timer object
 * @return   Reported time
 */
OSQPFloat osqp_toc(OSQPTimer* t);

#endif /* #ifdef OSQP_ENABLE_PROFILING */

#ifdef __cplusplus
}
#endif

#endif /* #ifdef TIMING_H_ */
