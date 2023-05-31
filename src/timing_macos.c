/*
 * Timing functions for macOS.
 */
#include "timing.h"
#include "osqp_configure.h"
#include "types.h"

#include <mach/mach_time.h>

/* Use MAC OSX mach_time for timing */
struct OSQPTimer_ {
  uint64_t                  tic;
  uint64_t                  toc;
  mach_timebase_info_data_t tinfo;
};


/* Create the timer */
OSQPTimer* OSQPTimer_new() {
    return c_malloc(sizeof(struct OSQPTimer_));
}

/* Destroy the timer */
void OSQPTimer_free(OSQPTimer* t) {
    if (t) c_free(t);
}

void osqp_tic(OSQPTimer* t)
{
  /* Read current clock cycles */
  t->tic = mach_absolute_time();
}

OSQPFloat osqp_toc(OSQPTimer* t)
{
  uint64_t duration; /* Elapsed time in clock cycles*/

  t->toc   = mach_absolute_time();
  duration = t->toc - t->tic;

  /* Conversion from clock cycles to nanoseconds*/
  mach_timebase_info(&(t->tinfo));
  duration *= t->tinfo.numer;
  duration /= t->tinfo.denom;

  return (OSQPFloat)duration / 1e9;
}
