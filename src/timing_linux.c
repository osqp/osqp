/*
 * Timing functions for Linux.
 */
#include "timing.h"
#include "osqp_configure.h"
#include "types.h"

/* Use POSIX clock_gettime() for timing on non-Windows machines */
#include <time.h>
#include <sys/time.h>

struct OSQPTimer_ {
  struct timespec tic;
  struct timespec toc;
};


/* Create the timer */
OSQPTimer* OSQPTimer_new() {
    return c_malloc(sizeof(struct OSQPTimer_));
}

/* Destroy the timer */
void OSQPTimer_free(OSQPTimer* t) {
    if (t) c_free(t);
}

/* Read current time */
void osqp_tic(OSQPTimer* t) {
  clock_gettime(CLOCK_MONOTONIC, &t->tic);
}

/* Return time passed since last call to tic on this timer */
OSQPFloat osqp_toc(OSQPTimer* t) {
  struct timespec temp;

  clock_gettime(CLOCK_MONOTONIC, &t->toc);

  if ((t->toc.tv_nsec - t->tic.tv_nsec) < 0) {
    temp.tv_sec  = t->toc.tv_sec - t->tic.tv_sec - 1;
    temp.tv_nsec = 1e9 + t->toc.tv_nsec - t->tic.tv_nsec;
  } else {
    temp.tv_sec  = t->toc.tv_sec - t->tic.tv_sec;
    temp.tv_nsec = t->toc.tv_nsec - t->tic.tv_nsec;
  }
  return (OSQPFloat)temp.tv_sec + (OSQPFloat)temp.tv_nsec / 1e9;
}
