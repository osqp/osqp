/*
 * Timing functions for Windows.
 */
#include "timing.h"
#include "osqp_configure.h"
#include "types.h"

/* Some R packages clash with elements of the windows.h header, so use a
   slimmer version for conflict avoidance (we really don't need much from it). */
#define NOGDI

#include <windows.h>

struct OSQPTimer_ {
  LARGE_INTEGER tic;
  LARGE_INTEGER toc;
  LARGE_INTEGER freq;
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
  QueryPerformanceFrequency(&t->freq);
  QueryPerformanceCounter(&t->tic);
}

OSQPFloat osqp_toc(OSQPTimer* t)
{
  QueryPerformanceCounter(&t->toc);
  return (t->toc.QuadPart - t->tic.QuadPart) / (OSQPFloat)t->freq.QuadPart;
}
