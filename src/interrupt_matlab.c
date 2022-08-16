/*
 * Implements interrupt handling using ctrl-c for MATLAB mex files.
 */

#include "interrupt.h"

/* No header file available here; define the prototypes ourselves */
bool utIsInterruptPending(void);
bool utSetInterruptEnabled(bool);

static int istate;

void osqp_start_interrupt_listener(void) {
  istate = utSetInterruptEnabled(1);
}

void osqp_end_interrupt_listener(void) {
  utSetInterruptEnabled(istate);
}

int osqp_is_interrupted(void) {
  return utIsInterruptPending();
}
