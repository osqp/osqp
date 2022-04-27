#include "custom_interrupts.h"

static int counter = 0;

void osqp_start_interrupt_listener(void) {
  counter = 0;
}

void osqp_end_interrupt_listener(void) {
  /* Do nothing */
}

int osqp_is_interrupted(void) {
  return 1;

  counter = counter + 1;

  /* A simple example showing how this can be used by just replicating the iteration counter */
  /* Return 0 for all iterations that should continue, 1 for when it should stop */
  return ( counter < 5 ) ? 0 : 1;
}
