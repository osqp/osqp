/*
 * Implements interrupt using ctrl-c on unix (linux + macos) systems.
 */

#include "interrupt.h"
#include <signal.h>

static int int_detected;
struct sigaction oact;

static void handle_ctrlc(int dummy) {
  int_detected = dummy ? dummy : -1;
}

void osqp_start_interrupt_listener(void) {
  struct sigaction act;

  int_detected = 0;
  act.sa_flags = 0;
  sigemptyset(&act.sa_mask);
  act.sa_handler = handle_ctrlc;
  sigaction(SIGINT, &act, &oact);
}

void osqp_end_interrupt_listener(void) {
  struct sigaction act;

  sigaction(SIGINT, &oact, &act);
}

int osqp_is_interrupted(void) {
  return int_detected;
}
