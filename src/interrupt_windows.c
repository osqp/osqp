/*
 * Implements interrupt using ctrl-c on Windows.
 */

#include "interrupt.h"
#include <windows.h>

/* Use Windows SetConsoleCtrlHandler for signal handling */
static int int_detected;
static BOOL WINAPI handle_ctrlc(DWORD dwCtrlType) {
  if (dwCtrlType != CTRL_C_EVENT) return FALSE;

  int_detected = 1;
  return TRUE;
}

void osqp_start_interrupt_listener(void) {
  int_detected = 0;
  SetConsoleCtrlHandler(handle_ctrlc, TRUE);
}

void osqp_end_interrupt_listener(void) {
  SetConsoleCtrlHandler(handle_ctrlc, FALSE);
}

int osqp_is_interrupted(void) {
  return int_detected;
}
