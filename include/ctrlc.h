/*
 * Interface for OSQP signal handling.
 */

#ifndef CTRLC_H
#define CTRLC_H

#if defined MATLAB

/* No header file available here; define the prototypes ourselves */
int IsInterruptPending(void);
int SetInterruptEnabled(int x);

#elif defined IS_WINDOWS

/* Use Windows SetConsoleCtrlHandler for signal handling */
#include <windows.h>

#else

/* Use sigaction for signal handling on non-Windows machines */
#include <signal.h>

#endif

/* METHODS are the same for both */

/**
 * Start listner for ctrl-c interrupts
 */
void startInterruptListener(void);

/**
 * End listner for ctrl-c interrupts
 */
void endInterruptListener(void);

/**
 * Check if the solver has been interrupted
 * @return  Boolean indicating if the solver has been interrupted
 */
int isInterrupted(void);


#endif /* END IFDEF CTRLC */
