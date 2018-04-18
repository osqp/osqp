/*
 * Interface for OSQP signal handling.
 */

#ifndef CTRLC_H
# define CTRLC_H

# ifdef __cplusplus
extern "C" {
# endif // ifdef __cplusplus

# include "glob_opts.h"

# if defined MATLAB

/* No header file available here; define the prototypes ourselves */
bool utIsInterruptPending(void);
bool utSetInterruptEnabled(bool);

# elif defined IS_WINDOWS

/* Use Windows SetConsoleCtrlHandler for signal handling */
#  include <windows.h>

# else // if defined MATLAB

/* Use sigaction for signal handling on non-Windows machines */
#  include <signal.h>

# endif // if defined MATLAB

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
int  isInterrupted(void);


# ifdef __cplusplus
}
# endif // ifdef __cplusplus


#endif /* END IFDEF CTRLC */
