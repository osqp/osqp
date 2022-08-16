#ifndef INTERRUPT_H_
#define INTERRUPT_H_

/*
 * Interface for interrupting the OSQP solver.
 */

/**
 * Start listener for interrupts
 */
void osqp_start_interrupt_listener(void);

/**
 * End listener for interrupts
 */
void osqp_end_interrupt_listener(void);

/**
 * Check if the solver has been interrupted
 * @return  Boolean indicating if the solver has been interrupted
 */
int osqp_is_interrupted(void);


#endif /* ifndef INTERRUPT_H_ */
