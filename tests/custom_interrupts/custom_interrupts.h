#ifndef OSQP_CUSTOM_INTERRUPTS_H
#define OSQP_CUSTOM_INTERRUPTS_H

# ifdef __cplusplus
extern "C" {
# endif

/* Must have the same signature as the functions in interrupt_defs.h else branch */
void osqp_start_interrupt_listener(void);
void osqp_end_interrupt_listener(void);
int osqp_is_interrupted(void);

# ifdef __cplusplus
}
# endif

#endif /* ifndef  OSQP_CUSTOM_INTERRUPTS_H */
