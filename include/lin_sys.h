/* KKT linear system definition and solution */

#ifndef LIN_SYS_H
#define LIN_SYS_H


#ifdef __cplusplus
extern "C" {
#endif

// #include "cs.h"
#include "types.h"

#ifndef EMBEDDED
// Initialize private variable for solver
// NB: Only the upper triangular part of P is stuffed!
Priv *init_priv(const csc * P, const csc * A,
                const OSQPSettings *settings, c_int polish);

// Free LDL Factorization structure
void free_priv(Priv *p);

#endif


#if EMBEDDED != 1
// Update private structure with new P and A
c_int update_priv(Priv * p, const csc *P, const csc *A,
                  const OSQPWorkspace * work, const OSQPSettings *settings);
#endif

/* solves Ax = b for x, and stores result in b */
c_int solve_lin_sys(const OSQPSettings *settings, Priv *p, c_float *b);


#ifdef __cplusplus
}
#endif

#endif
