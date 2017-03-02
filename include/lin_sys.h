/* KKT linear system definition and solution */

#ifndef LIN_SYS_H
#define LIN_SYS_H


#ifdef __cplusplus
extern "C" {
#endif

#ifndef EMBEDDED
#include "cs.h"
#endif

#include "types.h"


/* solves Ax = b for x, and stores result in b */
c_int solve_lin_sys(const OSQPSettings *settings, Priv *p, c_float *b);


#ifndef EMBEDDED

// Initialize private variable for solver
// NB: Only the upper triangular part of P is stuffed!
Priv *init_priv(const csc * P, const csc * A, const OSQPSettings *settings,
                c_int polish);

// Free LDL Factorization structure
void free_priv(Priv *p);

#endif  // #end EMBEDDED


#ifdef __cplusplus
}
#endif

#endif
