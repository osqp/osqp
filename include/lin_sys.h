/* KKT linear system definition and solution */

#ifndef LIN_SYS_H
#define LIN_SYS_H

#include "cs.h"

/* stores the necessary private workspace, only the linear system solver
 * interacts with this struct */
typedef struct c_priv Priv;


// Initialize private variable for solver
// NB: Only the upper triangular part of P is stuffed!
Priv *initPriv(const csc * P, const csc * A, const Settings *settings);

/* solves Ax = b for x, and stores result in b */
c_int solveLinSys(const Settings *settings, Priv *p, c_float *b);



// TODO: Remove these functions
// Initialize private variable with given matrix L, and vector D and P
Priv *setPriv(csc *L, c_float *D, c_int *P);

// Form KKT matrix
csc * formKKT(const csc * P, const  csc * A, c_float rho);

#endif
