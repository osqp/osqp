/* KKT linear system definition and solution */

#ifndef LIN_SYS_H
#define LIN_SYS_H

#include "cs.h"

/* stores the necessary private workspace, only the linear system solver
 * interacts with this struct */
typedef struct c_priv Priv;


// Initialize private variable for solver
Priv *initPriv(const csc * Q, const csc * A, const Settings *settings);

// Initialize private variable with given matrix L, and vector D and P
Priv *setPriv(csc *L, c_float *D, c_int *P);

/* solves Ax = b for x, and stores result in b */
c_int solveLinSys(const Settings *settings, Priv *p, c_float *b);

// // /* TODO: This function is here just for a quick test. It should not be here */
// void LDLSolve(c_float *x, c_float *b, csc *L, c_float *D,
//               c_int *P, c_float *bp);

#endif
