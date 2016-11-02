/* KKT linear system definition and solution */

#ifndef LIN_SYS_H
#define LIN_SYS_H

#include "cs.h"

/* solves Ax = b for x, and stores result in b */
// c_int solveLinSys(const c_priv *p, c_float *b);

/* TODO: This function is here just for a quick test. It should not be here */
void LDLSolve(c_float *x, c_float *b, const csc *L, const c_float *D,
              const c_int *P, c_float *bp);

#endif
