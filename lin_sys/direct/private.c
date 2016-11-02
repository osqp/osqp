#include "private.h"


// TODO: Add functions for defining factorizing and solving linear systems with direct methods

// formKKT, factorize... (see scs)


/* Form KKT matrix of the form
[Q + rhoI,   F',  G';
 F      G,  -1./rhoI]

Arguments
---------
Q : cost matrix (upper triangular part)
Ft: Transpose of linear equalities
Gt: transpose of linear inequalities
rho: ADMM step
N.B. Only the upper triangular part is stuffed!
*/
csc * formKKT(csc * Q, csc * F, csc *G, c_float rho);

/* TODO: Adjust arguments of the function with other linear system solvers */
c_int solveLinSys(const c_priv *p, scs_float *b) {
    /* returns solution to linear system */
    /* Ax = b with solution stored in b */
    LDLSolve(b, b, p->L, p->D, p->P, p->bp);
    return 0;
}

void LDLSolve(c_float *x, c_float *b, csc *L, c_float *D, c_int *P,
              c_float *bp) {
    /* solves PLDL'P' x = b for x */
    c_int n = L->n;
    LDL_perm(n, bp, b, P);
    LDL_lsolve(n, bp, L->p, L->i, L->x);
    LDL_dsolve(n, bp, D);
    LDL_ltsolve(n, bp, L->p, L->i, L->x);
    LDL_permt(n, x, bp, P);
}
