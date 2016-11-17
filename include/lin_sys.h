/* KKT linear system definition and solution */

#ifndef LIN_SYS_H
#define LIN_SYS_H

#include "cs.h"

/* stores the necessary private workspace, only the linear system solver
 * interacts with this struct */
typedef struct c_priv Priv;

/* stores the necessary workspace for polishing */
typedef struct c_polish Polish;


// Initialize private variable for solver
// NB: Only the upper triangular part of P is stuffed!
Priv *init_priv(const csc * P, const csc * A, const Settings *settings);

/* solves Ax = b for x, and stores result in b */
c_int solve_lin_sys(const Settings *settings, Priv *p, c_float *b);

// Free LDL Factorization structure
void free_priv(Priv *p);

// Solution polishing: Solve equality constrained QP with assumed active constr.
void polish(Work *work);


// TODO: Remove these functions

// Initialize private variable with given matrix L, and vector D and P
// Priv *set_priv(csc *L, c_float *D, c_int *P);

// Form KKT matrix
// csc * form_KKT(const csc * P, const  csc * A, c_float rho);

#endif
