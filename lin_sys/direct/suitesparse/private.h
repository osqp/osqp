#ifndef PRIVATE_H
#define PRIVATE_H

#include "osqp.h"
#include "amd.h"
#include "ldl.h"

struct c_priv{
    csc *L;         /* lower triangular matrix in LDL factorization */
    c_float *Dinv;  /* inverse of diag matrix in LDL (as a vector)  */
    c_int *P;       /* permutation of KKT matrix for factorization  */
    c_float *bp;    /* workspace memory for solves                  */
};


#endif
