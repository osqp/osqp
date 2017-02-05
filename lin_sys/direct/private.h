#ifndef PRIVATE_H
#define PRIVATE_H

// #include "glob_opts.h"
#include "osqp.h"
// #include "cs.h"
// #include "lin_sys.h"
// #include "kkt.h"
#include "amd.h"
#include "ldl.h"
#include "../common.h"

struct c_priv{
    csc *L;         /* lower triangular matrix in LDL factorization */
    c_float *Dinv;  /* inverse of diag matrix in LDL (as a vector)  */
    c_int *P;       /* permutation of KKT matrix for factorization  */
    c_float *bp;    /* workspace memory for solves                  */
    /* reporting */
    // c_float total_solve_time;
};


#endif
