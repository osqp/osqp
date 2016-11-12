#ifndef PRIVATE_H
#define PRIVATE_H

#include "glob_opts.h"
#include "osqp.h"
#include "cs.h"
#include "lin_sys.h"
#include "amd.h"
#include "ldl.h"
#include "../common.h"

struct c_priv{
    csc *L;         /* lower triangular matrix in LDL factorization */
    c_float *Dinv;  /* inverse of diag matrix in LDL (as a vector)  */
    c_int *P;       /* permutation of KKT matrix for factorization  */
    c_float *bp;    /* workspace memory for solves                  */
    /* reporting */
    c_float total_solve_time;
};

struct c_polish{
    csc *KKT_trip;  /* KKT matrix in triplet form */
    csc *KKT;       /* KKT matrix in CSC form */
    // c_int *w_KKT,   /* workspace for KKT format conversion */
    csc *L;         /* lower triangular matrix in LDL factorization */
    c_float *Dinv;  /* inverse of diag matrix in LDL (as a vector)  */
    c_int *P;       /* permutation of KKT matrix for factorization  */
    c_float *bp;    /* workspace memory for solves                  */
    /* active constraints */
    c_int *ind_lA, n_lA;  // lower-active
    c_int *ind_uA, n_uA;  // upper-active
    c_int *ind_fA, n_fA;  // free
    c_int *tableA;        // table that maps rows of A to rows of Ared
};

//TODO: Add structure for KKT factorization (see scs)


#endif
