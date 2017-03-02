#ifndef PRIVATE_H
#define PRIVATE_H

// #include "osqp.h"
#include "types.h"
#include "lin_alg.h"
#include "kkt.h"
#include "amd.h"
#include "ldl.h"

struct c_priv{
    csc *L;         /* lower triangular matrix in LDL factorization */
    c_float *Dinv;  /* inverse of diag matrix in LDL (as a vector)  */
    c_int *P;       /* permutation of KKT matrix for factorization  */
    c_float *bp;    /* workspace memory for solves                  */


    #if EMBEDDED != 1
    // These are required for matrix updates
    csc * KKT;                 // KKT matrix in sparse form (used to update P and A matrices)
    c_int * PtoKKT, * AtoKKT;  // Index of elements from P and A to KKT matrix
    c_int * Pinv;              // Inverse of permuation matrix stored as vector
    // LDL Numeric workspace
    c_int *Lnz;                 // Number of nonzeros in each column of L
    c_float *Y;                 // LDL Numeric workspace
    c_int *Pattern, *Flag;      // LDL Numeric workspace
    c_int *Parent;              // LDL numeric workspace
    #endif

};


#endif
