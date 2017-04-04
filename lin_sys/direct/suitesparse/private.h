#ifndef PRIVATE_H
#define PRIVATE_H

#include "types.h"
#include "lin_alg.h"
#include "kkt.h"
#include "ldl.h"

#ifndef EMBEDDED
#include "amd.h"
#endif

struct c_priv{
    csc *L;         /* lower triangular matrix in LDL factorization */
    c_float *Dinv;  /* inverse of diag matrix in LDL (as a vector)  */
    c_int *P;       /* permutation of KKT matrix for factorization  */
    c_float *bp;    /* workspace memory for solves                  */


    #if EMBEDDED != 1
    // These are required for matrix updates
    c_int * Pdiag_idx, Pdiag_n;  // index and number of diagonal elements in P
    csc * KKT;                   // Permuted KKT matrix in sparse form (used to update P and A matrices)
    c_int * PtoKKT, * AtoKKT;    // Index of elements from P and A to KKT matrix
    // LDL Numeric workspace
    c_int *Lnz;                  // Number of nonzeros in each column of L
    c_float *Y;                  // LDL Numeric workspace
    c_int *Pattern, *Flag;       // LDL Numeric workspace
    c_int *Parent;               // LDL numeric workspace
    #endif

};


#endif
//
