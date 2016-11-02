#ifndef PRIVATE_H
#define PRIVATE_H

#include "glob_opts.h"
#include "osqp.h"
#include "cs.h"
#include "amd.h"
#include "ldl.h"
#include "../common.h"

struct c_priv{
    csc *L;       /* lower triagnular matrix in LDL factorization */
    c_float *D;   /* diagonal matrix in LDL factorization         */
    c_int *P;     /* permutation of KKT matrix for factorization  */
    c_float *bp;  /* workspace memory for solves                  */
    /* reporting */
    c_float solveTime;
};

//TODO: Add structure for KKT factorization (see scs)




#endif
