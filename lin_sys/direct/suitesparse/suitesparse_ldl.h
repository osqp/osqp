#ifndef SUITESPARSE_LDL_H
#define SUITESPARSE_LDL_H

#include "types.h"
#include "lin_alg.h"
#include "kkt.h"
#include "ldl.h"

#ifndef EMBEDDED
#include "amd.h"
#endif


typedef struct suitesparse_ldl suitesparse_ldl_solver;

struct suitesparse_ldl {
    enum linsys_solver_type type;
    
    // Functions
    c_int (*solve)(struct suitesparse_ldl * self, c_float * b, const OSQPSettings * settings);
    void (*free)(struct suitesparse_ldl * self);
    #if EMBEDDED != 1
    c_int (*update_matrices)(struct suitesparse_ldl * self, const csc *P, const csc *A, const OSQPWorkspace * work, const OSQPSettings *settings);
    #endif

    // Attributes
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



/**
 * Initialize Suitesparse LDL Solver 
 *
 * @param  P        Cost function matrix (upper triangular form)
 * @param  A        Constraints matrix
 * @param  settings Solver settings
 * @param  polish   Flag whether we are initializing for polish or not
 * @return          Initialized private structure
 */
suitesparse_ldl_solver *init_linsys_solver_suitesparse_ldl(const csc * P, const csc * A, const OSQPSettings *settings, c_int polish);
	
	
// Solve linear system
c_int solve_linsys_suitesparse_ldl(suitesparse_ldl_solver * s, c_float * b, const OSQPSettings * settings);


#if EMBEDDED != 1
// Update system matrices
c_int update_linsys_solver_matrices_suitesparse_ldl(suitesparse_ldl_solver * s, 
		const csc *P, const csc *A, const OSQPWorkspace * work, const OSQPSettings *settings);
#endif


// Free linear system solver
void free_linsys_solver_suitesparse_ldl(suitesparse_ldl_solver * s);


#endif
