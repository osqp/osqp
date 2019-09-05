#ifndef PARDISO_H
#define PARDISO_H

#ifdef __cplusplus
extern "C" {
#endif

#include "lin_alg.h"
#include "kkt.h"

/**
 * Pardiso solver structure
 *
 * NB: If we use Pardiso, we suppose that EMBEDDED is not enabled
 */
typedef struct pardiso pardiso_solver;

struct pardiso {
    enum linsys_solver_type type;

    /**
     * @name Functions
     * @{
     */
    c_int (*solve)(struct pardiso * self, c_float * b);

    void (*free)(struct pardiso * self); ///< Free workspace (only if desktop)

    c_int (*update_matrices)(struct pardiso * self, const csc *P, const csc *A);    ///< Update solver matrices
    c_int (*update_rho_vec)(struct pardiso * self, const c_float * rho_vec);        ///< Update rho_vec parameter

    c_int nthreads;
    /** @} */


    /**
     * @name Attributes
     * @{
     */
    // Attributes
    csc *KKT;               ///< KKT matrix (in CSR format!)
    c_int *KKT_i;           ///< KKT column indices in 1-indexing for Pardiso
    c_int *KKT_p;           ///< KKT row pointers in 1-indexing for Pardiso
    c_float *bp;            ///< workspace memory for solves (rhs)
    c_float *sol;           ///< solution to the KKT system
    c_float *rho_inv_vec;   ///< parameter vector
    c_float sigma;          ///< scalar parameter
    c_int polish;           ///< polishing flag
    c_int n;                ///< number of QP variables
    c_int m;                ///< number of QP constraints

    // Pardiso variables
    void *pt[64];     ///< internal solver memory pointer pt
    c_int iparm[64];  ///< Pardiso control parameters
    c_int nKKT;       ///< dimension of the linear system
    c_int mtype;      ///< matrix type (-2 for real and symmetric indefinite)
    c_int nrhs;       ///< number of right-hand sides (1 for our needs)
    c_int maxfct;     ///< maximum number of factors (1 for our needs)
    c_int mnum;       ///< indicates matrix for the solution phase (1 for our needs)
    c_int phase;      ///< control the execution phases of the solver
    c_int error;      ///< the error indicator (0 for no error)
    c_int msglvl;     ///< Message level information (0 for no output)
    c_int idum;       ///< dummy integer
    c_float fdum;     ///< dummy float

    // These are required for matrix updates
    c_int * Pdiag_idx, Pdiag_n;  ///< index and number of diagonal elements in P
    c_int * PtoKKT, * AtoKKT;    ///< Index of elements from P and A to KKT matrix
    c_int * rhotoKKT;            ///< Index of rho places in KKT matrix

    /** @} */
};


/**
 * Initialize Pardiso Solver
 *
 * @param  s         Pointer to a private structure
 * @param  P         Cost function matrix (upper triangular form)
 * @param  A         Constraints matrix
 * @param  sigma     Algorithm parameter. If polish, then sigma = delta.
 * @param  rho_vec   Algorithm parameter. If polish, then rho_vec = OSQP_NULL.
 * @param  polish    Flag whether we are initializing for polish or not
 * @return           Exitflag for error (0 if no errors)
 */
c_int init_linsys_solver_pardiso(pardiso_solver ** sp, const csc * P, const csc * A, c_float sigma, const c_float * rho_vec, c_int polish);


/**
 * Solve linear system and store result in b
 * @param  s        Linear system solver structure
 * @param  b        Right-hand side
 * @return          Exitflag
 */
c_int solve_linsys_pardiso(pardiso_solver * s, c_float * b);


/**
 * Update linear system solver matrices
 * @param  s        Linear system solver structure
 * @param  P        Matrix P
 * @param  A        Matrix A
 * @return          Exitflag
 */
c_int update_linsys_solver_matrices_pardiso(pardiso_solver * s, const csc *P, const csc *A);


/**
 * Update rho parameter in linear system solver structure
 * @param  s        Linear system solver structure
 * @param  rho_vec  new rho_vec value
 * @return          exitflag
 */
c_int update_linsys_solver_rho_vec_pardiso(pardiso_solver * s, const c_float * rho_vec);


/**
 * Free linear system solver
 * @param s linear system solver object
 */
void free_linsys_solver_pardiso(pardiso_solver * s);

#ifdef __cplusplus
}
#endif

#endif
