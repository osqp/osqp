#ifndef PARDISO_INTERFACE_H
#define PARDISO_INTERFACE_H


#include "osqp.h"
#include "types.h"  //OSQPMatrix and OSQPVector[fi] types

/**
 * Pardiso solver structure
 *
 * NB: If we use Pardiso, we suppose that EMBEDDED is not enabled
 */
typedef struct pardiso pardiso_solver;

struct pardiso {
    enum osqp_linsys_solver_type type;

    /**
     * @name Functions
     * @{
     */
    const char* (*name)(void);

    c_int (*solve)(struct pardiso *self,
                   OSQPVectorf    *b,
                   c_int           admm_iter);

    void (*update_settings)(struct pardiso     *self,
                            const OSQPSettings *settings);

    void (*warm_start)(struct pardiso    *self,
                       const OSQPVectorf *x);

    void (*free)(struct pardiso * self);

    c_int (*update_matrices)(struct pardiso   *self,
                             const OSQPMatrix *P,
                             const c_int* Px_new_idx,
                             c_int P_new_n,
                             const OSQPMatrix *A,
                             const c_int* Ax_new_idx,
                             c_int A_new_n);

    c_int (*update_rho_vec)(struct pardiso    *self,
                            const OSQPVectorf *rho_vec,
                            c_float            rho_sc);

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
    c_float rho_inv;        ///< scalar parameter (used if rho_inv_vec == NULL)
    c_int polishing;        ///< polishing flag
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
    c_int * PtoKKT, * AtoKKT;    ///< Index of elements from P and A to KKT matrix
    c_int * rhotoKKT;            ///< Index of rho places in KKT matrix

    /** @} */
};


/**
 * Initialize Pardiso Solver
 *
 * @param  s         Pointer to a private structure
 * @param  P         Objective function matrix (upper triangular form)
 * @param  A         Constraints matrix
 * @param  rho_vec   Algorithm parameter. If polishing, then rho_vec = OSQP_NULL.
 * @param  settings  Solver settings
 * @param  polishing Flag whether we are initializing for polishing or not
 * @return           Exitflag for error (0 if no errors)
 */
c_int init_linsys_solver_pardiso(pardiso_solver    **sp,
                                 const OSQPMatrix   *P,
                                 const OSQPMatrix   *A,
                                 const OSQPVectorf  *rho_vec,
                                 const OSQPSettings *settings,
                                 c_int               polishing);


/**
 * Get the user-friendly name of the MKL Pardiso solver.
 * @return The user-friendly name
 */
const char* name_pardiso();


/**
 * Solve linear system and store result in b
 * @param  s        Linear system solver structure
 * @param  b        Right-hand side
 * @return          Exitflag
 */
c_int solve_linsys_pardiso(pardiso_solver *s,
                           OSQPVectorf    *b,
                           c_int           admm_iter);

void update_settings_linsys_solver_pardiso(pardiso_solver *s,
                                           const OSQPSettings    *settings);

void update_settings_linsys_solver_pardiso(pardiso_solver *s,
                                           const OSQPSettings    *settings);


void warm_start_linsys_solver_pardiso(pardiso_solver    *s,
                                      const OSQPVectorf *x);

/**
 * Update linear system solver matrices
 * @param  s        Linear system solver structure
 * @param  P        Matrix P
 * @param  A        Matrix A
 * @return          Exitflag
 */
c_int update_linsys_solver_matrices_pardiso(
                    pardiso_solver * s,
                    const OSQPMatrix *P,
                    const c_int *Px_new_idx,
                    c_int P_new_n,
                    const OSQPMatrix *A,
                    const c_int *Ax_new_idx,
                    c_int A_new_n);


/**
 * Update rho parameter in linear system solver structure
 * @param  s        Linear system solver structure
 * @param  rho_vec  new rho_vec value
 * @return          exitflag
 */
c_int update_linsys_solver_rho_vec_pardiso(pardiso_solver    *s,
                                           const OSQPVectorf *rho_vec,
                                           c_float            rho_sc);


/**
 * Free linear system solver
 * @param s linear system solver object
 */
void free_linsys_solver_pardiso(pardiso_solver * s);


#endif /* PARDISO_INTERFACE_H */
