#ifndef PARDISO_INTERFACE_H
#define PARDISO_INTERFACE_H


#include "osqp.h"
#include "types.h"  //OSQPMatrix and OSQPVector[fi] types

/**
 * Pardiso solver structure
 *
 * NB: If we use Pardiso, we suppose that OSQP_EMBEDDED_MODE is not enabled
 */
typedef struct pardiso pardiso_solver;

struct pardiso {
    enum osqp_linsys_solver_type type;

    /**
     * @name Functions
     * @{
     */
    const char* (*name)(struct pardiso* self);

    OSQPInt (*solve)(struct pardiso* self,
                     OSQPVectorf*    b,
                     OSQPInt         admm_iter);

    void (*update_settings)(struct pardiso*     self,
                            const OSQPSettings* settings);

    void (*warm_start)(struct pardiso*    self,
                       const OSQPVectorf* x);

    OSQPInt (*adjoint_derivative)(struct pardiso* self);

    void (*free)(struct pardiso* self);

    OSQPInt (*update_matrices)(struct pardiso*   self,
                               const OSQPMatrix* P,
                               const OSQPInt*    Px_new_idx,
                               OSQPInt           P_new_n,
                               const OSQPMatrix* A,
                               const OSQPInt*    Ax_new_idx,
                               OSQPInt           A_new_n);

    OSQPInt (*update_rho_vec)(struct pardiso*    self,
                              const OSQPVectorf* rho_vec,
                              OSQPFloat          rho_sc);

    OSQPInt nthreads;
    /** @} */


    /**
     * @name Attributes
     * @{
     */
    // Attributes
    OSQPCscMatrix* KKT;         ///< KKT matrix (in CSR format!)
    OSQPInt*       KKT_i;       ///< KKT column indices in 1-indexing for Pardiso
    OSQPInt*       KKT_p;       ///< KKT row pointers in 1-indexing for Pardiso
    OSQPFloat*     bp;          ///< workspace memory for solves (rhs)
    OSQPFloat*     sol;         ///< solution to the KKT system
    OSQPFloat*     rho_inv_vec; ///< parameter vector
    OSQPFloat      sigma;       ///< scalar parameter
    OSQPFloat      rho_inv;     ///< scalar parameter (used if rho_inv_vec == NULL)
    OSQPInt        polishing;   ///< polishing flag
    OSQPInt        n;           ///< number of QP variables
    OSQPInt        m;           ///< number of QP constraints

    // Pardiso variables
    void*     pt[64];     ///< internal solver memory pointer pt
    OSQPInt   iparm[64];  ///< Pardiso control parameters
    OSQPInt   nKKT;       ///< dimension of the linear system
    OSQPInt   mtype;      ///< matrix type (-2 for real and symmetric indefinite)
    OSQPInt   nrhs;       ///< number of right-hand sides (1 for our needs)
    OSQPInt   maxfct;     ///< maximum number of factors (1 for our needs)
    OSQPInt   mnum;       ///< indicates matrix for the solution phase (1 for our needs)
    OSQPInt   phase;      ///< control the execution phases of the solver
    OSQPInt   error;      ///< the error indicator (0 for no error)
    OSQPInt   msglvl;     ///< Message level information (0 for no output)
    OSQPInt   idum;       ///< dummy integer
    OSQPFloat fdum;       ///< dummy float

    // These are required for matrix updates
    OSQPInt* PtoKKT;    ///< Index of elements from P to KKT matrix
    OSQPInt* AtoKKT;    ///< Index of elements from A to KKT matrix
    OSQPInt* rhotoKKT;  ///< Index of rho places in KKT matrix

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
OSQPInt init_linsys_solver_pardiso(pardiso_solver**    sp,
                                   const OSQPMatrix*   P,
                                   const OSQPMatrix*   A,
                                   const OSQPVectorf*  rho_vec,
                                   const OSQPSettings* settings,
                                   OSQPInt             polishing);


/**
 * Get the user-friendly name of the MKL Pardiso solver.
 * @return The user-friendly name
 */
const char* name_pardiso(pardiso_solver* s);


/**
 * Solve linear system and store result in b
 * @param  s        Linear system solver structure
 * @param  b        Right-hand side
 * @return          Exitflag
 */
OSQPInt solve_linsys_pardiso(pardiso_solver* s,
                             OSQPVectorf*    b,
                             OSQPInt         admm_iter);

void update_settings_linsys_solver_pardiso(pardiso_solver* s,
                                           const OSQPSettings* settings);

void update_settings_linsys_solver_pardiso(pardiso_solver* s,
                                           const OSQPSettings* settings);


void warm_start_linsys_solver_pardiso(pardiso_solver*   s,
                                      const OSQPVectorf* x);

/**
 * Update linear system solver matrices
 * @param  s        Linear system solver structure
 * @param  P        Matrix P
 * @param  A        Matrix A
 * @return          Exitflag
 */
OSQPInt update_linsys_solver_matrices_pardiso(pardiso_solver*   s,
                                              const OSQPMatrix* P,
                                              const OSQPInt*    Px_new_idx,
                                              OSQPInt           P_new_n,
                                              const OSQPMatrix* A,
                                              const OSQPInt*    Ax_new_idx,
                                              OSQPInt           A_new_n);


/**
 * Update rho parameter in linear system solver structure
 * @param  s        Linear system solver structure
 * @param  rho_vec  new rho_vec value
 * @return          exitflag
 */
OSQPInt update_linsys_solver_rho_vec_pardiso(pardiso_solver*    s,
                                             const OSQPVectorf* rho_vec,
                                             OSQPFloat          rho_sc);


/**
 * Free linear system solver
 * @param s linear system solver object
 */
void free_linsys_solver_pardiso(pardiso_solver* s);


#endif /* PARDISO_INTERFACE_H */
