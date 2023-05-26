#ifndef QDLDL_INTERFACE_H
#define QDLDL_INTERFACE_H


#include "osqp.h"
#include "types.h"  //OSQPMatrix and OSQPVector[fi] types
#include "qdldl_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * QDLDL solver structure
 */
typedef struct qdldl qdldl_solver;

struct qdldl {
    enum osqp_linsys_solver_type type;

    /**
     * @name Functions
     * @{
     */
    const char* (*name)(struct qdldl* s);

    OSQPInt (*solve)(struct qdldl*       self,
                            OSQPVectorf* b,
                            OSQPInt      admm_iter);

    void (*update_settings)(struct qdldl*        self,
                            const  OSQPSettings* settings);

    void (*warm_start)(struct qdldl*       self,
                       const  OSQPVectorf* x);

#ifndef OSQP_EMBEDDED_MODE
    OSQPInt (*adjoint_derivative)(struct qdldl* self);

    void (*free)(struct qdldl* self); ///< Free workspace (only if desktop)
#endif

    // This used only in non embedded or embedded 2 version
#if OSQP_EMBEDDED_MODE != 1
    OSQPInt (*update_matrices)(struct qdldl*      self,
                               const  OSQPMatrix* P,
                               const  OSQPInt*    Px_new_idx,
                                      OSQPInt     P_new_n,
                               const  OSQPMatrix* A,
                               const  OSQPInt*    Ax_new_idx,
                                      OSQPInt     A_new_n);   ///< Update solver matrices

    OSQPInt (*update_rho_vec)(struct qdldl*       self,
                              const  OSQPVectorf* rho_vec,
                                     OSQPFloat    rho_sc);    ///< Update rho_vec parameter
#endif

    OSQPInt nthreads;

    /** @} */

    /**
     * @name Attributes
     * @{
     */
    OSQPCscMatrix* L;             ///< lower triangular matrix in LDL factorization
    OSQPFloat*     Dinv;          ///< inverse of diag matrix in LDL (as a vector)
    OSQPInt*       P;             ///< permutation of KKT matrix for factorization
    OSQPFloat*     bp;            ///< workspace memory for solves
    OSQPFloat*     sol;           ///< solution to the KKT system
    OSQPFloat*     rho_inv_vec;   ///< parameter vector
    OSQPFloat      sigma;         ///< scalar parameter
    OSQPFloat      rho_inv;       ///< scalar parameter (used if rho_inv_vec == NULL)
#ifndef OSQP_EMBEDDED_MODE
    OSQPInt        polishing;     ///< polishing flag
#endif
    OSQPInt        n;             ///< number of QP variables
    OSQPInt        m;             ///< number of QP constraints


#if OSQP_EMBEDDED_MODE != 1
    // These are required for matrix updates
    OSQPCscMatrix* KKT;           ///< Permuted KKT matrix in sparse form (used to update P and A matrices)
    OSQPInt*       PtoKKT;        ///< Index of elements from P to KKT matrix
    OSQPInt*       AtoKKT;        ///< Index of elements from A to KKT matrix
    OSQPInt*       rhotoKKT;      ///< Index of rho places in KKT matrix
    // QDLDL Numeric workspace
    QDLDL_float* D;
    QDLDL_int*   etree;
    QDLDL_int*   Lnz;
    QDLDL_int*   iwork;
    QDLDL_bool*  bwork;
    QDLDL_float* fwork;

    OSQPCscMatrix* adj;
#endif

    /** @} */
};



/**
 * Initialize QDLDL Solver
 *
 * @param  s         Pointer to a private structure
 * @param  P         Objective function matrix (upper triangular form)
 * @param  A         Constraints matrix
 * @param  rho_vec   Algorithm parameter. If polish, then rho_vec = OSQP_NULL.
 * @param  settings  Solver settings
 * @param  polishing Flag whether we are initializing for polishing or not
 * @return           Exitflag for error (0 if no errors)
 */
OSQPInt init_linsys_solver_qdldl(qdldl_solver**      sp,
                                 const OSQPMatrix*   P,
                                 const OSQPMatrix*   A,
                                 const OSQPVectorf*  rho_vec,
                                 const OSQPSettings* settings,
                                 OSQPInt             polishing);

/**
 * Get the user-friendly name of the QDLDL solver.
 * @return The user-friendly name
 */
const char* name_qdldl(qdldl_solver* s);

/**
 * Solve linear system and store result in b
 * @param  s        Linear system solver structure
 * @param  b        Right-hand side
 * @return          Exitflag
 */
OSQPInt solve_linsys_qdldl(qdldl_solver* s,
                           OSQPVectorf*  b,
                           OSQPInt       admm_iter);


void update_settings_linsys_solver_qdldl(qdldl_solver*       s,
                                         const OSQPSettings* settings);

void warm_start_linsys_solver_qdldl(qdldl_solver*      s,
                                    const OSQPVectorf* x);


#if OSQP_EMBEDDED_MODE != 1
/**
 * Update linear system solver matrices
 * @param  s          Linear system solver structure
 * @param  P          Matrix P
 * @param  Px_new_idx elements of P to update,
 * @param  P_new_n    number of elements to update
 * @param  A          Matrix A
 * @param  Ax_new_idx elements of A to update,
 * @param  A_new_n    number of elements to update
 * @return            Exitflag
 */
OSQPInt update_linsys_solver_matrices_qdldl(qdldl_solver*     s,
                                            const OSQPMatrix* P,
                                            const OSQPInt*    Px_new_idx,
                                            OSQPInt           P_new_n,
                                            const OSQPMatrix* A,
                                            const OSQPInt*    Ax_new_idx,
                                            OSQPInt           A_new_n);




/**
 * Update rho_vec parameter in linear system solver structure
 * @param  s        Linear system solver structure
 * @param  rho_vec  new rho_vec value
 * @return          exitflag
 */
OSQPInt update_linsys_solver_rho_vec_qdldl(qdldl_solver*      s,
                                           const OSQPVectorf* rho_vec,
                                           OSQPFloat          rho_sc);

#endif

#ifndef OSQP_EMBEDDED_MODE
/**
 * Free linear system solver
 * @param s linear system solver object
 */
void free_linsys_solver_qdldl(qdldl_solver* s);

OSQPInt adjoint_derivative_qdldl(qdldl_solver*      s,
                                 const OSQPMatrix*  P,
                                 const OSQPMatrix*  G,
                                 const OSQPMatrix*  A_eq,
                                 const OSQPMatrix*  GDiagLambda,
                                 const OSQPVectorf* slacks,
                                 const OSQPVectorf* rhs);

#endif

#ifdef __cplusplus
}
#endif

#endif /* QDLDL_INTERFACE_H */
