#ifndef MKL_CG_INTERFACE_H
#define MKL_CG_INTERFACE_H


#include "osqp.h"
#include "types.h"    //OSQPMatrix and OSQPVector[fi] types
#include <mkl_rci.h>  //MKL_INT


typedef struct mklcg_solver_ {

  enum osqp_linsys_solver_type type;

  /**
   * @name Functions
   * @{
   */
  const char* (*name)(struct mklcg_solver_* self);
  OSQPInt (*solve)(struct mklcg_solver_* self, OSQPVectorf* b, OSQPInt admm_iter);
  void    (*update_settings)(struct mklcg_solver_* self, const OSQPSettings* settings);
  void    (*warm_start)(struct mklcg_solver_* self, const OSQPVectorf* x);
  OSQPInt (*adjoint_derivative)(struct mklcg_solver_* self);
  void    (*free)(struct mklcg_solver_* self);
  OSQPInt (*update_matrices)(struct mklcg_solver_* self,
                             const  OSQPMatrix*    P,
                             const  OSQPInt*       Px_new_idx,
                                    OSQPInt        P_new_n,
                             const  OSQPMatrix*    A,
                             const  OSQPInt*       Ax_new_idx,
                                    OSQPInt        A_new_n);
  OSQPInt (*update_rho_vec)(struct mklcg_solver_* self,
                            const OSQPVectorf* rho_vec,
                                  OSQPFloat    rho_sc);

  //threads count
  OSQPInt nthreads;

  // Maximum number of iterations
  OSQPInt max_iter;

   /* @name Attributes
   * @{
   */
  // Attributes
  OSQPMatrix*  P;               // The P matrix provided by OSQP (just a pointer, don't delete it!)
  OSQPMatrix*  A;               // The A matrix provided by OSQP (just a pointer, don't delete it!)
  OSQPVectorf* rho_vec;         // The rho vector provided by OSQP (just a pointer, don't delete it!)
  OSQPFloat*   scaled_prim_res; // The primal residual provided by OSQP (just a pointer)
  OSQPFloat*   scaled_dual_res; // The dual residual provided by OSQP (just a pointer)
  OSQPFloat    sigma;           // The sigma value provided by OSQP
  OSQPInt      m;               // Number of constraints
  OSQPInt      n;               // Number of variables
  OSQPInt      polish;          // Polishing or not?

  osqp_precond_type precond_type; // Preconditioner to use

  // Adaptable termination variables
  OSQPFloat eps_prev;   // Tolerance for previous ADMM iteration

  OSQPInt   reduction_interval; // Number of iterations between reduction factor updates
  OSQPFloat reduction_factor;   // Amount to change tolerance by each iteration
  OSQPFloat tol_fraction;       // Tolerance (fraction of ADMM residuals)

  // Count for the number of consecutive iterations that no CG iterations have been required
  OSQPInt cg_zero_iters;

  // Hold an internal copy of the solution x to
  // enable warm starting between successive solves
  OSQPVectorf* x;

  // A work array for intermediate CG products
  OSQPVectorf* ywork;

  // MKL CG internal data
  MKL_INT      iparm[128];      ///< MKL control parameters (integer)
  double       dparm[128];      ///< MKL control parameters (double)
  OSQPVectorf* tmp;             ///< MKL work array

  // NB: the work array must be accessed by MKL directly through
  // its underlying pointer, but we make it an OSQPVectorf
  // so that we can make some views into it for multiplication

  // Vector views into tmp for K*mvm_pre = mvm_post
  OSQPVectorf* mvm_pre;
  OSQPVectorf* mvm_post;

  // Vector views into tmp for preconditioner application
  OSQPVectorf* precond_pre;
  OSQPVectorf* precond_post;

  // Vector views of the input vector
  OSQPVectorf* r1;
  OSQPVectorf* r2;

  // Preconditioner vector
  OSQPVectorf* precond;
  OSQPVectorf* precond_inv;
} mklcg_solver;



/**
 * Initialize MKL Conjugate Gradient Solver
 *
 * @param s               Pointer to a private structure
 * @param P               Cost function matrix (upper triangular form)
 * @param A               Constraints matrix
 * @param rho_vec         Algorithm parameter. If polish, then rho_vec = OSQP_NULL.
 * @param settings        Solver settings
 * @param scaled_prim_res Pointer to OSQP's scaled primal residual
 * @param scaled_dual_res Pointer to OSQP's scaled dual residual
 * @param polish          Flag whether we are initializing for polish or not
 * @return                Exitflag for error (0 if no errors)
 */
OSQPInt init_linsys_mklcg(mklcg_solver**     sp,
                          const OSQPMatrix*  P,
                          const OSQPMatrix*  A,
                          const OSQPVectorf* rho_vec,
                          const OSQPSettings*settings,
                                OSQPFloat*   scaled_prim_res,
                                OSQPFloat*   scaled_dual_res,
                                OSQPInt      polish);


/**
 * Get the user-friendly name of the MKL CG solver.
 * @return The user-friendly name
 */
const char* name_mklcg(mklcg_solver* s);


/**
 * Solve linear system and store result in b
 * @param  s        Linear system solver structure
 * @param  b        Right-hand side
 * @return          Exitflag
 */
OSQPInt solve_linsys_mklcg(mklcg_solver* s, OSQPVectorf* b, OSQPInt admm_iter);


void update_settings_linsys_solver_mklcg(mklcg_solver*      s,
                                         const OSQPSettings* settings);


void warm_start_linys_mklcg(mklcg_solver*      s,
                            const OSQPVectorf* x);


/**
 * Update linear system solver matrices
 * @param  s        Linear system solver structure
 * @param  P        Matrix P
 * @param  A        Matrix A
 * @return          Exitflag
 */
OSQPInt update_matrices_linsys_mklcg(mklcg_solver* s,
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
OSQPInt update_rho_linsys_mklcg(mklcg_solver* s,
                                const OSQPVectorf* rho_vec,
                                OSQPFloat rho_sc);


/**
 * Free linear system solver
 * @param s linear system solver object
 */
void free_linsys_mklcg(mklcg_solver* s);


#endif /* ifndef MKL_CG_INTERFACE_H */

