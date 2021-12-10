#ifndef MKL_CG_INTERFACE_H
#define MKL_CG_INTERFACE_H


#include "osqp.h"
#include "types.h"    //OSQPMatrix and OSQPVector[fi] types
#include <mkl_rci.h>  //MKL_INT


typedef struct mklcg_solver_ {

  enum linsys_solver_type type;

  /**
   * @name Functions
   * @{
   */
  c_int (*solve)(struct mklcg_solver_ *self, OSQPVectorf * b, c_int admm_iter);
  void (*update_settings)(struct mklcg_solver_ *self, const OSQPSettings *settings);
  void (*warm_start)(struct mklcg_solver_ *self, const OSQPVectorf *x);
  void (*free)(struct mklcg_solver_ *self);
  c_int (*update_matrices)(struct mklcg_solver_ * self,
                           const OSQPMatrix *P,
                           const OSQPMatrix *A);
  c_int (*update_rho_vec)(struct mklcg_solver_ *self,
                          const OSQPVectorf * rho_vec,
                          c_float rho_sc);

  //threads count
  c_int nthreads;

   /* @name Attributes
   * @{
   */
  // Attributes
  OSQPMatrix  *P;       // The P matrix provided by OSQP (just a pointer, don't delete it!)
  OSQPMatrix  *A;       // The A matrix provided by OSQP (just a pointer, don't delete it!)
  OSQPVectorf *rho_vec; // The rho vector provided by OSQP (just a pointer, don't delete it!)
  c_float     sigma;    // The sigma value provided by OSQP
  c_int       m;        // number of constraints
  c_int       n;        // number of variables
  c_int       polish;   //polishing or not?

  // Hold an internal copy of the solution x to
  // enable warm starting between successive solves
  OSQPVectorf *x;

  // A work array for intermediate CG products
  OSQPVectorf* ywork;

  // MKL CG internal data
  MKL_INT     iparm[128];       ///< MKL control parameters (integer)
  double      dparm[128];       ///< MKL control parameters (double)
  OSQPVectorf* tmp;             ///< MKL work array

  // NB: the work array must be accessed by MKL directly through
  // its underlying pointer, but we make it an OSQPVectorf
  // so that we can make some views into it for multiplication

  // Vector views into tmp for K*v1 = v2
  OSQPVectorf *v1, *v2;

  // Vector views of the input vector
  OSQPVectorf *r1, *r2;

} mklcg_solver;



/**
 * Initialize MKL Conjugate Gradient Solver
 *
 * @param  s         Pointer to a private structure
 * @param  P         Cost function matrix (upper triangular form)
 * @param  A         Constraints matrix
 * @param  rho_vec   Algorithm parameter. If polish, then rho_vec = OSQP_NULL.
 * @param  settings  Solver settings
 * @param  polish    Flag whether we are initializing for polish or not
 * @return           Exitflag for error (0 if no errors)
 */
c_int init_linsys_mklcg(mklcg_solver       **sp,
                        const OSQPMatrix    *P,
                        const OSQPMatrix    *A,
                        const OSQPVectorf   *rho_vec,
                        const OSQPSettings  *settings,
                        c_int                polish);


/**
 * Solve linear system and store result in b
 * @param  s        Linear system solver structure
 * @param  b        Right-hand side
 * @return          Exitflag
 */
c_int solve_linsys_mklcg(mklcg_solver * s, OSQPVectorf * b, c_int admm_iter);


void update_settings_linsys_solver_mklcg(mklcg_solver *s,
                                           const OSQPSettings    *settings);


void warm_start_linys_mklcg(mklcg_solver    *s,
                                      const OSQPVectorf *x);


/**
 * Update linear system solver matrices
 * @param  s        Linear system solver structure
 * @param  P        Matrix P
 * @param  A        Matrix A
 * @return          Exitflag
 */
c_int update_matrices_linsys_mklcg(
          mklcg_solver * s,
          const OSQPMatrix *P,
          const OSQPMatrix *A);


/**
 * Update rho parameter in linear system solver structure
 * @param  s        Linear system solver structure
 * @param  rho_vec  new rho_vec value
 * @return          exitflag
 */
c_int update_rho_linsys_mklcg(
            mklcg_solver * s,
            const OSQPVectorf * rho_vec,
            c_float rho_sc);


/**
 * Free linear system solver
 * @param s linear system solver object
 */
void free_linsys_mklcg(mklcg_solver * s);


#endif /* ifndef MKL_CG_INTERFACE_H */

