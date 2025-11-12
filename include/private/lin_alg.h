#ifndef LIN_ALG_H
#define LIN_ALG_H

#include "osqp.h"
#include "algebra_vector.h"
#include "algebra_matrix.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 *  Opaque struct holding information about the linear algebra system.
 *  Implementation specific.
 */
typedef struct OSQPAlgebraContext_ OSQPAlgebraContext;

/* Return which linear system solvers are supported */
OSQPInt osqp_algebra_linsys_supported(void);

/* Return the default linear system the algebra backend prefers */
enum osqp_linsys_solver_type osqp_algebra_default_linsys(void);

/* Initialize libraries that implement algebra. */
OSQPInt osqp_algebra_init_ctx(OSQPAlgebraContext** alg_context, OSQPInt device);

/* Free libraries that implement algebra. */
void osqp_algebra_free_ctx(OSQPAlgebraContext* alg_context);

/* Get the name of the linear algebra backend */
OSQPInt osqp_algebra_name(OSQPAlgebraContext* alg_context, char* name, OSQPInt nameLen);

/* Get the name of the device the linear algebra backend is using */
OSQPInt osqp_algebra_device_name(OSQPAlgebraContext* alg_context, char* name, OSQPInt nameLen);

/* KKT linear system definition and solution */

/**
 * Linear system solver structure.
 */

typedef struct linsys_solver LinSysSolver;

/**
 * Define linsys_solver prototype structure
 *
 * NB: The details are defined when the linear solver is initialized depending
 *      on the choice
 */
struct linsys_solver {
  enum osqp_linsys_solver_type type;             ///< linear system solver type functions

  const char* (*name)(LinSysSolver* self);

  OSQPInt (*solve)(LinSysSolver* self,
                   OSQPVectorf*  b,
                   OSQPInt       admm_iter);

  void (*update_settings)(LinSysSolver*       self,
                          const OSQPSettings* settings);

  void (*warm_start)(LinSysSolver*      self,
                     const OSQPVectorf* x);

# ifndef OSQP_EMBEDDED_MODE
  OSQPInt (*adjoint_derivative)(LinSysSolver* self);

  void (*free)(LinSysSolver* self);         ///< free linear system solver (only in desktop version)
# endif // ifndef OSQP_EMBEDDED_MODE

# if OSQP_EMBEDDED_MODE != 1
  OSQPInt (*update_matrices)(LinSysSolver*     self,
                             const OSQPMatrix* P,            ///< update matrices P
                             const OSQPInt*    Px_new_idx,
                             OSQPInt           P_new_n,
                             const OSQPMatrix* A,            //   and A in the solver
                             const OSQPInt*    Ax_new_idx,
                             OSQPInt           A_new_n);

  OSQPInt (*update_rho_vec)(LinSysSolver*      self,
                            const OSQPVectorf* rho_vec,
                            OSQPFloat          rho_sc);  ///< Update rho_vec
# endif // if OSQP_EMBEDDED_MODE != 1

  OSQPInt nthreads; ///< number of threads active
};

// NB: Only the upper triangular part of P is filled.

/**
 * Initialize linear system solver structure
 * @param   s                Pointer to linear system solver structure
 * @param   P                Objective function matrix
 * @param   A                Constraint matrix
 * @param   rho_vec          Algorithm parameter
 * @param   settings         Solver settings
 * @param   scaled_prim_res  Pointer to the scaled primal residual
 * @param   scaled_dual_res  Pointer to the scaled dual residual
 * @param   polishing        0/1 depending whether we are allocating for polishing or not
 * @return                   Exitflag for error (0 if no errors)
 */
OSQPInt osqp_algebra_init_linsys_solver(LinSysSolver**      s,
                                        const OSQPMatrix*   P,
                                        const OSQPMatrix*   A,
                                        const OSQPVectorf*  rho_vec,
                                        const OSQPSettings* settings,
                                        OSQPFloat*          scaled_prim_res,
                                        OSQPFloat*          scaled_dual_res,
                                        OSQPInt             polishing);


#ifdef OSQP_ALGEBRA_BUILTIN
#ifndef OSQP_EMBEDDED_MODE
OSQPInt adjoint_derivative_linsys_solver(LinSysSolver**      s,
                                         const OSQPSettings* settings,
                                         const OSQPMatrix*   P,
                                         const OSQPMatrix*   G,
                                         const OSQPMatrix*   A_eq,
                                         OSQPMatrix*         GDiagLambda,
                                         OSQPVectorf*        slacks,
                                         OSQPVectorf*        rhs);

#endif
#endif

#ifdef __cplusplus
}
#endif

#endif /* ifndef LIN_ALG_H */
