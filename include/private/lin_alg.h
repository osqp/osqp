#ifndef LIN_ALG_H
#define LIN_ALG_H

# include "algebra_vector.h"
# include "algebra_matrix.h"
# include "types.h"

# ifdef __cplusplus
extern "C" {
# endif

/* Return which linear system solvers are supported */
c_int osqp_algebra_linsys_supported(void);

/* Return the default linear system the algebra backend prefers */
enum osqp_linsys_solver_type osqp_algebra_default_linsys(void);

/* Initialize libraries that implement algebra. */
c_int osqp_algebra_init_libs(c_int device);

/* Free libraries that implement algebra. */
void osqp_algebra_free_libs(void);

/* Get the name of the linear algebra backend */
const char* osqp_algebra_name(void);

/* KKT linear system definition and solution */

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
c_int osqp_algebra_init_linsys_solver(LinSysSolver      **s,
                                      const OSQPMatrix   *P,
                                      const OSQPMatrix   *A,
                                      const OSQPVectorf  *rho_vec,
                                      const OSQPSettings *settings,
                                      c_float            *scaled_prim_res,
                                      c_float            *scaled_dual_res,
                                      c_int               polishing);


# ifdef __cplusplus
}
# endif

#endif /* ifndef LIN_ALG_H */
