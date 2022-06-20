#ifndef LIN_SYS_H
#define LIN_SYS_H

/* KKT linear system definition and solution */

# include "types.h"


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
c_int init_linsys_solver(LinSysSolver      **s,
                         const OSQPMatrix   *P,
                         const OSQPMatrix   *A,
                         const OSQPVectorf  *rho_vec,
                         const OSQPSettings *settings,
                         c_float            *scaled_prim_res,
                         c_float            *scaled_dual_res,
                         c_int               polishing);

#ifndef EMBEDDED

c_int adjoint_derivative_linsys_solver(LinSysSolver **s,
                                       const OSQPSettings *settings,
                                       const OSQPMatrix *P,
                                       const OSQPMatrix *G,
                                       const OSQPMatrix *A_eq,
                                       OSQPMatrix *GDiagLambda,
                                       OSQPVectorf *slacks,
                                       OSQPVectorf *rhs);

#endif

#endif /* ifndef LIN_SYS_H */
