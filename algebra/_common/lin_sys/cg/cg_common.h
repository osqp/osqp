
#ifndef CG_COMMON_H_
#define CG_COMMON_H_

#include "osqp.h"

/*
 * Compute an updated tolerance to use when determining if the CG
 * iteration is finished.
 *
 * @param admm_iter         The current outer ADMM iteration number
 * @param rhs_norm          The current norm of the linsys' RHS
 * @param scaled_prim_res   The current scaled primal residual in the outer ADMM
 * @param scaled_dual_res   The current scaled dual residual in the outer ADMM
 * @param reduction_factor  The factor to reduce the tolerance by each time
 * @param eps_rev           The previous CG tolerance
 *
 * @return The new CG tolerance
 */
OSQPFloat cg_compute_tolerance(OSQPInt    admm_iter,
                               OSQPFloat  rhs_norm,
                               OSQPFloat  scaled_prim_res,
                               OSQPFloat  scaled_dual_res,
                               OSQPFloat  reduction_factor,
                               OSQPFloat* eps_prev);

#endif
