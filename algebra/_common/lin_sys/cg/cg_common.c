
#include "cg_common.h"
#include "glob_opts.h"

OSQPFloat cg_compute_tolerance(OSQPInt    admm_iter,
                               OSQPFloat  rhs_norm,
                               OSQPFloat  scaled_prim_res,
                               OSQPFloat  scaled_dual_res,
                               OSQPFloat  reduction_factor,
                               OSQPFloat* eps_prev) {

  OSQPFloat eps = 1.0;

  if (admm_iter == 1) {
    // In case rhs = 0.0 we don't want to set eps_prev to 0.0
    if (rhs_norm < OSQP_CG_TOL_MIN)
      *eps_prev = 1.0;
    else
      *eps_prev = rhs_norm * reduction_factor;

    // Return early since scaled_prim_res and scaled_dual_res are meaningless before the first ADMM iteration
    return *eps_prev;
  }

  eps = reduction_factor * c_sqrt(scaled_prim_res * scaled_dual_res);
  eps = c_max(c_min(eps, (*eps_prev)), OSQP_CG_TOL_MIN);
  *eps_prev = eps;

  return eps;
}
