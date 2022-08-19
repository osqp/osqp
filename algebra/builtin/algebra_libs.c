#include "osqp_api_constants.h"
#include "osqp_api_types.h"
#include "qdldl_interface.h"

c_int osqp_algebra_linsys_supported(void) {
  /* Only has QDLDL (direct solver) */
  return OSQP_CAPABILITIY_DIRECT_SOLVER;
}

enum osqp_linsys_solver_type osqp_algebra_default_linsys(void) {
  /* Prefer QDLDL (it is also the only one available) */
  return OSQP_DIRECT_SOLVER;
}

c_int osqp_algebra_init_libs(c_int device) {return 0;}

void osqp_algebra_free_libs(void) {return;}

const char* osqp_algebra_name(void) {
  return "Built-in";
}

#ifndef EMBEDDED

// Initialize linear system solver structure
// NB: Only the upper triangular part of P is filled
c_int osqp_algebra_init_linsys_solver(LinSysSolver      **s,
                                      const OSQPMatrix   *P,
                                      const OSQPMatrix   *A,
                                      const OSQPVectorf  *rho_vec,
                                      const OSQPSettings *settings,
                                      c_float            *scaled_prim_res,
                                      c_float            *scaled_dual_res,
                                      c_int               polishing) {

  switch (settings->linsys_solver) {
  default:
  case OSQP_DIRECT_SOLVER:
    return init_linsys_solver_qdldl((qdldl_solver **)s, P, A, rho_vec, settings, polishing);
  }
}

#endif
