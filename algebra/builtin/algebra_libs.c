#include "osqp_api_constants.h"
#include "osqp_api_types.h"
#include "qdldl_interface.h"

OSQPInt osqp_algebra_linsys_supported(void) {
  /* Only has QDLDL (direct solver) */
  return OSQP_CAPABILITY_DIRECT_SOLVER;
}

enum osqp_linsys_solver_type osqp_algebra_default_linsys(void) {
  /* Prefer QDLDL (it is also the only one available) */
  return OSQP_DIRECT_SOLVER;
}

OSQPInt osqp_algebra_init_libs(OSQPInt device) {return 0;}

void osqp_algebra_free_libs(void) {return;}

const char* osqp_algebra_name(void) {
  return "Built-in";
}

#ifndef OSQP_EMBEDDED_MODE

// Initialize linear system solver structure
// NB: Only the upper triangular part of P is filled
OSQPInt osqp_algebra_init_linsys_solver(LinSysSolver**      s,
                                        const OSQPMatrix*   P,
                                        const OSQPMatrix*   A,
                                        const OSQPVectorf*  rho_vec,
                                        const OSQPSettings* settings,
                                        OSQPFloat*          scaled_prim_res,
                                        OSQPFloat*          scaled_dual_res,
                                        OSQPInt             polishing) {

  switch (settings->linsys_solver) {
  default:
  case OSQP_DIRECT_SOLVER:
    return init_linsys_solver_qdldl((qdldl_solver **)s, P, A, rho_vec, settings, polishing);
  }
}

OSQPInt adjoint_derivative_linsys_solver(LinSysSolver**      s,
                                         const OSQPSettings* settings,
                                         const OSQPMatrix*   P,
                                         const OSQPMatrix*   G,
                                         const OSQPMatrix*   A_eq,
                                         OSQPMatrix*         GDiagLambda,
                                         OSQPVectorf*        slacks,
                                         OSQPVectorf*        rhs) {

return adjoint_derivative_qdldl((qdldl_solver **)s, P, G, A_eq, GDiagLambda, slacks, rhs);
}

#endif
