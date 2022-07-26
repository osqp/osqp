#include "osqp_api_constants.h"
#include "osqp_api_types.h"

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
