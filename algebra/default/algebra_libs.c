#include "osqp_api_constants.h"
#include "osqp_api_types.h"

c_int osqp_algebra_linsys_supported(void) {
  /* Only has QDLDL (direct solver) */
  return OSQP_CAPABILITIY_DIRECT_SOLVER;
}

c_int osqp_algebra_init_libs(c_int device) {return 0;}

void osqp_algebra_free_libs(void) {return;}
