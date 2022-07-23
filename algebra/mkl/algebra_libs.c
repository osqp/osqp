#include "osqp_api_constants.h"
#include "osqp_api_types.h"

c_int osqp_algebra_linsys_supported(void) {
  /* Has both Paradiso (direct solver) and a PCG solver (indirect solver) */
  return OSQP_CAPABILITIY_DIRECT_SOLVER | OSQP_CAPABILITIY_INDIRECT_SOLVER;
}

c_int osqp_algebra_init_libs(void) {return 0;}

void osqp_algebra_free_libs(void) {return;}
