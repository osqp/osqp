#include "osqp_configure.h"
#include "osqp_api_constants.h"
#include "osqp_api_types.h"
#include <mkl.h>

c_int osqp_algebra_linsys_supported(void) {
  /* Has both Paradiso (direct solver) and a PCG solver (indirect solver) */
  return OSQP_CAPABILITIY_DIRECT_SOLVER | OSQP_CAPABILITIY_INDIRECT_SOLVER;
}

enum osqp_linsys_solver_type osqp_algebra_default_linsys(void) {
  /* Prefer Pardiso */
  return OSQP_DIRECT_SOLVER;
}

c_int osqp_algebra_init_libs(c_int device) {
    c_int retval = 0;

#ifdef DLONG
    retval = mkl_set_interface_layer(MKL_INTERFACE_ILP64);
#else
    retval = mkl_set_interface_layer(MKL_INTERFACE_LP64);
#endif

    // Positive value is the interface chosen, so -1 is the error condition
    if(retval == -1)
        return 1;

    return 0;
}

void osqp_algebra_free_libs(void) {return;}
