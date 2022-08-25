#include "osqp_configure.h"
#include "osqp_api_constants.h"
#include "osqp_api_types.h"

#include "pardiso_interface.h"
#include "mkl-cg_interface.h"

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

#ifdef OSQP_USE_LONG
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

const char* osqp_algebra_name(void) {
  return "MKL";
}

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
        return init_linsys_solver_pardiso((pardiso_solver **)s, P, A, rho_vec, settings, polishing);

    case OSQP_INDIRECT_SOLVER:
        return init_linsys_mklcg((mklcg_solver **)s, P, A, rho_vec, settings, polishing);
    }
}
