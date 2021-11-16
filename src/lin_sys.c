#include "lin_sys.h"


const char *LINSYS_SOLVER_NAME[] = {
  "direct", "indirect"
};


#ifdef ALGEBRA_CUDA

# include "cuda_pcg_interface.h"

#else /* ifdef ALGEBRA_CUDA */

#ifdef ALGEBRA_MKL
# include "pardiso_interface.h"
# include "mkl-cg_interface.h"
#else  /* ifdef ALGEBRA_MKL */
# include "qdldl_interface.h"
#endif

#endif /* ifdef ALGEBRA_CUDA */


// Load linear system solver shared library
c_int load_linsys_solver(enum linsys_solver_type linsys_solver) {

  switch (linsys_solver) {

#ifdef ALGEBRA_CUDA

  default:
    /* CUDA libraries have already been loaded by osqp_algebra_init_libs() */
    return 0;

#else /* ifdef ALGEBRA_CUDA */

#ifdef ALGEBRA_MKL
  case DIRECT_SOLVER:
    // Load Pardiso library
    // return lh_load_pardiso(OSQP_NULL);
    return 0;

  case INDIRECT_SOLVER:
    // statically linked for now
    return 0;
#else
  // We do not load QDLDL solver. We have the source.
  default:
    return 0;
#endif
#endif /* ifdef ALGEBRA_CUDA */
  }
}

// Unload linear system solver shared library
c_int unload_linsys_solver(enum linsys_solver_type linsys_solver) {
  switch (linsys_solver) {

#ifdef ALGEBRA_CUDA

  case DIRECT_SOLVER:
    /* CUDA libraries have already been unloaded by osqp_algebra_free_libs() */
    return 0;

  default:
    return 0;

#else /* ifdef ALGEBRA_CUDA */

#ifdef ALGEBRA_MKL
  default:
    return 0;
#endif
#endif /* ifdef ALGEBRA_CUDA */
  }
}

// Initialize linear system solver structure
// NB: Only the upper triangular part of P is filled
c_int init_linsys_solver(LinSysSolver      **s,
                         const OSQPMatrix   *P,
                         const OSQPMatrix   *A,
                         const OSQPVectorf  *rho_vec,
                         const OSQPSettings *settings,
                         c_float            *scaled_prim_res,
                         c_float            *scaled_dual_res,
                         c_int               polishing) {

  switch (settings->linsys_solver) {

#ifdef ALGEBRA_CUDA

  default:
    return init_linsys_solver_cudapcg((cudapcg_solver **)s, P, A, rho_vec, settings, scaled_prim_res, scaled_dual_res, polishing);

#else /* ifdef ALGEBRA_CUDA */

#ifdef ALGEBRA_MKL
  case DIRECT_SOLVER:
    return init_linsys_solver_pardiso((pardiso_solver **)s, P, A, rho_vec, settings, polishing);

  case INDIRECT_SOLVER:
    return init_linsys_mklcg((mklcg_solver **)s, P, A, rho_vec, settings, polishing);
#else
  default:
    return init_linsys_solver_qdldl((qdldl_solver **)s, P, A, rho_vec, settings, polishing);
#endif

#endif /* ifdef ALGEBRA_CUDA */
  }
}
