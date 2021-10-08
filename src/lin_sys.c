#include "lin_sys.h"


const char *LINSYS_SOLVER_NAME[] = {
  "qdldl", "mkl pardiso", "cuda pcg"
};


#ifdef CUDA_SUPPORT

# include "cuda_pcg_interface.h"

#else /* ifdef CUDA_SUPPORT */

# include "qdldl_interface.h"

#ifdef ENABLE_MKL_PARDISO
# include "pardiso_interface.h"
# include "pardiso_loader.h"
#endif

#endif /* ifdef CUDA_SUPPORT */


// Load linear system solver shared library
c_int load_linsys_solver(enum linsys_solver_type linsys_solver) {

  switch (linsys_solver) {

#ifdef CUDA_SUPPORT

  case CUDA_PCG_SOLVER:
    /* CUDA libraries have already been loaded by osqp_algebra_init_libs() */
    return 0;

  default: /* CUDA_PCG_SOLVER */
    return 0;

#else /* ifdef CUDA_SUPPORT */

  case QDLDL_SOLVER:
    // We do not load QDLDL solver. We have the source.
    return 0;

# ifdef ENABLE_MKL_PARDISO
  case MKL_PARDISO_SOLVER:
    // Load Pardiso library
    return lh_load_pardiso(OSQP_NULL);
# endif

  default: // QDLDL
    return 0;

#endif /* ifdef CUDA_SUPPORT */
  }
}

// Unload linear system solver shared library
c_int unload_linsys_solver(enum linsys_solver_type linsys_solver) {
  switch (linsys_solver) {

#ifdef CUDA_SUPPORT

  case CUDA_PCG_SOLVER:
    /* CUDA libraries have already been unloaded by osqp_algebra_free_libs() */
    return 0;

  default: /* CUDA_PCG_SOLVER */
    return 0;

#else /* ifdef CUDA_SUPPORT */

  case QDLDL_SOLVER:

    // We do not load QDLDL solver. We have the source.
    return 0;

# ifdef ENABLE_MKL_PARDISO
  case MKL_PARDISO_SOLVER:
    // Unload Pardiso library
    return lh_unload_pardiso();
# endif

  default: //  QDLDL
    return 0;

#endif /* ifdef CUDA_SUPPORT */
  }
}

// Initialize linear system solver structure
// NB: Only the upper triangular part of P is filled
c_int init_linsys_solver(LinSysSolver      **s,
                         const OSQPMatrix   *P,
                         const OSQPMatrix   *A,
                         const OSQPVectorf  *rho_vec,
                         OSQPSettings       *settings,
                         c_float            *scaled_prim_res,
                         c_float            *scaled_dual_res,
                         c_int               polishing) {

  switch (settings->linsys_solver) {

#ifdef CUDA_SUPPORT

  case CUDA_PCG_SOLVER:
    return init_linsys_solver_cudapcg((cudapcg_solver **)s, P, A, rho_vec, settings, scaled_prim_res, scaled_dual_res, polishing);

  default: /* CUDA_PCG_SOLVER */
    return init_linsys_solver_cudapcg((cudapcg_solver **)s, P, A, rho_vec, settings, scaled_prim_res, scaled_dual_res, polishing);

#else /* ifdef CUDA_SUPPORT */

  case QDLDL_SOLVER:
    return init_linsys_solver_qdldl((qdldl_solver **)s, P, A, rho_vec, settings, polishing);

# ifdef ENABLE_MKL_PARDISO
  case MKL_PARDISO_SOLVER:
    return init_linsys_solver_pardiso((pardiso_solver **)s, P, A, rho_vec, settings, polishing);
# endif

  default: // QDLDL
    return init_linsys_solver_qdldl((qdldl_solver **)s, P, A, rho_vec, settings, polishing);

#endif /* ifdef CUDA_SUPPORT */
  }
}
