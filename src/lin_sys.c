#include "lin_sys.h"


const char *LINSYS_SOLVER_NAME[] = {
  "direct", "indirect"
};


#ifdef ALGEBRA_DEFAULT
# include "qdldl_interface.h"
#endif

#ifdef ALGEBRA_MKL
  # include "pardiso_interface.h"
  # include "mkl-cg_interface.h"
#endif

#ifdef ALGEBRA_CUDA
  # include "cuda_pcg_interface.h"
#endif


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
