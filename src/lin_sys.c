#include "lin_sys.h"

#include "qdldl_interface.h" // Include only this solver in the same directory

const char *LINSYS_SOLVER_NAME[] = {
  "qdldl", "mkl pardiso"
};

#ifdef ENABLE_MKL_PARDISO
# include "pardiso_interface.h"
# include "pardiso_loader.h"
#endif /* ifdef ENABLE_MKL_PARDISO */

// Load linear system solver shared library
c_int load_linsys_solver(enum linsys_solver_type linsys_solver) {
  switch (linsys_solver) {
  case QDLDL_SOLVER:

    // We do not load  QDLDL solver. We have the source.
    return 0;

# ifdef ENABLE_MKL_PARDISO
  case MKL_PARDISO_SOLVER:

    // Load Pardiso library
    return lh_load_pardiso(OSQP_NULL);

# endif /* ifdef ENABLE_MKL_PARDISO */
  default: // QDLDL
    return 0;
  }
}

// Unload linear system solver shared library
c_int unload_linsys_solver(enum linsys_solver_type linsys_solver) {
  switch (linsys_solver) {
  case QDLDL_SOLVER:

    // We do not load QDLDL solver. We have the source.
    return 0;

# ifdef ENABLE_MKL_PARDISO
  case MKL_PARDISO_SOLVER:

    // Unload Pardiso library
    return lh_unload_pardiso();

# endif /* ifdef ENABLE_MKL_PARDISO */
  default: //  QDLDL
    return 0;
  }
}

// Initialize linear system solver structure
// NB: Only the upper triangular part of P is filled
c_int init_linsys_solver(LinSysSolver      **s,
                         const OSQPMatrix   *P,
                         const OSQPMatrix   *A,
                         const OSQPVectorf  *rho_vec,
                         OSQPSettings       *settings,
                         c_int               polish) {

  switch (settings->linsys_solver) {
  case QDLDL_SOLVER:
    return init_linsys_solver_qdldl((qdldl_solver **)s, P, A, rho_vec, settings, polish);

# ifdef ENABLE_MKL_PARDISO
  case MKL_PARDISO_SOLVER:
    return init_linsys_solver_pardiso((pardiso_solver **)s, P, A, rho_vec, settings, polish);

# endif /* ifdef ENABLE_MKL_PARDISO */
  default: // QDLDL
    return init_linsys_solver_qdldl((qdldl_solver **)s, P, A, rho_vec, settings, polish);
  }
}
