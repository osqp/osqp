#include "cuda_pcg_interface.h"

#include "glob_opts.h"


c_int init_linsys_solver_cudapcg(cudapcg_solver    **sp,
                                 const OSQPMatrix   *P,
                                 const OSQPMatrix   *A,
                                 const OSQPVectorf  *rho_vec,
                                 OSQPSettings       *settings,
                                 c_float            *scaled_pri_res,
                                 c_float            *scaled_dua_res,
                                 c_int               polish) {

  /* Allocate linsys solver structure */
  cudapcg_solver *s = c_calloc(1, sizeof(cudapcg_solver));
  *sp = s;

  /* Assign type */
  s->type = CUDA_PCG_SOLVER;

  /* Link functions */
  s->free = &free_linsys_solver_cudapcg;

  return 0;
}

void free_linsys_solver_cudapcg(cudapcg_solver *s) {

  if (s) {
    c_free(s);
  }

}

