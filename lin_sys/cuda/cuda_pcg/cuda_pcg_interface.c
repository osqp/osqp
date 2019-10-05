#include "cuda_pcg_interface.h"


c_int init_linsys_solver_cudapcg(cudapcg_solver    **sp,
                                 const OSQPMatrix   *P,
                                 const OSQPMatrix   *A,
                                 const OSQPVectorf  *rho_vec,
                                 OSQPSettings       *settings,
                                 c_float            *scaled_pri_res,
                                 c_float            *scaled_dua_res,
                                 c_int               polish) {

  // Assign type
  s->type = CUDA_PCG_SOLVER;

  return 0;
}

