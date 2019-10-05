#include "cuda_pcg_interface.h"

#include "cuda_malloc.h"

#include "glob_opts.h"


c_int init_linsys_solver_cudapcg(cudapcg_solver    **sp,
                                 const OSQPMatrix   *P,
                                 const OSQPMatrix   *A,
                                 const OSQPVectorf  *rho_vec,
                                 OSQPSettings       *settings,
                                 c_float            *scaled_pri_res,
                                 c_float            *scaled_dua_res,
                                 c_int               polish) {

  c_int n, m;
  c_float H_MINUS_ONE = -1.0;

  /* Allocate linsys solver structure */
  cudapcg_solver *s = c_calloc(1, sizeof(cudapcg_solver));
  *sp = s;

  /* Assign type and the number of threads */
  s->type     = CUDA_PCG_SOLVER;
  s->nthreads = 1;

  /* Problem dimensions */
  n = OSQPMatrix_get_n(P);
  m = OSQPMatrix_get_m(A);
  s->n = n;
  s->m = m;

  /* PCG states */
  s->polish = polish;
  s->zero_pcg_iters = 0;

  /* Default norm and tolerance strategy */
  s->eps_strategy   = RESIDUAL_STRATEGY;
  s->norm           = CUDA_PCG_NORM;
  s->precondition   = CUDA_PCG_PRECONDITION;
  s->warm_start_pcg = CUDA_PCG_WARM_START;
  s->max_iter       = CUDA_PCG_MAX_ITER;

  /* Tolerance strategy parameters */
  s->start_tol           = CUDA_PCG_START_TOL;
  s->dec_rate            = CUDA_PCG_DECAY_RATE;
  s->reduction_threshold = CUDA_PCG_REDUCTION_THRESHOLD;
  s->reduction_factor    = CUDA_PCG_REDUCTION_FACTOR;
  s->scaled_pri_res      = scaled_pri_res;
  s->scaled_dua_res      = scaled_dua_res;

  /* Set pointers to problem data and ADMM settings */
  s->A            = A->S;
  s->At           = A->At;
  s->P            = P->S;
  s->d_P_diag_ind = P->d_P_diag_ind;
  if (rho_vec)
    s->d_rho_vec  = rho_vec->d_val;
  if (!polish) {
    s->h_sigma = &settings->sigma;
    s->h_rho   = &settings->rho;
  }
  else {
    s->h_sigma = &settings->delta;
    s->h_rho   = (c_float*) c_malloc(sizeof(c_float));
    *s->h_rho  = 1. / settings->delta;
  }

  /* Allocate PCG iterates */
  cuda_calloc((void **) &s->d_x,   n * sizeof(c_float));    /* Set d_x to zero */
  cuda_malloc((void **) &s->d_p,   n * sizeof(c_float));
  cuda_malloc((void **) &s->d_Kp,  n * sizeof(c_float));
  cuda_malloc((void **) &s->d_y,   n * sizeof(c_float));
  cuda_malloc((void **) &s->d_r,   n * sizeof(c_float));
  cuda_malloc((void **) &s->d_rhs, n * sizeof(c_float));
  if (m != 0) cuda_malloc((void **) &s->d_z, m * sizeof(c_float));

  /* Allocate scalar in host memory that is page-locked and accessible to device */
  cuda_malloc_host((void **) &s->h_r_norm, sizeof(c_float));

  /* Link functions */
  s->free = &free_linsys_solver_cudapcg;

  return 0;
}

void free_linsys_solver_cudapcg(cudapcg_solver *s) {

  if (s) {
    if (s->polish) c_free(s->h_rho);

    /* PCG iterates */
    cuda_free((void **) &s->d_x);
    cuda_free((void **) &s->d_p);
    cuda_free((void **) &s->d_Kp);
    cuda_free((void **) &s->d_y);
    cuda_free((void **) &s->d_r);
    cuda_free((void **) &s->d_rhs);
    cuda_free((void **) &s->d_z);

    /* Free page-locked host memory */
    cuda_free_host((void **) s->h_r_norm);

    c_free(s);
  }

}

