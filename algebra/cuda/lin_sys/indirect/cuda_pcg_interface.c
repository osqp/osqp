/**
 *  Copyright (c) 2019-2021 ETH Zurich, Automatic Control Lab,
 *  Michel Schubiger, Goran Banjac.
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#include "cuda_pcg_interface.h"
#include "cuda_pcg.h"

#include "cuda_lin_alg.h"
#include "cuda_malloc.h"

#include "glob_opts.h"


/*******************************************************************************
 *                           Private Functions                                 *
 *******************************************************************************/

static c_float compute_tolerance(cudapcg_solver *s,
                                 c_int           admm_iter) {

  c_float eps, rhs_norm;

  /* Compute the norm of RHS of the linear system */
  cuda_vec_norm_inf(s->d_rhs, s->n, &rhs_norm);

  if (s->polishing) return c_max(rhs_norm * OSQP_CG_POLISH_TOL, OSQP_CG_TOL_MIN);

  if (admm_iter == 1) {
    // Set reduction_factor to its default value
    s->reduction_factor = s->tol_fraction;

    // In case rhs = 0.0 we don't want to set eps_prev to 0.0
    if (rhs_norm < OSQP_CG_TOL_MIN) s->eps_prev = 1.0;
    else s->eps_prev = rhs_norm * s->reduction_factor;

    // Return early since scaled_prim_res and scaled_dual_res are meaningless before the first ADMM iteration
    return s->eps_prev;
  }

  if (s->zero_pcg_iters >= s->reduction_threshold) {
    s->reduction_factor /= 2;
    s->zero_pcg_iters = 0;
  }

  eps = s->reduction_factor * sqrt((*s->scaled_prim_res) * (*s->scaled_dual_res));
  eps = c_max(c_min(eps, s->eps_prev), OSQP_CG_TOL_MIN);
  s->eps_prev = eps;

  return eps;
}

/* d_rhs = d_b1 + A' * rho * d_b2 */
static void compute_rhs(cudapcg_solver *s,
                        c_float        *d_b) {

  c_int n = s->n;
  c_int m = s->m;

  /* d_rhs = d_b1 */
  cuda_vec_copy_d2d(s->d_rhs, d_b, n);

  if (m == 0) return;

  /* d_z = d_b2 */
  cuda_vec_copy_d2d(s->d_z, d_b + n, m);

  if (!s->d_rho_vec) {
    /* d_z *= rho */
    cuda_vec_mult_sc(s->d_z, *s->h_rho, m);
  }
  else {
    /* d_z = diag(d_rho_vec) * d_z */
    cuda_vec_ew_prod(s->d_z, s->d_z, s->d_rho_vec, m);
  }

  /* d_rhs += A' * d_z */
  cuda_mat_Axpy(s->At, s->d_z, s->d_rhs, 1.0, 1.0);
}


/*******************************************************************************
 *                              API Functions                                  *
 *******************************************************************************/

c_int init_linsys_solver_cudapcg(cudapcg_solver    **sp,
                                 const OSQPMatrix   *P,
                                 const OSQPMatrix   *A,
                                 const OSQPVectorf  *rho_vec,
                                 const OSQPSettings *settings,
                                 c_float            *scaled_prim_res,
                                 c_float            *scaled_dual_res,
                                 c_int               polishing) {

  c_int n, m;
  c_float H_MINUS_ONE = -1.0;

  /* Allocate linsys solver structure */
  cudapcg_solver *s = c_calloc(1, sizeof(cudapcg_solver));
  *sp = s;

  /* Assign type and the number of threads */
  s->type     = INDIRECT_SOLVER;
  s->nthreads = 1;

  /* Problem dimensions */
  n = OSQPMatrix_get_n(P);
  m = OSQPMatrix_get_m(A);
  s->n = n;
  s->m = m;

  /* PCG states */
  s->polishing = polishing;
  s->zero_pcg_iters = 0;

  /* Maximum number of PCG iterations */
  s->max_iter = settings->cg_max_iter;

  /* Tolerance strategy parameters */
  s->reduction_threshold = settings->cg_tol_reduction;
  s->tol_fraction        = settings->cg_tol_fraction;
  s->reduction_factor    = settings->cg_tol_fraction;
  s->scaled_prim_res     = scaled_prim_res;
  s->scaled_dual_res     = scaled_dual_res;

  /* Set pointers to problem data and ADMM settings */
  s->A            = A->S;
  s->At           = A->At;
  s->P            = P->S;
  s->d_P_diag_ind = P->d_P_diag_ind;
  if (rho_vec)
    s->d_rho_vec  = rho_vec->d_val;
  if (!polishing) {
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

  /* Allocate device-side scalar values. This way scalars are packed in device memory */
  cuda_malloc((void **) &s->d_r_norm, 8 * sizeof(c_float));
  s->rTy         = s->d_r_norm + 1;
  s->rTy_prev    = s->d_r_norm + 2;
  s->alpha       = s->d_r_norm + 3;
  s->beta        = s->d_r_norm + 4;
  s->pKp         = s->d_r_norm + 5;
  s->D_MINUS_ONE = s->d_r_norm + 6;
  s->d_sigma     = s->d_r_norm + 7;
  cuda_vec_copy_h2d(s->D_MINUS_ONE, &H_MINUS_ONE, 1);
  cuda_vec_copy_h2d(s->d_sigma,     s->h_sigma,   1);

  /* Allocate memory for PCG preconditioning */
  cuda_malloc((void **) &s->d_P_diag_val,       n * sizeof(c_float));
  cuda_malloc((void **) &s->d_AtRA_diag_val,    n * sizeof(c_float));
  cuda_malloc((void **) &s->d_diag_precond,     n * sizeof(c_float));
  cuda_malloc((void **) &s->d_diag_precond_inv, n * sizeof(c_float));
  if (!s->d_rho_vec) cuda_malloc((void **) &s->d_AtA_diag_val, n * sizeof(c_float));

  /* Link functions */
  s->solve           = &solve_linsys_cudapcg;
  s->warm_start      = &warm_start_linsys_solver_cudapcg;
  s->free            = &free_linsys_solver_cudapcg;
  s->update_matrices = &update_linsys_solver_matrices_cudapcg;
  s->update_rho_vec  = &update_linsys_solver_rho_vec_cudapcg;
  s->update_settings = &update_settings_linsys_solver_cudapcg;

  /* Initialize PCG preconditioner */
  cuda_pcg_update_precond(s, 1, 1, 1);

  /* No error */
  return 0;
}


c_int solve_linsys_cudapcg(cudapcg_solver *s,
                           OSQPVectorf    *b,
                           c_int           admm_iter) {

  c_int   pcg_iters;
  c_float eps;

  /* Compute the RHS of the reduced KKT system and store it in s->d_rhs */
  compute_rhs(s, b->d_val);

  /* Compute the required solution precision */
  eps = compute_tolerance(s, admm_iter);

  /* Solve the linear system with PCG */
  pcg_iters = cuda_pcg_alg(s, eps, s->max_iter);

  /* Copy the first part of the solution to b->d_val */
  cuda_vec_copy_d2d(b->d_val, s->d_x, s->n);

  if (!s->polishing) {
    /* Compute d_z = A * d_x */
    if (s->m) cuda_mat_Axpy(s->A, s->d_x, b->d_val + s->n, 1.0, 0.0);
  }
  else {
    // yred = (A * d_x - b) / delta
    cuda_mat_Axpy(s->A, s->d_x, b->d_val + s->n, 1.0, -1.0);
    cuda_vec_mult_sc(b->d_val + s->n, *s->h_rho, s->m);
  }

  // Number of consecutive ADMM iterations with zero PCG iterations
  if (pcg_iters == 0) s->zero_pcg_iters++;
  else                s->zero_pcg_iters = 0;

  return 0;
}


void update_settings_linsys_solver_cudapcg(cudapcg_solver     *s,
                                           const OSQPSettings *settings) {

  s->max_iter            = settings->cg_max_iter;
  s->reduction_threshold = settings->cg_tol_reduction;
  s->tol_fraction        = settings->cg_tol_fraction;
}


void warm_start_linsys_solver_cudapcg(cudapcg_solver    *s,
                                      const OSQPVectorf *x) {

  cuda_vec_copy_d2d(s->d_x, x->d_val, x->length);
}


void free_linsys_solver_cudapcg(cudapcg_solver *s) {

  if (s) {
    if (s->polishing) c_free(s->h_rho);

    /* PCG iterates */
    cuda_free((void **) &s->d_x);
    cuda_free((void **) &s->d_p);
    cuda_free((void **) &s->d_Kp);
    cuda_free((void **) &s->d_y);
    cuda_free((void **) &s->d_r);
    cuda_free((void **) &s->d_rhs);
    cuda_free((void **) &s->d_z);

    /* Free page-locked host memory */
    cuda_free_host((void **) &s->h_r_norm);

    /* Device-side scalar values */
    cuda_free((void **) &s->d_r_norm);

    /* PCG preconditioner */
    cuda_free((void **) &s->d_P_diag_val);
    cuda_free((void **) &s->d_AtA_diag_val);
    cuda_free((void **) &s->d_AtRA_diag_val);
    cuda_free((void **) &s->d_diag_precond);
    cuda_free((void **) &s->d_diag_precond_inv);

    c_free(s);
  }
}


c_int update_linsys_solver_matrices_cudapcg(cudapcg_solver   *s,
                                            const OSQPMatrix *P,
                                            const OSQPMatrix *A) {

  cuda_pcg_update_precond(s, 1, 1, 0);
  return 0;
}


c_int update_linsys_solver_rho_vec_cudapcg(cudapcg_solver    *s,
                                           const OSQPVectorf *rho_vec,
                                           c_float            rho_sc) {

  cuda_pcg_update_precond(s, 0, 0, 1);
  return 0;
}

