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

#ifndef CUDA_PCG_INTERFACE_H
#define CUDA_PCG_INTERFACE_H

#ifdef __cplusplus
extern "C" {
#endif

#include "osqp.h"
#include "types.h"                /* OSQPMatrix and OSQPVector[fi] types */
#include "algebra_types.h"        /* csr type */


/**
 * CUDA PCG solver structure
 */
typedef struct cudapcg_solver_ {

  enum linsys_solver_type type;

  /**
   * @name Functions
   * @{
   */
  c_int (*solve)(struct cudapcg_solver_ *self,
                 OSQPVectorf            *b,
                 c_int                   admm_iter);

  void (*update_settings)(struct cudapcg_solver_  *self,
                          const OSQPSettings      *settings);

  void (*warm_start)(struct cudapcg_solver_  *self,
                     const OSQPVectorf       *x);

  void (*free)(struct cudapcg_solver_ *self);

  c_int (*update_matrices)(struct cudapcg_solver_ *self,
                           const OSQPMatrix       *P,
                           const OSQPMatrix       *A);

  c_int (*update_rho_vec)(struct cudapcg_solver_  *self,
                          const OSQPVectorf       *rho_vec,
                          c_float                  rho_sc);

  /* threads count */
  c_int nthreads;

  /* Dimensions */
  c_int n;                  ///<  dimension of the linear system
  c_int m;                  ///<  number of rows in A

  /* States */
  c_int polishing;
  c_int zero_pcg_iters;     ///<  state that counts zero PCG iterations

  /* Settings */
  c_int max_iter;
  
  /* Residual tolerance strategy parameters */
  c_int    reduction_threshold;
  c_float  tol_fraction;
  c_float  reduction_factor;
  c_float  eps_prev;
  c_float *scaled_prim_res;
  c_float *scaled_dual_res;

  /* ADMM settings and pointers to problem data */
  c_float  h_rho;
  c_float  h_sigma;
  csr     *A;
  csr     *At;
  csr     *P;
  c_int   *d_P_diag_ind;
  c_float *d_rho_vec;

  /* PCG iterates */
  c_float *d_x;             ///<  current iterate solution
  c_float *d_p;             ///<  current search direction
  c_float *d_Kp;            ///<  holds K*p
  c_float *d_y;             ///<  solution of the preconditioner r = M*y
  c_float *d_r;             ///<  residual r = K*x - b
  c_float *d_rhs;           ///<  right-hand side of Kx = b
  c_float *d_z;             ///<  holds z = A*x for computing A'*z = A'*(A*x);

  /* Pointer to page-locked host memory */
  c_float *h_r_norm;

  /* PCG scalar values (in device memory) */
  c_float *d_r_norm;
  c_float *rTy;
  c_float *rTy_prev;
  c_float *alpha;
  c_float *beta;
  c_float *pKp;
  c_float *D_MINUS_ONE;     ///<  constant -1.0 in device memory
  c_float *d_sigma;

  /* PCG preconditioner */
  c_float *d_P_diag_val;
  c_float *d_AtA_diag_val;
  c_float *d_AtRA_diag_val;
  c_float *d_diag_precond;
  c_float *d_diag_precond_inv;

  /* Function pointer to handle different vector norms */
  void (*vector_norm)(const c_float *d_x,
                      c_int          n,
                      c_float       *res);

} cudapcg_solver;



c_int init_linsys_solver_cudapcg(cudapcg_solver    **sp,
                                 const OSQPMatrix   *P,
                                 const OSQPMatrix   *A,
                                 const OSQPVectorf  *rho_vec,
                                 const OSQPSettings *settings,
                                 c_float            *scaled_prim_res,
                                 c_float            *scaled_dual_res,
                                 c_int               polishing);


c_int solve_linsys_cudapcg(cudapcg_solver *s,
                           OSQPVectorf    *b,
                           c_int           admm_iter);

void update_settings_linsys_solver_cudapcg(cudapcg_solver     *s,
                                           const OSQPSettings *settings);

void warm_start_linsys_solver_cudapcg(cudapcg_solver    *s,
                                      const OSQPVectorf *x);

void free_linsys_solver_cudapcg(cudapcg_solver *s);

c_int update_linsys_solver_matrices_cudapcg(cudapcg_solver   *s,
                                            const OSQPMatrix *P,
                                            const OSQPMatrix *A);

c_int update_linsys_solver_rho_vec_cudapcg(cudapcg_solver    *s,
                                           const OSQPVectorf *rho_vec,
                                           c_float            rho_sc);


#ifdef __cplusplus
}
#endif

#endif /* ifndef OSQP_API_TYPES_H */
