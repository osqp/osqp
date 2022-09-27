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


#include <cusparse.h>
#include "osqp.h"
#include "types.h"                /* OSQPMatrix and OSQPVector[fi] types */
#include "algebra_types.h"        /* csr type */


/**
 * CUDA PCG solver structure
 */
typedef struct cudapcg_solver_ {

  enum osqp_linsys_solver_type type;

  /**
   * @name Functions
   * @{
   */
  const char* (*name)(struct cudapcg_solver_* self);

  OSQPInt (*solve)(struct cudapcg_solver_* self,
                          OSQPVectorf*     b,
                          OSQPInt          admm_iter);

  void (*update_settings)(struct cudapcg_solver_* self,
                          const  OSQPSettings*    settings);

  void (*warm_start)(struct cudapcg_solver_* self,
                     const  OSQPVectorf*     x);

  OSQPInt (*adjoint_derivative)(struct cudapcg_solver_* self);

  void (*free)(struct cudapcg_solver_* self);

  OSQPInt (*update_matrices)(struct cudapcg_solver_* self,
                             const  OSQPMatrix*      P,
                             const  OSQPInt*         Px_new_idx,
                                    OSQPInt          P_new_n,
                             const  OSQPMatrix*      A,
                             const  OSQPInt*         Ax_new_idx,
                                    OSQPInt          A_new_n);

  OSQPInt (*update_rho_vec)(struct cudapcg_solver_* self,
                            const  OSQPVectorf*     rho_vec,
                                   OSQPFloat        rho_sc);

  /* threads count */
  OSQPInt nthreads;

  /* Dimensions */
  OSQPInt n;                  ///<  dimension of the linear system
  OSQPInt m;                  ///<  number of rows in A

  /* States */
  OSQPInt polishing;
  OSQPInt zero_pcg_iters;     ///<  state that counts zero PCG iterations

  /* Settings */
  OSQPInt           max_iter;
  osqp_precond_type precond_type;

  /* Residual tolerance strategy parameters */
  OSQPInt    reduction_threshold;
  OSQPFloat  tol_fraction;
  OSQPFloat  reduction_factor;
  OSQPFloat  eps_prev;
  OSQPFloat* scaled_prim_res;
  OSQPFloat* scaled_dual_res;

  /* ADMM settings and pointers to problem data */
  OSQPFloat  h_rho;
  OSQPFloat  h_sigma;
  csr*       A;
  csr*       At;
  csr*       P;
  OSQPInt*   d_P_diag_ind;
  OSQPFloat* d_rho_vec;

  /* PCG iterates: raw vectors */
  OSQPFloat* d_x;             ///<  current iterate solution
  OSQPFloat* d_p;             ///<  current search direction
  OSQPFloat* d_Kp;            ///<  holds K*p
  OSQPFloat* d_y;             ///<  solution of the preconditioner r = M*y
  OSQPFloat* d_r;             ///<  residual r = K*x - b
  OSQPFloat* d_rhs;           ///<  right-hand side of Kx = b
  OSQPFloat* d_z;             ///<  holds z = A*x for computing A'*z = A'*(A*x);

  /* PCG iterates: dense vector desciptors */
  cusparseDnVecDescr_t vecx;
  cusparseDnVecDescr_t vecp;
  cusparseDnVecDescr_t vecKp;
  cusparseDnVecDescr_t vecr;
  cusparseDnVecDescr_t vecrhs;
  cusparseDnVecDescr_t vecz;

  /* Pointer to page-locked host memory */
  OSQPFloat* h_r_norm;

  /* PCG scalar values (in device memory) */
  OSQPFloat* d_r_norm;
  OSQPFloat* rTy;
  OSQPFloat* rTy_prev;
  OSQPFloat* alpha;
  OSQPFloat* beta;
  OSQPFloat* pKp;
  OSQPFloat* D_MINUS_ONE;     ///<  constant -1.0 in device memory
  OSQPFloat* d_sigma;

  /* PCG preconditioner */
  OSQPFloat* d_P_diag_val;
  OSQPFloat* d_AtA_diag_val;
  OSQPFloat* d_AtRA_diag_val;
  OSQPFloat* d_diag_precond;
  OSQPFloat* d_diag_precond_inv;

  /* Function pointer to handle different vector norms */
  void (*vector_norm)(const OSQPFloat* d_x,
                            OSQPInt    n,
                            OSQPFloat* res);

} cudapcg_solver;


#ifdef __cplusplus
extern "C" {
#endif

OSQPInt init_linsys_solver_cudapcg(cudapcg_solver**    sp,
                                   const OSQPMatrix*   P,
                                   const OSQPMatrix*   A,
                                   const OSQPVectorf*  rho_vec,
                                   const OSQPSettings* settings,
                                   OSQPFloat*          scaled_prim_res,
                                   OSQPFloat*          scaled_dual_res,
                                   OSQPInt             polishing);

#ifdef __cplusplus
}
#endif

/**
 * Get the user-friendly name of the PCG solver.
 * @return The user-friendly name
 */
const char* name_cudapcg(cudapcg_solver* s);


OSQPInt solve_linsys_cudapcg(cudapcg_solver* s,
                             OSQPVectorf*    b,
                             OSQPInt         admm_iter);

void update_settings_linsys_solver_cudapcg(cudapcg_solver*     s,
                                           const OSQPSettings* settings);

void warm_start_linsys_solver_cudapcg(cudapcg_solver*    s,
                                      const OSQPVectorf* x);

void free_linsys_solver_cudapcg(cudapcg_solver* s);

OSQPInt update_linsys_solver_matrices_cudapcg(cudapcg_solver*   s,
                                              const OSQPMatrix* P,
                                              const OSQPInt*    Px_new_idx,
                                              OSQPInt           P_new_n,
                                              const OSQPMatrix* A,
                                              const OSQPInt*    Ax_new_idx,
                                              OSQPInt           A_new_n);

OSQPInt update_linsys_solver_rho_vec_cudapcg(cudapcg_solver*    s,
                                             const OSQPVectorf* rho_vec,
                                             OSQPFloat          rho_sc);


#endif /* ifndef OSQP_API_TYPES_H */
