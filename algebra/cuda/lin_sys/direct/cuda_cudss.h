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

#ifndef CUDA_CUDSS_H_
#define CUDA_CUDSS_H_

#include "cudss.h"

#include "osqp.h"
#include "types.h"                /* OSQPMatrix and OSQPVector[fi] types */
#include "algebra_types.h"        /* csr type */

/**
 * CUDA cuDSS solver structure
 */
typedef struct cudss_solver_ {

  enum osqp_linsys_solver_type type;

  /**
   * @name Functions
   * @{
   */
  const char* (*name)(struct cudss_solver_* self);

  OSQPInt (*solve)(struct cudss_solver_* self,
                          OSQPVectorf*     b,
                          OSQPInt          admm_iter);

  void (*update_settings)(struct cudss_solver_* self,
                          const  OSQPSettings*    settings);

  void (*warm_start)(struct cudss_solver_* self,
                     const  OSQPVectorf*     x);

  OSQPInt (*adjoint_derivative)(struct cudss_solver_* self);

  void (*free)(struct cudss_solver_* self);

  OSQPInt (*update_matrices)(struct cudss_solver_* self,
                             const  OSQPMatrix*      P,
                             const  OSQPInt*         Px_new_idx,
                                    OSQPInt          P_new_n,
                             const  OSQPMatrix*      A,
                             const  OSQPInt*         Ax_new_idx,
                                    OSQPInt          A_new_n);

  OSQPInt (*update_rho_vec)(struct cudss_solver_* self,
                            const  OSQPVectorf*     rho_vec,
                                   OSQPFloat        rho_sc);

  OSQPInt    nthreads;             ///< Number of threads (we don't use this, so it is 1 all the time)
  /*
   * }
   */
  cudssHandle_t lib_handle;        ///< Handle to the cuDSS library
  cudssConfig_t config_handle;     ///< Handle to cuDSS config
  cudssData_t   data_handle;       ///< cuDSS solver data

  /* Dimensions */
  OSQPInt n;                       ///<  dimension of the linear system
  OSQPInt m;                       ///<  number of rows in A

  /* Parameters */
  OSQPFloat  sigma;                ///< Signal parameter
  OSQPInt    rho_is_vec;           ///< Rho is a vector?
  OSQPFloat  rho_inv;              ///< Scalar rho inverse value (used when rho_inv_vec is null)
  OSQPFloat* rho_inv_vec;          ///< Vector rho inverse

  /* States */
  OSQPInt       polishing;         ///< Are we in polishing mode?
  cudssStatus_t solveStatus;       ///< Status of last cuDSS operation

  /* Various matrices */
  OSQPCscMatrix* h_kkt_csr;        ///< Host KKT matrix in CSR form
  cudssMatrix_t  d_cudss_kkt;      ///< Device KKT matrix for cuDSS
  cudssMatrix_t  d_cudss_x;        ///< Device x vector
  cudssMatrix_t  d_cudss_b;        ///< Device RNS (b) vector

  /* Device vectors during solve*/
  OSQPInt*   d_kkt_p;              ///< Device vector for KKT matrix column pointers
  OSQPInt*   d_kkt_i;              ///< Device vector for KKT matrix row indices
  OSQPFloat* d_kkt_x;              ///< Device vector for KKT matrix matrix values
  OSQPFloat* d_x;                  ///< Device vector for x vector
  OSQPFloat* d_b;                  ///< Device vector for RHS (b) vector

  /* Device vectors used during KKT update */
  OSQPInt* d_PtoKKT;                 ///< Indices of P matrix in assembled KKT matrix
  OSQPInt* d_AtoKKT;                 ///< Indices of A matrix in assembled KKT matrix
  OSQPInt* d_rhotoKKT;               ///< Indices of rho places in assembled KKT matrix
} cudss_solver;


#ifdef __cplusplus
extern "C" {
#endif

OSQPInt init_linsys_solver_cudss(cudss_solver**    sp,
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
 * Get the user-friendly name of the solver.
 * @return The user-friendly name
 */
const char* name_cudss(cudss_solver* s);


OSQPInt solve_linsys_cudss(cudss_solver* s,
                             OSQPVectorf*    b,
                             OSQPInt         admm_iter);

void update_settings_linsys_solver_cudss(cudss_solver*     s,
                                           const OSQPSettings* settings);

void warm_start_linsys_solver_cudss(cudss_solver*    s,
                                      const OSQPVectorf* x);

void free_linsys_solver_cudss(cudss_solver* s);

OSQPInt update_linsys_solver_matrices_cudss(cudss_solver*   s,
                                              const OSQPMatrix* P,
                                              const OSQPInt*    Px_new_idx,
                                              OSQPInt           P_new_n,
                                              const OSQPMatrix* A,
                                              const OSQPInt*    Ax_new_idx,
                                              OSQPInt           A_new_n);

OSQPInt update_linsys_solver_rho_vec_cudss(cudss_solver*    s,
                                             const OSQPVectorf* rho_vec,
                                             OSQPFloat          rho_sc);


#endif /* ifndef CUDA_CUDSS_H_ */
