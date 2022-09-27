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

#include "cuda_pcg.h"
#include "csr_type.h"
#include "cuda_configure.h"
#include "cuda_handler.h"
#include "cuda_malloc.h"
#include "cuda_lin_alg.h"
#include "cuda_wrapper.h"
#include "helper_cuda.h"    /* --> checkCudaErrors */

extern CUDA_Handle_t *CUDA_handle;

/*******************************************************************************
 *                              GPU Kernels                                    *
 *******************************************************************************/

__global__ void scalar_division_kernel(OSQPFloat*       res,
                                       const OSQPFloat* num,
                                       const OSQPFloat* den) {

  *res = (*num) / (*den);
}


/*******************************************************************************
 *                            Private Functions                                *
 *******************************************************************************/

/*
 * y = (P + sigma*I + A'*R*A) * x
 */
static void mat_vec_prod(cudapcg_solver*             s,
                         OSQPFloat*                  d_y,
                         cusparseDnVecDescr_t        vecy,
                         const OSQPFloat*            d_x,
                         const cusparseDnVecDescr_t  vecx,
                         OSQPInt                     is_device) {

  OSQPFloat* sigma;
  OSQPFloat  H_ZERO = 0.0;
  OSQPFloat  H_ONE  = 1.0;
  OSQPInt n = s->n;
  OSQPInt m = s->m;

  csr* P  = s->P;
  csr* A  = s->A;
  csr* At = s->At;

  sigma = is_device ? s->d_sigma : &s->h_sigma;

  /* y = x */
  checkCudaErrors(cudaMemcpy(d_y, d_x, n * sizeof(OSQPFloat), cudaMemcpyDeviceToDevice));

  /* y *= sigma */
  checkCudaErrors(cublasTscal(CUDA_handle->cublasHandle, n, sigma, d_y, 1));

  /* y += P * x */
  checkCudaErrors(cusparseSpMV(
    CUDA_handle->cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
    &H_ONE, P->SpMatDescr, vecx, &H_ONE, vecy,
    CUDA_FLOAT, CUSPARSE_SPMV_ALG_DEFAULT, P->SpMatBuffer));

  if (m == 0) return;

  if (!s->d_rho_vec) {
    /* z = rho * A * x */
    checkCudaErrors(cusparseSpMV(
      CUDA_handle->cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
      &s->h_rho, A->SpMatDescr, vecx, &H_ZERO, s->vecz,
      CUDA_FLOAT, CUSPARSE_SPMV_ALG_DEFAULT, A->SpMatBuffer));
  }
  else {
    /* z = A * x */
    checkCudaErrors(cusparseSpMV(
      CUDA_handle->cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
      &H_ONE, A->SpMatDescr, vecx, &H_ZERO, s->vecz,
      CUDA_FLOAT, CUSPARSE_SPMV_ALG_DEFAULT, A->SpMatBuffer));

    /* z = diag(rho_vec) * z */
    cuda_vec_ew_prod(s->d_z, s->d_z, s->d_rho_vec, m);
  }

  /* y += A' * z */
  checkCudaErrors(cusparseSpMV(
    CUDA_handle->cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
    &H_ONE, At->SpMatDescr, s->vecz, &H_ONE, vecy,
    CUDA_FLOAT, CUSPARSE_SPMV_ALG_DEFAULT, At->SpMatBuffer));
}


/*******************************************************************************
 *                              API Functions                                  *
 *******************************************************************************/

OSQPInt cuda_pcg_alg(cudapcg_solver* s,
                     OSQPFloat       eps,
                     OSQPInt         max_iter) {

  OSQPFloat* tmp;

  OSQPInt   iter = 0;
  OSQPInt   n    = s->n;
  OSQPFloat H_MINUS_ONE = -1.0;

  if (!s->warm_start) {
    /* x = 0 */
    checkCudaErrors(cudaMemset(s->d_x, 0, n * sizeof(OSQPFloat)));
  }

  /* p = 0 */
  checkCudaErrors(cudaMemset(s->d_p, 0, n * sizeof(OSQPFloat)));

  /* r = K * x */
  mat_vec_prod(s, s->d_r, s->vecr, s->d_x, s->vecx, 0);

  /* r -= rhs */
  checkCudaErrors(cublasTaxpy(CUDA_handle->cublasHandle, n, &H_MINUS_ONE, s->d_rhs, 1, s->d_r, 1));

  /* h_r_norm = |r| */
  cuda_vec_norm_inf(s->d_r, n, s->h_r_norm);

  /* From here on cuBLAS is operating in device pointer mode */
  cublasSetPointerMode(CUDA_handle->cublasHandle, CUBLAS_POINTER_MODE_DEVICE);

  /* y = M \ r */
  cuda_vec_ew_prod(s->d_y, s->d_diag_precond_inv, s->d_r, n);

  /* p = -y */
  checkCudaErrors(cublasTaxpy(CUDA_handle->cublasHandle, n, s->D_MINUS_ONE, s->d_y, 1, s->d_p, 1));

  /* rTy = r' * y */
  checkCudaErrors(cublasTdot(CUDA_handle->cublasHandle, n, s->d_y, 1, s->d_r, 1, s->rTy));

  cudaDeviceSynchronize();

  /* Run the PCG algorithm */
  while ( *(s->h_r_norm) > eps && iter < max_iter ) {

    /* Kp = K * p */
    mat_vec_prod(s, s->d_Kp, s->vecKp, s->d_p, s->vecp, 1);

    /* pKp = p' * Kp */
    checkCudaErrors(cublasTdot(CUDA_handle->cublasHandle, n, s->d_p, 1, s->d_Kp, 1, s->pKp));

    /* alpha = rTy / pKp */
    scalar_division_kernel<<<1,1>>>(s->alpha, s->rTy, s->pKp);

    /* x += alpha * p */
    checkCudaErrors(cublasTaxpy(CUDA_handle->cublasHandle, n, s->alpha, s->d_p, 1, s->d_x, 1));

    /* r += alpha * Kp */
    checkCudaErrors(cublasTaxpy(CUDA_handle->cublasHandle, n, s->alpha, s->d_Kp, 1, s->d_r, 1));

    /* y = M \ r */
    cuda_vec_ew_prod(s->d_y, s->d_diag_precond_inv, s->d_r, n);

    /* Swap pointers to rTy and rTy_prev */
    tmp = s->rTy_prev;
    s->rTy_prev = s->rTy;
    s->rTy = tmp;

    /* rTy = r' * y */
    checkCudaErrors(cublasTdot(CUDA_handle->cublasHandle, n, s->d_y, 1, s->d_r, 1, s->rTy));

    /* Update residual norm */
    cuda_vec_norm_inf(s->d_r, n, s->d_r_norm);
    checkCudaErrors(cudaMemcpyAsync(s->h_r_norm, s->d_r_norm, sizeof(OSQPFloat), cudaMemcpyDeviceToHost));

    /* beta = rTy / rTy_prev */
    scalar_division_kernel<<<1,1>>>(s->beta, s->rTy, s->rTy_prev);

    /* p *= beta */
    checkCudaErrors(cublasTscal(CUDA_handle->cublasHandle, n, s->beta, s->d_p, 1));

    /* p -= y */
    checkCudaErrors(cublasTaxpy(CUDA_handle->cublasHandle, n, s->D_MINUS_ONE, s->d_y, 1, s->d_p, 1));

    cudaDeviceSynchronize();
    iter++;

  } /* End of the PCG algorithm */

  /* From here on cuBLAS is operating in host pointer mode again */
  cublasSetPointerMode(CUDA_handle->cublasHandle, CUBLAS_POINTER_MODE_HOST);

  return iter;
}


void cuda_pcg_update_precond_diagonal(cudapcg_solver* s,
                                      OSQPInt         P_updated,
                                      OSQPInt         A_updated,
                                      OSQPInt         R_updated) {

  void*      buffer;
  OSQPFloat* tmp;
  OSQPInt    n  = s->n;
  csr*       At = s->At;

  size_t Buffer_size_in_bytes = n * (sizeof(OSQPFloat) + sizeof(OSQPInt));

  if (!P_updated && !A_updated && !R_updated) return;

  if (P_updated) {
    /* Update d_P_diag_val */
    cuda_vec_gather(n, s->P->val, s->d_P_diag_val, s->d_P_diag_ind);
  }

  if (A_updated || R_updated) {
    /* Allocate memory */
    cuda_malloc((void **) &tmp, At->nnz * sizeof(OSQPFloat));
    cuda_malloc((void **) &buffer, Buffer_size_in_bytes);

    /* Update d_AtRA_diag_val */
    if (!s->d_rho_vec) {  /* R = rho*I  -->  A'*R*A = rho * A'*A */

      if (A_updated) {
        /* Update d_AtA_diag_val */
        cuda_vec_ew_prod(tmp, At->val, At->val, At->nnz);
        cuda_vec_segmented_sum(tmp, At->row_ind, s->d_AtA_diag_val, buffer, n, At->nnz);
      }

      /* d_AtRA_diag_val = rho * d_AtA_diag_val */
      cuda_vec_add_scaled(s->d_AtRA_diag_val, s->d_AtA_diag_val, NULL, s->h_rho, 0.0, n);
    }
    else {    /* R = diag(d_rho_vec)  -->  A'*R*A = A' * diag(d_rho_vec) * A */
      cuda_mat_rmult_diag_new(At, tmp, s->d_rho_vec);   /* tmp = A' * R */
      cuda_vec_ew_prod(tmp, tmp, At->val, At->nnz);     /* tmp = tmp * A */
      cuda_vec_segmented_sum(tmp, At->row_ind, s->d_AtRA_diag_val, buffer, n, At->nnz);
    }

    /* Free memory */
    cuda_free((void **) &tmp);
    cuda_free((void **) &buffer);
  }

  /* d_diag_precond = sigma */
  cuda_vec_set_sc(s->d_diag_precond, s->h_sigma, n);

  /* d_diag_precond += d_P_diag_val + d_AtRA_diag_val */
  cuda_vec_add_scaled3(s->d_diag_precond, s->d_diag_precond, s->d_P_diag_val, s->d_AtRA_diag_val, 1.0, 1.0, 1.0, n);

  /* d_diag_precond_inv = 1 / d_diag_precond */
  cuda_vec_reciprocal(s->d_diag_precond_inv, s->d_diag_precond, n);
}


void cuda_pcg_update_precond(cudapcg_solver* s,
                             OSQPInt         P_updated,
                             OSQPInt         A_updated,
                             OSQPInt         R_updated) {

  switch(s->precond_type) {
  /* No preconditioner, just initialize the inverse vector to all 1s */
  case OSQP_NO_PRECONDITIONER:
    cuda_vec_set_sc(s->d_diag_precond_inv, 1.0, s->n);
    break;

  /* Diagonal preconditioner computation */
  case OSQP_DIAGONAL_PRECONDITIONER:
    cuda_pcg_update_precond_diagonal(s, P_updated, A_updated, R_updated);
    break;
  }
}
