/**
 *  Copyright (c) 2019-2020 ETH Zurich, Automatic Control Lab,
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

#ifdef __cplusplus
extern "C" {extern CUDA_Handle_t *CUDA_handle;}
#endif

/*******************************************************************************
 *                              GPU Kernels                                    *
 *******************************************************************************/

__global__ void scalar_division_kernel(c_float       *res,
                                       const c_float *num,
                                       const c_float *den) {

  *res = (*num) / (*den);
}


/*******************************************************************************
 *                            Private Functions                                *
 *******************************************************************************/

/*
 * d_y = (P + sigma*I + A'*R*A) * d_x
 */
static void mat_vec_prod(cudapcg_solver *s,
                         c_float        *d_y,
                         const c_float  *d_x,
                         c_int           device) {

  c_float *sigma;
  c_float H_ZERO = 0.0;
  c_float H_ONE  = 1.0;
  c_int n = s->n;
  c_int m = s->m;
  csr *P  = s->P;
  csr *A  = s->A;
  csr *At = s->At;

  sigma = device ? s->d_sigma : s->h_sigma;

  /* d_y = d_x */
  checkCudaErrors(cudaMemcpy(d_y, d_x, n * sizeof(c_float), cudaMemcpyDeviceToDevice));

  /* d_y *= sigma */
  checkCudaErrors(cublasTscal(CUDA_handle->cublasHandle, n, sigma, d_y, 1));

  /* d_y += P * d_x */
  checkCudaErrors(cusparseCsrmvEx(CUDA_handle->cusparseHandle, P->alg,
                                  CUSPARSE_OPERATION_NON_TRANSPOSE,
                                  P->m, P->n, P->nnz, &H_ONE,
                                  CUDA_FLOAT, P->MatDescription, P->val,
                                  CUDA_FLOAT, P->row_ptr, P->col_ind, d_x,
                                  CUDA_FLOAT, &H_ONE, CUDA_FLOAT, d_y,
                                  CUDA_FLOAT, CUDA_FLOAT, P->buffer));

  if (m == 0) return;

  if (!s->d_rho_vec) {
    /* d_z = rho * A * d_x */
    checkCudaErrors(cusparseCsrmvEx(CUDA_handle->cusparseHandle, A->alg,
                                    CUSPARSE_OPERATION_NON_TRANSPOSE,
                                    A->m, A->n, A->nnz, s->h_rho,
                                    CUDA_FLOAT, A->MatDescription, A->val,
                                    CUDA_FLOAT, A->row_ptr, A->col_ind, d_x,
                                    CUDA_FLOAT, &H_ZERO, CUDA_FLOAT, s->d_z,
                                    CUDA_FLOAT, CUDA_FLOAT, A->buffer));
  }
  else {
    /* d_z = A * d_x */
    checkCudaErrors(cusparseCsrmvEx(CUDA_handle->cusparseHandle, A->alg,
                                    CUSPARSE_OPERATION_NON_TRANSPOSE,
                                    A->m, A->n, A->nnz, &H_ONE,
                                    CUDA_FLOAT, A->MatDescription, A->val,
                                    CUDA_FLOAT, A->row_ptr, A->col_ind, d_x,
                                    CUDA_FLOAT, &H_ZERO, CUDA_FLOAT, s->d_z,
                                    CUDA_FLOAT, CUDA_FLOAT, A->buffer));

    /* d_z = diag(d_rho_vec) * dz */
    cuda_vec_ew_prod(s->d_z, s->d_z, s->d_rho_vec, m);
  }

  /* d_y += A' * d_z */
  checkCudaErrors(cusparseCsrmvEx(CUDA_handle->cusparseHandle, At->alg,
                                  CUSPARSE_OPERATION_NON_TRANSPOSE,
                                  At->m, At->n, At->nnz, &H_ONE,
                                  CUDA_FLOAT, At->MatDescription, At->val,
                                  CUDA_FLOAT, At->row_ptr, At->col_ind, s->d_z,
                                  CUDA_FLOAT, &H_ONE, CUDA_FLOAT, d_y,
                                  CUDA_FLOAT, CUDA_FLOAT, A->buffer));
}


/*******************************************************************************
 *                              API Functions                                  *
 *******************************************************************************/

c_int cuda_pcg_alg(cudapcg_solver *s,
                   c_float         eps,
                   c_int           max_iter) {

  c_float *tmp;

  c_int iter = 0;
  c_int n    = s->n;
  c_float H_MINUS_ONE = -1.0;

  if (!s->warm_start) {
    /* d_x = 0 */
    checkCudaErrors(cudaMemset(s->d_x, 0, n * sizeof(c_float)));
  }

  /* d_p = 0 */
  checkCudaErrors(cudaMemset(s->d_p, 0, n * sizeof(c_float)));

  /* d_r = K * d_x */
  mat_vec_prod(s, s->d_r, s->d_x, 0);

  /* d_r -= d_rhs */
  checkCudaErrors(cublasTaxpy(CUDA_handle->cublasHandle, n, &H_MINUS_ONE, s->d_rhs, 1, s->d_r, 1));

  /* h_r_norm = |d_r| */
  s->vector_norm(s->d_r, n, s->h_r_norm);

  /* From here on cuBLAS is operating in device pointer mode */
  cublasSetPointerMode(CUDA_handle->cublasHandle, CUBLAS_POINTER_MODE_DEVICE);

  if (s->precondition) {
    /* d_y = M \ d_r */
    cuda_vec_ew_prod(s->d_y, s->d_diag_precond_inv, s->d_r, n);
  }

  /* d_p = -d_y */
  checkCudaErrors(cublasTaxpy(CUDA_handle->cublasHandle, n, s->D_MINUS_ONE, s->d_y, 1, s->d_p, 1));

  /* rTy = d_r' * d_y */
  checkCudaErrors(cublasTdot(CUDA_handle->cublasHandle, n, s->d_y, 1, s->d_r, 1, s->rTy));

  cudaDeviceSynchronize();

  /* Run the PCG algorithm */
  while ( *(s->h_r_norm) > eps && iter < max_iter ) {

    /* d_Kp = K * d_p */
    mat_vec_prod(s, s->d_Kp, s->d_p, 1);

    /* pKp = d_p' * d_Kp */
    checkCudaErrors(cublasTdot(CUDA_handle->cublasHandle, n, s->d_p, 1, s->d_Kp, 1, s->pKp));

    /* alpha = rTy / pKp */
    scalar_division_kernel<<<1,1>>>(s->alpha, s->rTy, s->pKp);

    /* d_x += alpha * d_p */
    checkCudaErrors(cublasTaxpy(CUDA_handle->cublasHandle, n, s->alpha, s->d_p, 1, s->d_x, 1));

    /* d_r += alpha * d_Kp */
    checkCudaErrors(cublasTaxpy(CUDA_handle->cublasHandle, n, s->alpha, s->d_Kp, 1, s->d_r, 1));

    if (s->precondition) {
      /* d_y = M \ d_r */
      cuda_vec_ew_prod(s->d_y, s->d_diag_precond_inv, s->d_r, n);
    }

    /* Swap pointers to rTy and rTy_prev */
    tmp = s->rTy_prev;
    s->rTy_prev = s->rTy;
    s->rTy = tmp;

    /* rTy = d_r' * d_y */
    checkCudaErrors(cublasTdot(CUDA_handle->cublasHandle, n, s->d_y, 1, s->d_r, 1, s->rTy));

    /* Update residual norm */
    s->vector_norm(s->d_r, n, s->d_r_norm);
    checkCudaErrors(cudaMemcpyAsync(s->h_r_norm, s->d_r_norm, sizeof(c_float), cudaMemcpyDeviceToHost));

    /* beta = rTy / rTy_prev */
    scalar_division_kernel<<<1,1>>>(s->beta, s->rTy, s->rTy_prev);

    /* d_p *= beta */
    checkCudaErrors(cublasTscal(CUDA_handle->cublasHandle, n, s->beta, s->d_p, 1));

    /* d_p -= d_y */
    checkCudaErrors(cublasTaxpy(CUDA_handle->cublasHandle, n, s->D_MINUS_ONE, s->d_y, 1, s->d_p, 1));

    cudaDeviceSynchronize();
    iter++;

  } /* End of the PCG algorithm */

  /* From here on cuBLAS is operating in host pointer mode again */
  cublasSetPointerMode(CUDA_handle->cublasHandle, CUBLAS_POINTER_MODE_HOST);

  return iter;
}


void cuda_pcg_update_precond(cudapcg_solver *s,
                             c_int           P_updated,
                             c_int           A_updated,
                             c_int           R_updated) {

  void    *buffer;
  c_float *tmp;
  c_int    n  = s->n;
  csr     *At = s->At;

  size_t Buffer_size_in_bytes = n * (sizeof(c_float) + sizeof(c_int));

  if (!P_updated && !A_updated && !R_updated) return;

  if (P_updated) {
    /* Update d_P_diag_val */
    cuda_vec_gather(n, s->P->val, s->d_P_diag_val, s->d_P_diag_ind);
  }

  if (A_updated || R_updated) {
    /* Allocate memory */
    cuda_malloc((void **) &tmp, At->nnz * sizeof(c_float));
    cuda_malloc((void **) &buffer, Buffer_size_in_bytes);

    /* Update d_AtRA_diag_val */
    if (!s->d_rho_vec) {  /* R = rho*I  -->  A'*R*A = rho * A'*A */

      if (A_updated) {
        /* Update d_AtA_diag_val */
        cuda_vec_ew_prod(tmp, At->val, At->val, At->nnz);
        cuda_vec_segmented_sum(tmp, At->row_ind, s->d_AtA_diag_val, buffer, n, At->nnz);
      }

      /* d_AtRA_diag_val = rho * d_AtA_diag_val */
      cuda_vec_add_scaled(s->d_AtRA_diag_val, s->d_AtA_diag_val, NULL, *s->h_rho, 0.0, n);
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
  cuda_vec_set_sc(s->d_diag_precond, *s->h_sigma, n);

  /* d_diag_precond += d_P_diag_val + d_AtRA_diag_val */
  cuda_vec_add_scaled3(s->d_diag_precond, s->d_diag_precond, s->d_P_diag_val, s->d_AtRA_diag_val, 1.0, 1.0, 1.0, n);

  /* d_diag_precond_inv = 1 / d_diag_precond */
  cuda_vec_reciprocal(s->d_diag_precond_inv, s->d_diag_precond, n);
}
