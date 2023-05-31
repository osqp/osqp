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

#include "cuda_lin_alg.h"
#include "cuda_configure.h"
#include "cuda_handler.h"
#include "cuda_malloc.h"
#include "cuda_wrapper.h"
#include "helper_cuda.h"    /* --> checkCudaErrors */

#include "csr_type.h"
#include "glob_opts.h"

#include <thrust/gather.h>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>

extern CUDA_Handle_t *CUDA_handle;


/*******************************************************************************
 *                              GPU Kernels                                    *
 *******************************************************************************/

 __global__ void vec_set_sc_kernel(OSQPFloat* a,
                                   OSQPFloat  sc,
                                   OSQPInt    n) {

  OSQPInt idx = threadIdx.x + blockDim.x * blockIdx.x;
  OSQPInt grid_size = blockDim.x * gridDim.x;

  for(OSQPInt i = idx; i < n; i += grid_size) {
    a[i] = sc;
  }
}

__global__ void vec_set_sc_cond_kernel(OSQPFloat*     a,
                                       const OSQPInt* test,
                                       OSQPFloat      sc_if_neg,
                                       OSQPFloat      sc_if_zero,
                                       OSQPFloat      sc_if_pos,
                                       OSQPInt        n) {

  OSQPInt idx = threadIdx.x + blockDim.x * blockIdx.x;
  OSQPInt grid_size = blockDim.x * gridDim.x;

  for(OSQPInt i = idx; i < n; i += grid_size) {
    if (test[i] == 0)      a[i] = sc_if_zero;
    else if (test[i] > 0)  a[i] = sc_if_pos;
    else                   a[i] = sc_if_neg;
  }
}

__global__ void vec_prod_pos_kernel(const OSQPFloat* a,
                                    const OSQPFloat* b,
                                          OSQPFloat* res,
                                          OSQPInt    n) {

  OSQPInt idx = threadIdx.x + blockDim.x * blockIdx.x;
  OSQPInt grid_size = blockDim.x * gridDim.x;

  OSQPFloat res_kernel = 0.0;

  for(OSQPInt i = idx; i < n; i += grid_size) {
    res_kernel += a[i] * c_max(b[i], 0.0);
  }
  atomicAdd(res, res_kernel);
}

__global__ void vec_prod_neg_kernel(const OSQPFloat* a,
                                    const OSQPFloat* b,
                                          OSQPFloat* res,
                                          OSQPInt    n) {

  OSQPInt idx = threadIdx.x + blockDim.x * blockIdx.x;
  OSQPInt grid_size = blockDim.x * gridDim.x;

  OSQPFloat res_kernel = 0.0;

  for(OSQPInt i = idx; i < n; i += grid_size) {
    res_kernel += a[i] * c_min(b[i], 0.0);
  }
  atomicAdd(res, res_kernel);
}

__global__ void vec_ew_prod_kernel(OSQPFloat*       c,
                                   const OSQPFloat* a,
                                   const OSQPFloat* b,
                                   OSQPInt          n) {

  OSQPInt idx = threadIdx.x + blockDim.x * blockIdx.x;
  OSQPInt grid_size = blockDim.x * gridDim.x;

  for(OSQPInt i = idx; i < n; i += grid_size) {
#ifdef OSQP_USE_FLOAT
    c[i] = __fmul_rn(a[i], b[i]);
#else
    c[i] = __dmul_rn(a[i], b[i]);
#endif
  }
}

__global__ void vec_eq_kernel(const OSQPFloat* a,
                              const OSQPFloat* b,
                                    OSQPFloat  tol,
                                    OSQPInt*   res,
                                    OSQPInt    n) {

  OSQPInt idx = threadIdx.x + blockDim.x * blockIdx.x;
  OSQPInt grid_size = blockDim.x * gridDim.x;

  *res = 1;

  for(OSQPInt i = idx; i < n; i += grid_size) {
      if (c_absval(a[i] - b[i]) > tol) {
          *res = 0;
          break;
      }
  }
}

__global__ void vec_leq_kernel(const OSQPFloat* l,
                               const OSQPFloat* u,
                                     OSQPInt*   res,
                                     OSQPInt    n) {

  OSQPInt idx = threadIdx.x + blockDim.x * blockIdx.x;
  OSQPInt grid_size = blockDim.x * gridDim.x;

  for(OSQPInt i = idx; i < n; i += grid_size) {
    if (l[i] > u[i]) atomicAnd(res, 0);
  }
}

__global__ void vec_bound_kernel(OSQPFloat*       x,
                                 const OSQPFloat* z,
                                 const OSQPFloat* l,
                                 const OSQPFloat* u,
                                 OSQPInt          n) {

  OSQPInt idx = threadIdx.x + blockDim.x * blockIdx.x;
  OSQPInt grid_size = blockDim.x * gridDim.x;

  for(OSQPInt i = idx; i < n; i += grid_size) {
    x[i] = c_min(c_max(z[i], l[i]), u[i]);
  }
}

__global__ void vec_project_polar_reccone_kernel(OSQPFloat*       y,
                                                 const OSQPFloat* l,
                                                 const OSQPFloat* u,
                                                 OSQPFloat        infval,
                                                 OSQPInt          n) {

  OSQPInt idx = threadIdx.x + blockDim.x * blockIdx.x;
  OSQPInt grid_size = blockDim.x * gridDim.x;

  for(OSQPInt i = idx; i < n; i += grid_size) {
    if (u[i] > +infval) {
      if (l[i] < -infval) {
        /* Both bounds infinite */
        y[i] = 0.0;
      }
      else {
        /* Only upper bound infinite */
        y[i] = c_min(y[i], 0.0);
      }
    }
    else if (l[i] < -infval) {
      /* Only lower bound infinite */
      y[i] = c_max(y[i], 0.0);
    }
  }
}

__global__ void vec_in_reccone_kernel(const OSQPFloat* y,
                                      const OSQPFloat* l,
                                      const OSQPFloat* u,
                                            OSQPFloat  infval,
                                            OSQPFloat  tol,
                                            OSQPInt*   res,
                                            OSQPInt    n) {

  OSQPInt idx = threadIdx.x + blockDim.x * blockIdx.x;
  OSQPInt grid_size = blockDim.x * gridDim.x;

  for(OSQPInt i = idx; i < n; i += grid_size) {
    if ( (u[i] < +infval && y[i] > +tol) ||
         (l[i] > -infval && y[i] < -tol) )
      atomicAnd(res, 0);
  }
}

__global__ void vec_reciprocal_kernel(OSQPFloat*       b,
                                      const OSQPFloat* a,
                                      OSQPInt          n) {

  OSQPInt idx = threadIdx.x + blockDim.x * blockIdx.x;
  OSQPInt grid_size = blockDim.x * gridDim.x;

  for(OSQPInt i = idx; i < n; i += grid_size) {
#ifdef OSQP_USE_FLOAT
    b[i] = __frcp_rn(a[i]);
#else
    b[i] = __drcp_rn(a[i]);
#endif
  }
}

__global__ void vec_sqrt_kernel(OSQPFloat* a,
                                OSQPInt    n) {

  OSQPInt idx = threadIdx.x + blockDim.x * blockIdx.x;
  OSQPInt grid_size = blockDim.x * gridDim.x;

  for(OSQPInt i = idx; i < n; i += grid_size) {
#ifdef OSQP_USE_FLOAT
    a[i] = __fsqrt_rn(a[i]);
#else
    a[i] = __dsqrt_rn(a[i]);
#endif
  }
}

__global__ void vec_max_kernel(OSQPFloat*       c,
                               const OSQPFloat* a,
                               const OSQPFloat* b,
                               OSQPInt          n) {

  OSQPInt idx = threadIdx.x + blockDim.x * blockIdx.x;
  OSQPInt grid_size = blockDim.x * gridDim.x;

  for(OSQPInt i = idx; i < n; i += grid_size) {
    c[i] = c_max(a[i], b[i]);
  }
}

__global__ void vec_min_kernel(OSQPFloat*       c,
                               const OSQPFloat* a,
                               const OSQPFloat* b,
                               OSQPInt          n) {

  OSQPInt idx = threadIdx.x + blockDim.x * blockIdx.x;
  OSQPInt grid_size = blockDim.x * gridDim.x;

  for(OSQPInt i = idx; i < n; i += grid_size) {
    c[i] = c_min(a[i], b[i]);
  }
}

__global__ void vec_bounds_type_kernel(OSQPInt*         iseq,
                                       const OSQPFloat* l,
                                       const OSQPFloat* u,
                                       OSQPFloat        infval,
                                       OSQPFloat        tol,
                                       OSQPInt*         has_changed,
                                       OSQPInt          n) {

  OSQPInt idx = threadIdx.x + blockDim.x * blockIdx.x;
  OSQPInt grid_size = blockDim.x * gridDim.x;

  for(OSQPInt i = idx; i < n; i += grid_size) {
    if (u[i] - l[i] < tol) {
      /* Equality constraints */
      if (iseq[i] != 1) {
        iseq[i] = 1;
        atomicOr(has_changed, 1);
      }
    }
    else if ( (l[i] < -infval) && (u[i] > infval) ) {
      /* Loose bounds */
      if (iseq[i] != -1) {
        iseq[i] = -1;
        atomicOr(has_changed, 1);
      }
    }
    else {
      /* Inequality constraints */
      if (iseq[i] != 0) {
        iseq[i] = 0;
        atomicOr(has_changed, 1);
      }
    }
  }
}

__global__ void vec_set_sc_if_lt_kernel(OSQPFloat*       x,
                                        const OSQPFloat* z,
                                        OSQPFloat        testval,
                                        OSQPFloat        newval,
                                        OSQPInt          n) {

  OSQPInt idx = threadIdx.x + blockDim.x * blockIdx.x;
  OSQPInt grid_size = blockDim.x * gridDim.x;

  for(OSQPInt i = idx; i < n; i += grid_size) {
    x[i] = z[i] < testval ? newval : z[i];
  }
}

__global__ void vec_set_sc_if_gt_kernel(OSQPFloat*       x,
                                        const OSQPFloat* z,
                                        OSQPFloat        testval,
                                        OSQPFloat        newval,
                                        OSQPInt          n) {

  OSQPInt idx = threadIdx.x + blockDim.x * blockIdx.x;
  OSQPInt grid_size = blockDim.x * gridDim.x;

  for(OSQPInt i = idx; i < n; i += grid_size) {
    x[i] = z[i] > testval ? newval : z[i];
  }
}

__global__ void mat_lmult_diag_kernel(const OSQPInt*   row_ind,
                                      const OSQPFloat* diag,
                                            OSQPFloat* data,
                                            OSQPInt    nnz) {

  OSQPInt idx = threadIdx.x + blockDim.x * blockIdx.x;
  OSQPInt grid_size = blockDim.x * gridDim.x;

  for(OSQPInt i = idx; i < nnz; i += grid_size) {
    OSQPInt row = row_ind[i];
    data[i] *= diag[row];
  }
}

__global__ void mat_rmult_diag_kernel(const OSQPInt*   col_ind,
                                      const OSQPFloat* diag,
                                            OSQPFloat* data,
                                            OSQPInt    nnz) {

  OSQPInt idx = threadIdx.x + blockDim.x * blockIdx.x;
  OSQPInt grid_size = blockDim.x * gridDim.x;

  for(OSQPInt i = idx; i < nnz; i += grid_size) {
    OSQPInt column = col_ind[i];
    data[i] *= diag[column];
  }
}

__global__ void mat_rmult_diag_new_kernel(const OSQPInt*   col_ind,
                                          const OSQPFloat* diag,
                                          const OSQPFloat* data_in,
                                                OSQPFloat* data_out,
                                                OSQPInt    nnz) {

  OSQPInt idx = threadIdx.x + blockDim.x * blockIdx.x;
  OSQPInt grid_size = blockDim.x * gridDim.x;

  for(OSQPInt i = idx; i < nnz; i += grid_size) {
    OSQPInt column = col_ind[i];
    data_out[i] = data_in[i] * diag[column];
  }
}

__global__ void vec_abs_kernel(OSQPFloat* a,
                               OSQPInt    n) {

  OSQPInt i  = threadIdx.x + blockDim.x * blockIdx.x;

  if (i < n) {
#ifdef OSQP_USE_FLOAT
    a[i] = fabsf(a[i]);
#else
    a[i] = fabs(a[i]);
#endif
  }
}

__global__ void scatter_kernel(OSQPFloat*       out,
                               const OSQPFloat* in,
                               const OSQPInt*   ind,
                               OSQPInt          n) {

  OSQPInt idx = threadIdx.x + blockDim.x * blockIdx.x;
  OSQPInt grid_size = blockDim.x * gridDim.x;

  for(OSQPInt i = idx; i < n; i += grid_size) {
    OSQPInt j = ind[i];
    out[j] = in[i];
  }
}

/*
 * This code complements the cublasITamax routine which only returns the 
 * one-based index to the maximum absolute value in d_x. 
*/
__global__ void abs_kernel(const OSQPInt*   index_one_based,
                           const OSQPFloat* d_x,
                                 OSQPFloat* res) {

  /* cublasITamax returns one-based index */
  (*res) = abs(d_x[(*index_one_based)-1]);
}


/*******************************************************************************
 *                         Private functions                                   *
 *******************************************************************************/

/*
 *  out[j] = in[i], where j = ind[i] for i in [0,n-1]
 */
void scatter(OSQPFloat*       out,
             const OSQPFloat* in,
             const OSQPInt*   ind,
             OSQPInt          n) {

  OSQPInt num_blocks = (n / THREADS_PER_BLOCK) + 1;
  scatter_kernel<<<num_blocks, THREADS_PER_BLOCK>>>(out, in, ind, n);
}


/*******************************************************************************
 *                          Thrust-related functions                           *
 *******************************************************************************/

template<typename BinaryFunction>
void Segmented_reduce(const OSQPInt*       key_start,
                            OSQPInt        number_of_keys,
                            OSQPInt        num_segments,
                      const OSQPFloat*     values,
                            void*          buffer,
                            OSQPFloat*     result,
                            BinaryFunction binary_op) {
 
  OSQPInt num_nnz_rows;

 /*  Memory layout of buffer:
  *  [ m*sizeof(OSQPFloat) Bytes | m*sizeof(OSQPInt) Bytes]
  *  where m = "number of rows"
  */
  OSQPFloat* intermediate_result = (OSQPFloat*) buffer;
  OSQPInt*   nnz_rows            = (OSQPInt*) (&intermediate_result[num_segments]);

  thrust::pair<OSQPInt*,OSQPFloat*> new_end;
  thrust::equal_to<OSQPInt> binary_pred;
  
  new_end = thrust::reduce_by_key(thrust::device,
                                  key_start,
                                  key_start + number_of_keys,
                                  values,
                                  nnz_rows,
                                  intermediate_result,
                                  binary_pred,
                                  binary_op);

  num_nnz_rows = new_end.first - nnz_rows;
  checkCudaErrors(cudaMemset(result, 0, num_segments * sizeof(OSQPFloat)));
  scatter(result, intermediate_result, nnz_rows, num_nnz_rows);
}

template<typename T>
struct abs_maximum {
  typedef T first_argument_type;
  typedef T second_argument_type;
  typedef T result_type;
  __host__ __device__ T operator()(const T &lhs, const T &rhs) const {return max(abs(lhs), abs(rhs));}
 };

template void Segmented_reduce<abs_maximum<OSQPFloat>>(const OSQPInt*                key_start,
                                                             OSQPInt                 number_of_keys,
                                                             OSQPInt                 number_of_segments,
                                                       const OSQPFloat*              values,
                                                             void*                   buffer,
                                                             OSQPFloat*              result,
                                                             abs_maximum<OSQPFloat>  binary_op);


/*******************************************************************************
 *                           API Functions                                     *
 *******************************************************************************/

void cuda_vec_create(cusparseDnVecDescr_t* vec,
                     const OSQPFloat*      d_x,
                     OSQPInt               n) {

  /* Some versions of CUDA don't accept n=0 as a valid size (e.g. can't accept a
   * zero-length vector), so don't create the vector in that case.
   */
  if (n > 0)
    checkCudaErrors(cusparseCreateDnVec(vec, n, (void*)d_x, CUDA_FLOAT));
  else
    vec = NULL;
}

void cuda_vec_destroy(cusparseDnVecDescr_t vec) {

  if (vec) checkCudaErrors(cusparseDestroyDnVec(vec));
}

void cuda_vec_copy_d2d(OSQPFloat*       d_y,
                       const OSQPFloat* d_x,
                       OSQPInt          n) {

  checkCudaErrors(cudaMemcpy(d_y, d_x, n * sizeof(OSQPFloat), cudaMemcpyDeviceToDevice));
}

void cuda_vec_copy_h2d(OSQPFloat*       d_y,
                       const OSQPFloat* h_x,
                       OSQPInt          n) {

  checkCudaErrors(cudaMemcpy(d_y, h_x, n * sizeof(OSQPFloat), cudaMemcpyHostToDevice));
}

void cuda_vec_copy_d2h(OSQPFloat*       h_y,
                       const OSQPFloat* d_x,
                       OSQPInt          n) {

  checkCudaErrors(cudaMemcpy(h_y, d_x, n * sizeof(OSQPFloat), cudaMemcpyDeviceToHost));
}

void cuda_vec_int_copy_h2d(OSQPInt*       d_y,
                           const OSQPInt* h_x,
                           OSQPInt        n) {

  checkCudaErrors(cudaMemcpy(d_y, h_x, n * sizeof(OSQPInt), cudaMemcpyHostToDevice));
}

void cuda_vec_int_copy_d2h(OSQPInt*       h_y,
                           const OSQPInt* d_x,
                           OSQPInt        n) {

  checkCudaErrors(cudaMemcpy(h_y, d_x, n * sizeof(OSQPInt), cudaMemcpyDeviceToHost));
}

void cuda_vec_set_sc(OSQPFloat* d_a,
                     OSQPFloat  sc,
                     OSQPInt    n) {

  OSQPInt number_of_blocks = (n / THREADS_PER_BLOCK) + 1;
  vec_set_sc_kernel<<<number_of_blocks, THREADS_PER_BLOCK>>>(d_a, sc, n);
}

void cuda_vec_set_sc_cond(OSQPFloat*     d_a,
                          const OSQPInt* d_test,
                          OSQPFloat      sc_if_neg,
                          OSQPFloat      sc_if_zero,
                          OSQPFloat      sc_if_pos,
                          OSQPInt        n) {

  OSQPInt number_of_blocks = (n / THREADS_PER_BLOCK) + 1;

  vec_set_sc_cond_kernel<<<number_of_blocks, THREADS_PER_BLOCK>>>(d_a, d_test, sc_if_neg, sc_if_zero, sc_if_pos, n);
}

void cuda_vec_mult_sc(OSQPFloat* d_a,
                      OSQPFloat  sc,
                      OSQPInt    n) {

  checkCudaErrors(cublasTscal(CUDA_handle->cublasHandle, n, &sc, d_a, 1));
}

void cuda_vec_add_scaled(OSQPFloat*       d_x,
                         const OSQPFloat* d_a,
                         const OSQPFloat* d_b,
                         OSQPFloat        sca,
                         OSQPFloat        scb,
                         OSQPInt          n) {

  if (d_x != d_a || sca != 1.0) {
    if (sca == 1.0) {
      /* d_x = d_a */
      checkCudaErrors(cudaMemcpy(d_x, d_a, n * sizeof(OSQPFloat), cudaMemcpyDeviceToDevice));
    }
    else if (d_x == d_a) {
      /* d_x *= sca */
      checkCudaErrors(cublasTscal(CUDA_handle->cublasHandle, n, &sca, d_x, 1));
    }
    else {
      /* d_x = 0 */
      checkCudaErrors(cudaMemset(d_x, 0, n * sizeof(OSQPFloat)));

      /* d_x += sca * d_a */
      checkCudaErrors(cublasTaxpy(CUDA_handle->cublasHandle, n, &sca, d_a, 1, d_x, 1));
    }
  }

  /* d_x += scb * d_b */
  checkCudaErrors(cublasTaxpy(CUDA_handle->cublasHandle, n, &scb, d_b, 1, d_x, 1));
}

void cuda_vec_add_scaled3(OSQPFloat*       d_x,
                          const OSQPFloat* d_a,
                          const OSQPFloat* d_b,
                          const OSQPFloat* d_c,
                          OSQPFloat        sca,
                          OSQPFloat        scb,
                          OSQPFloat        scc,
                          OSQPInt          n) {

  if (d_x != d_a || sca != 1.0) {
    if (sca == 1.0) {
      /* d_x = d_a */
      checkCudaErrors(cudaMemcpy(d_x, d_a, n * sizeof(OSQPFloat), cudaMemcpyDeviceToDevice));
    }
    else if (d_x == d_a) {
      /* d_x *= sca */
      checkCudaErrors(cublasTscal(CUDA_handle->cublasHandle, n, &sca, d_x, 1));
    }
    else {
      /* d_x = 0 */
      checkCudaErrors(cudaMemset(d_x, 0, n * sizeof(OSQPFloat)));

      /* d_x += sca * d_a */
      checkCudaErrors(cublasTaxpy(CUDA_handle->cublasHandle, n, &sca, d_a, 1, d_x, 1));
    }
  }

  /* d_x += scb * d_b */
  checkCudaErrors(cublasTaxpy(CUDA_handle->cublasHandle, n, &scb, d_b, 1, d_x, 1));

  /* d_x += scc * d_c */
  checkCudaErrors(cublasTaxpy(CUDA_handle->cublasHandle, n, &scc, d_c, 1, d_x, 1));
}

void cuda_vec_norm_inf(const OSQPFloat* d_x,
                             OSQPInt    n,
                             OSQPFloat* h_res) {

  cublasPointerMode_t mode;
  checkCudaErrors(cublasGetPointerMode(CUDA_handle->cublasHandle, &mode));

  if (mode == CUBLAS_POINTER_MODE_DEVICE) {
    checkCudaErrors(cublasITamax(CUDA_handle->cublasHandle, n, d_x, 1, CUDA_handle->d_index));
    abs_kernel<<<1,1>>>(CUDA_handle->d_index, d_x, h_res);  /* d_res actually */
  }
  else {
    OSQPInt idx;
    checkCudaErrors(cublasITamax(CUDA_handle->cublasHandle, n, d_x, 1, &idx));
    checkCudaErrors(cudaMemcpy(h_res, d_x + (idx-1), sizeof(OSQPFloat), cudaMemcpyDeviceToHost));
    (*h_res) = abs(*h_res);
  }
}

void cuda_vec_norm_2(const OSQPFloat* d_x,
                           OSQPInt    n,
                           OSQPFloat* h_res) {

  cublasTnrm2(CUDA_handle->cublasHandle, n, d_x, 1, h_res);
}

void cuda_vec_scaled_norm_inf(const OSQPFloat* d_S,
                              const OSQPFloat* d_v,
                                    OSQPInt    n,
                                    OSQPFloat* h_res) {

  OSQPFloat *d_v_scaled;

  cuda_malloc((void **) &d_v_scaled, n * sizeof(OSQPFloat));

  /* d_v_scaled = d_S * d_v */
  cuda_vec_ew_prod(d_v_scaled, d_S, d_v, n);

  /* (*h_res) = |d_v_scaled|_inf */
  cuda_vec_norm_inf(d_v_scaled, n, h_res);

  cuda_free((void **) &d_v_scaled);
}

void cuda_vec_diff_norm_inf(const OSQPFloat* d_a,
                            const OSQPFloat* d_b,
                                  OSQPInt    n,
                                  OSQPFloat* h_res) {

  OSQPFloat *d_diff;

  cuda_malloc((void **) &d_diff, n * sizeof(OSQPFloat));

  /* d_diff = d_a - d_b */
  cuda_vec_add_scaled(d_diff, d_a, d_b, 1.0, -1.0, n);

  /* (*h_res) = |d_diff|_inf */
  cuda_vec_norm_inf(d_diff, n, h_res);

  cuda_free((void **) &d_diff);
}

void cuda_vec_norm_1(const OSQPFloat* d_x,
                           OSQPInt    n,
                           OSQPFloat* h_res) {

  checkCudaErrors(cublasTasum(CUDA_handle->cublasHandle, n, d_x, 1, h_res));
}

void cuda_vec_prod(const OSQPFloat* d_a,
                   const OSQPFloat* d_b,
                         OSQPInt    n,
                         OSQPFloat* h_res) {

  checkCudaErrors(cublasTdot(CUDA_handle->cublasHandle, n, d_a, 1, d_b, 1, h_res));
}

void cuda_vec_prod_signed(const OSQPFloat* d_a,
                          const OSQPFloat* d_b,
                                OSQPInt    sign,
                                OSQPInt    n,
                                OSQPFloat* h_res) {

  OSQPFloat *d_res;
  OSQPInt number_of_blocks = (n / THREADS_PER_BLOCK) + 1;

  cuda_calloc((void **) &d_res, sizeof(OSQPFloat));

  if (sign == 1) {
    vec_prod_pos_kernel<<<number_of_blocks, THREADS_PER_BLOCK>>>(d_a, d_b, d_res, n);
    checkCudaErrors(cudaMemcpy(h_res, d_res, sizeof(OSQPFloat), cudaMemcpyDeviceToHost));
  }
  else if (sign == -1) {
    vec_prod_neg_kernel<<<number_of_blocks, THREADS_PER_BLOCK>>>(d_a, d_b, d_res, n);
    checkCudaErrors(cudaMemcpy(h_res, d_res, sizeof(OSQPFloat), cudaMemcpyDeviceToHost));
  }
  else {
    checkCudaErrors(cublasTdot(CUDA_handle->cublasHandle, n, d_a, 1, d_b, 1, h_res));
  }

  cuda_free((void **) &d_res);
}

void cuda_vec_ew_prod(OSQPFloat*       d_c,
                      const OSQPFloat* d_a,
                      const OSQPFloat* d_b,
                      OSQPInt          n) {

  OSQPInt number_of_blocks = (n / THREADS_PER_BLOCK) + 1;

  vec_ew_prod_kernel<<<number_of_blocks, THREADS_PER_BLOCK>>>(d_c, d_a, d_b, n);
}

void cuda_vec_eq(const OSQPFloat* a,
                 const OSQPFloat* b,
                       OSQPFloat  tol,
                       OSQPInt    n,
                       OSQPInt*   h_res) {

  OSQPInt *d_res;
  OSQPInt number_of_blocks = (n / THREADS_PER_BLOCK) + 1;

  cuda_malloc((void **) &d_res, sizeof(OSQPInt));

  /* Initialize d_res to 1 */
  *h_res = 1;
  checkCudaErrors(cudaMemcpy(d_res, h_res, sizeof(OSQPInt), cudaMemcpyHostToDevice));

  vec_eq_kernel<<<number_of_blocks, THREADS_PER_BLOCK>>>(a, b, tol, d_res, n);

  checkCudaErrors(cudaMemcpy(h_res, d_res, sizeof(OSQPInt), cudaMemcpyDeviceToHost));

  cuda_free((void **) &d_res);
}

void cuda_vec_leq(const OSQPFloat* d_l,
                  const OSQPFloat* d_u,
                        OSQPInt    n,
                        OSQPInt*   h_res) {

  OSQPInt *d_res;
  OSQPInt number_of_blocks = (n / THREADS_PER_BLOCK) + 1;

  cuda_malloc((void **) &d_res, sizeof(OSQPInt));

  /* Initialize d_res to 1 */
  *h_res = 1;
  checkCudaErrors(cudaMemcpy(d_res, h_res, sizeof(OSQPInt), cudaMemcpyHostToDevice));

  vec_leq_kernel<<<number_of_blocks, THREADS_PER_BLOCK>>>(d_l, d_u, d_res, n);

  checkCudaErrors(cudaMemcpy(h_res, d_res, sizeof(OSQPInt), cudaMemcpyDeviceToHost));

  cuda_free((void **) &d_res);
}

void cuda_vec_bound(OSQPFloat*       d_x,
                    const OSQPFloat* d_z,
                    const OSQPFloat* d_l,
                    const OSQPFloat* d_u,
                    OSQPInt          n) {

  OSQPInt number_of_blocks = (n / THREADS_PER_BLOCK) + 1;

  vec_bound_kernel<<<number_of_blocks, THREADS_PER_BLOCK>>>(d_x, d_z, d_l, d_u, n);
}

void cuda_vec_project_polar_reccone(OSQPFloat*       d_y,
                                    const OSQPFloat* d_l,
                                    const OSQPFloat* d_u,
                                    OSQPFloat        infval,
                                    OSQPInt          n) {

  OSQPInt number_of_blocks = (n / THREADS_PER_BLOCK) + 1;

  vec_project_polar_reccone_kernel<<<number_of_blocks, THREADS_PER_BLOCK>>>(d_y, d_l, d_u, infval, n);
}

void cuda_vec_in_reccone(const OSQPFloat* d_y,
                         const OSQPFloat* d_l,
                         const OSQPFloat* d_u,
                               OSQPFloat  infval,
                               OSQPFloat  tol,
                               OSQPInt    n,
                               OSQPInt*   h_res) {

  OSQPInt *d_res;
  OSQPInt number_of_blocks = (n / THREADS_PER_BLOCK) + 1;

  cuda_malloc((void **) &d_res, sizeof(OSQPInt));

  /* Initialize d_res to 1 */
  *h_res = 1;
  checkCudaErrors(cudaMemcpy(d_res, h_res, sizeof(OSQPInt), cudaMemcpyHostToDevice));

  vec_in_reccone_kernel<<<number_of_blocks, THREADS_PER_BLOCK>>>(d_y, d_l, d_u, infval, tol, d_res, n);

  checkCudaErrors(cudaMemcpy(h_res, d_res, sizeof(OSQPInt), cudaMemcpyDeviceToHost));

  cuda_free((void **) &d_res);
}

void cuda_vec_reciprocal(OSQPFloat*       d_b,
                         const OSQPFloat* d_a,
                         OSQPInt          n) {

  OSQPInt number_of_blocks = (n / THREADS_PER_BLOCK) + 1;

  vec_reciprocal_kernel<<<number_of_blocks, THREADS_PER_BLOCK>>>(d_b, d_a, n);
}

void cuda_vec_sqrt(OSQPFloat* d_a,
                   OSQPInt    n) {

  OSQPInt number_of_blocks = (n / THREADS_PER_BLOCK) + 1;

  vec_sqrt_kernel<<<number_of_blocks, THREADS_PER_BLOCK>>>(d_a, n);
}

void cuda_vec_max(OSQPFloat*       d_c,
                  const OSQPFloat* d_a,
                  const OSQPFloat* d_b,
                  OSQPInt          n) {

  OSQPInt number_of_blocks = (n / THREADS_PER_BLOCK) + 1;

  vec_max_kernel<<<number_of_blocks, THREADS_PER_BLOCK>>>(d_c, d_a, d_b, n);
}

void cuda_vec_min(OSQPFloat*       d_c,
                  const OSQPFloat* d_a,
                  const OSQPFloat* d_b,
                  OSQPInt          n) {

  OSQPInt number_of_blocks = (n / THREADS_PER_BLOCK) + 1;

  vec_min_kernel<<<number_of_blocks, THREADS_PER_BLOCK>>>(d_c, d_a, d_b, n);
}

void cuda_vec_bounds_type(OSQPInt*         d_iseq,
                          const OSQPFloat* d_l,
                          const OSQPFloat* d_u,
                          OSQPFloat        infval,
                          OSQPFloat        tol,
                          OSQPInt          n,
                          OSQPInt*         h_has_changed) {

  OSQPInt *d_has_changed;
  OSQPInt number_of_blocks = (n / THREADS_PER_BLOCK) + 1;

  /* Initialize d_has_changed to zero */
  cuda_calloc((void **) &d_has_changed, sizeof(OSQPInt));

  vec_bounds_type_kernel<<<number_of_blocks, THREADS_PER_BLOCK>>>(d_iseq, d_l, d_u, infval, tol, d_has_changed, n);

  checkCudaErrors(cudaMemcpy(h_has_changed, d_has_changed, sizeof(OSQPInt), cudaMemcpyDeviceToHost));

  cuda_free((void **) &d_has_changed);
}

void cuda_vec_set_sc_if_lt(OSQPFloat*       d_x,
                           const OSQPFloat* d_z,
                           OSQPFloat        testval,
                           OSQPFloat        newval,
                           OSQPInt          n) {

  OSQPInt number_of_blocks = (n / THREADS_PER_BLOCK) + 1;

  vec_set_sc_if_lt_kernel<<<number_of_blocks, THREADS_PER_BLOCK>>>(d_x, d_z, testval, newval, n);
}

void cuda_vec_set_sc_if_gt(OSQPFloat*       d_x,
                           const OSQPFloat* d_z,
                           OSQPFloat        testval,
                           OSQPFloat        newval,
                           OSQPInt          n) {

  OSQPInt number_of_blocks = (n / THREADS_PER_BLOCK) + 1;

  vec_set_sc_if_gt_kernel<<<number_of_blocks, THREADS_PER_BLOCK>>>(d_x, d_z, testval, newval, n);
}

void cuda_vec_segmented_sum(const OSQPFloat* d_values,
                            const OSQPInt*   d_keys,
                                  OSQPFloat* d_res,
                                  void*      d_buffer,
                                  OSQPInt    num_segments,
                                  OSQPInt    num_elements) {

  thrust::plus<OSQPFloat> binary_op;
  Segmented_reduce(d_keys, num_elements, num_segments, d_values, d_buffer, d_res, binary_op);
}

void cuda_vec_gather(OSQPInt          nnz,
                     const OSQPFloat* d_y,
                     OSQPFloat*       d_xVal,
                     const OSQPInt*   d_xInd) {

  thrust::gather(thrust::device, d_xInd, d_xInd + nnz, d_y, d_xVal);
}

void cuda_mat_mult_sc(csr*      S,
                      csr*      At,
                      OSQPFloat sc) {

  checkCudaErrors(cublasTscal(CUDA_handle->cublasHandle, S->nnz, &sc, S->val, 1));

  if (At) {
    /* Update At as well */
    checkCudaErrors(cublasTscal(CUDA_handle->cublasHandle, At->nnz, &sc, At->val, 1));
  }
}

void cuda_mat_lmult_diag(csr*             S,
                         csr*             At,
                         const OSQPFloat* d_diag) {

  OSQPInt nnz = S->nnz;
  OSQPInt number_of_blocks = (nnz / THREADS_PER_BLOCK) / ELEMENTS_PER_THREAD + 1;

  mat_lmult_diag_kernel<<<number_of_blocks,THREADS_PER_BLOCK>>>(S->row_ind, d_diag, S->val, nnz);

  if (At) {
    /* Multiply At from right */
    mat_rmult_diag_kernel<<<number_of_blocks,THREADS_PER_BLOCK>>>(At->col_ind, d_diag, At->val, nnz);
  }
}

void cuda_mat_rmult_diag(csr*             S,
                         csr*             At,
                         const OSQPFloat* d_diag) {

  OSQPInt nnz = S->nnz;
  OSQPInt number_of_blocks = (nnz / THREADS_PER_BLOCK) / ELEMENTS_PER_THREAD + 1;

  mat_rmult_diag_kernel<<<number_of_blocks,THREADS_PER_BLOCK>>>(S->col_ind, d_diag, S->val, nnz);

  if (At) {
    /* Multiply At from left */
    mat_lmult_diag_kernel<<<number_of_blocks,THREADS_PER_BLOCK>>>(At->row_ind, d_diag, At->val, nnz);
  }
}

void cuda_mat_rmult_diag_new(const csr*       S,
                                   OSQPFloat* d_buffer,
                             const OSQPFloat* d_diag) {

  OSQPInt nnz = S->nnz;
  OSQPInt number_of_blocks = (nnz / THREADS_PER_BLOCK) / ELEMENTS_PER_THREAD + 1;

  mat_rmult_diag_new_kernel<<<number_of_blocks,THREADS_PER_BLOCK>>>(S->col_ind, d_diag, S->val, d_buffer, nnz);
}

void cuda_mat_Axpy(const csr*                 A,
                   const cusparseDnVecDescr_t vecx,
                         cusparseDnVecDescr_t vecy,
                         OSQPFloat            alpha,
                         OSQPFloat            beta) {
 
  checkCudaErrors(cusparseSpMV(
    CUDA_handle->cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
    &alpha, A->SpMatDescr, vecx, &beta, vecy,
    CUDA_FLOAT, CUSPARSE_SPMV_ALG_DEFAULT, A->SpMatBuffer));
}

void cuda_mat_row_norm_inf(const csr*       S,
                                 OSQPFloat* d_res) {

  OSQPInt nnz      = S->nnz;
  OSQPInt num_rows = S->m;

  if (nnz == 0) return;

  abs_maximum<OSQPFloat> binary_op;
  void *d_buffer;
  cuda_malloc(&d_buffer, num_rows * (sizeof(OSQPFloat) + sizeof(OSQPInt)));

  /* 
  *  For rows with only one element, the element itself is returned.
  *  Therefore, we have to take the absolute value to get the inf-norm.
  */
  Segmented_reduce(S->row_ind, nnz, num_rows, S->val, d_buffer, d_res, binary_op);
  vec_abs_kernel<<<num_rows/THREADS_PER_BLOCK+1,THREADS_PER_BLOCK>>>(d_res, num_rows);

  cuda_free(&d_buffer);
}

