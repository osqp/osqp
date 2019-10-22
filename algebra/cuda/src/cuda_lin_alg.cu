/**
 *  Copyright (c) 2019 ETH Zurich, Automatic Control Lab, Michel Schubiger, Goran Banjac.
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

#include <thrust/reduce.h>
#include <thrust/execution_policy.h>

#ifdef __cplusplus
extern "C" {extern CUDA_Handle_t *CUDA_handle;}
#endif


/*******************************************************************************
 *                              GPU Kernels                                    *
 *******************************************************************************/

 __global__ void vec_set_sc_kernel(c_float *a,
                                   c_float  sc,
                                   c_int    n) {

  c_int idx = threadIdx.x + blockDim.x * blockIdx.x;
  c_int grid_size = blockDim.x * gridDim.x;

  for(c_int i = idx; i < n; i += grid_size) {
    a[i] = sc;
  }
}

__global__ void vec_set_sc_cond_kernel(c_float     *a,
                                       const c_int *test,
                                       c_float      sc_if_neg,
                                       c_float      sc_if_zero,
                                       c_float      sc_if_pos,
                                       c_int        n) {

  c_int idx = threadIdx.x + blockDim.x * blockIdx.x;
  c_int grid_size = blockDim.x * gridDim.x;

  for(c_int i = idx; i < n; i += grid_size) {
    if (test[i] == 0)      a[i] = sc_if_zero;
    else if (test[i] > 0)  a[i] = sc_if_pos;
    else                   a[i] = sc_if_neg;
  }
}

__global__ void vec_prod_pos_kernel(const c_float *a,
                                    const c_float *b,
                                    c_float       *res,
                                    c_int          n) {

  c_int idx = threadIdx.x + blockDim.x * blockIdx.x;
  c_int grid_size = blockDim.x * gridDim.x;

  c_float res_kernel = 0.0;

  for(c_int i = idx; i < n; i += grid_size) {
    res_kernel += a[i] * c_max(b[i], 0.0);
  }
  atomicAdd(res, res_kernel);
}

__global__ void vec_prod_neg_kernel(const c_float *a,
                                    const c_float *b,
                                    c_float       *res,
                                    c_int          n) {

  c_int idx = threadIdx.x + blockDim.x * blockIdx.x;
  c_int grid_size = blockDim.x * gridDim.x;

  c_float res_kernel = 0.0;

  for(c_int i = idx; i < n; i += grid_size) {
    res_kernel += a[i] * c_min(b[i], 0.0);
  }
  atomicAdd(res, res_kernel);
}

__global__ void vec_ew_prod_kernel(c_float       *c,
                                   const c_float *a,
                                   const c_float *b,
                                   c_int          n) {

  c_int idx = threadIdx.x + blockDim.x * blockIdx.x;
  c_int grid_size = blockDim.x * gridDim.x;

  for(c_int i = idx; i < n; i += grid_size) {
#ifdef DFLOAT
    c[i] = __fmul_rn(a[i], b[i]);
#else
    c[i] = __dmul_rn(a[i], b[i]);
#endif
  }
}

__global__ void vec_leq_kernel(const c_float *l,
                               const c_float *u,
                               c_int         *res,
                               c_int          n) {

  c_int idx = threadIdx.x + blockDim.x * blockIdx.x;
  c_int grid_size = blockDim.x * gridDim.x;

  for(c_int i = idx; i < n; i += grid_size) {
    if (l[i] > u[i]) atomicAnd(res, 0);
  }
}

__global__ void vec_bound_kernel(c_float       *x,
                                 const c_float *z,
                                 const c_float *l,
                                 const c_float *u,
                                 c_int          n) {

  c_int idx = threadIdx.x + blockDim.x * blockIdx.x;
  c_int grid_size = blockDim.x * gridDim.x;

  for(c_int i = idx; i < n; i += grid_size) {
    x[i] = c_min(c_max(z[i], l[i]), u[i]);
  }
}

__global__ void vec_project_polar_reccone_kernel(c_float       *y,
                                                 const c_float *l,
                                                 const c_float *u,
                                                 c_float        infval,
                                                 c_int          n) {

  c_int idx = threadIdx.x + blockDim.x * blockIdx.x;
  c_int grid_size = blockDim.x * gridDim.x;

  for(c_int i = idx; i < n; i += grid_size) {
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

__global__ void vec_in_reccone_kernel(const c_float *y,
                                      const c_float *l,
                                      const c_float *u,
                                      c_float        infval,
                                      c_float        tol,
                                      c_int         *res,
                                      c_int          n) {

  c_int idx = threadIdx.x + blockDim.x * blockIdx.x;
  c_int grid_size = blockDim.x * gridDim.x;

  for(c_int i = idx; i < n; i += grid_size) {
    if ( (u[i] < +infval && y[i] > +tol) ||
         (l[i] > -infval && y[i] < -tol) )
      atomicAnd(res, 0);
  }
}

__global__ void vec_reciprocal_kernel(c_float       *b,
                                      const c_float *a,
                                      c_int          n) {

  c_int idx = threadIdx.x + blockDim.x * blockIdx.x;
  c_int grid_size = blockDim.x * gridDim.x;

  for(c_int i = idx; i < n; i += grid_size) {
#ifdef DFLOAT
    b[i] = __frcp_rn(a[i]);
#else
    b[i] = __drcp_rn(a[i]);
#endif
  }
}

__global__ void vec_sqrt_kernel(c_float *a,
                                c_int    n) {

  c_int idx = threadIdx.x + blockDim.x * blockIdx.x;
  c_int grid_size = blockDim.x * gridDim.x;

  for(c_int i = idx; i < n; i += grid_size) {
#ifdef DFLOAT
    a[i] = __fsqrt_rn(a[i]);
#else
    a[i] = __dsqrt_rn(a[i]);
#endif
  }
}

__global__ void vec_max_kernel(c_float       *c,
                               const c_float *a,
                               const c_float *b,
                               c_int          n) {

  c_int idx = threadIdx.x + blockDim.x * blockIdx.x;
  c_int grid_size = blockDim.x * gridDim.x;

  for(c_int i = idx; i < n; i += grid_size) {
    c[i] = c_max(a[i], b[i]);
  }
}

__global__ void vec_min_kernel(c_float       *c,
                               const c_float *a,
                               const c_float *b,
                               c_int          n) {

  c_int idx = threadIdx.x + blockDim.x * blockIdx.x;
  c_int grid_size = blockDim.x * gridDim.x;

  for(c_int i = idx; i < n; i += grid_size) {
    c[i] = c_min(a[i], b[i]);
  }
}

__global__ void vec_bounds_type_kernel(c_int         *iseq,
                                       const c_float *l,
                                       const c_float *u,
                                       c_float        infval,
                                       c_float        tol,
                                       c_int         *has_changed,
                                       c_int          n) {

  c_int idx = threadIdx.x + blockDim.x * blockIdx.x;
  c_int grid_size = blockDim.x * gridDim.x;

  for(c_int i = idx; i < n; i += grid_size) {
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

__global__ void vec_set_sc_if_lt_kernel(c_float       *x,
                                        const c_float *z,
                                        c_float        testval,
                                        c_float        newval,
                                        c_int          n) {

  c_int idx = threadIdx.x + blockDim.x * blockIdx.x;
  c_int grid_size = blockDim.x * gridDim.x;

  for(c_int i = idx; i < n; i += grid_size) {
    x[i] = z[i] < testval ? newval : z[i];
  }
}

__global__ void vec_set_sc_if_gt_kernel(c_float       *x,
                                        const c_float *z,
                                        c_float        testval,
                                        c_float        newval,
                                        c_int          n) {

  c_int idx = threadIdx.x + blockDim.x * blockIdx.x;
  c_int grid_size = blockDim.x * gridDim.x;

  for(c_int i = idx; i < n; i += grid_size) {
    x[i] = z[i] > testval ? newval : z[i];
  }
}

__global__ void mat_lmult_diag_kernel(const c_int   *row_ind,
                                      const c_float *diag,
                                      c_float       *data,
                                      c_int          nnz) {

  c_int idx = threadIdx.x + blockDim.x * blockIdx.x;
  c_int grid_size = blockDim.x * gridDim.x;

  for(c_int i = idx; i < nnz; i += grid_size) {
    c_int row = row_ind[i];
    data[i] *= diag[row];
  }
}

__global__ void mat_rmult_diag_kernel(const c_int   *col_ind,
                                      const c_float *diag,
                                      c_float       *data,
                                      c_int          nnz) {

  c_int idx = threadIdx.x + blockDim.x * blockIdx.x;
  c_int grid_size = blockDim.x * gridDim.x;

  for(c_int i = idx; i < nnz; i += grid_size) {
    c_int column = col_ind[i];
    data[i] *= diag[column];
  }
}

__global__ void mat_rmult_diag_new_kernel(const c_int   *col_ind,
                                          const c_float *diag,
                                          const c_float *data_in,
                                          c_float       *data_out,
                                          c_int          nnz) {

  c_int idx = threadIdx.x + blockDim.x * blockIdx.x;
  c_int grid_size = blockDim.x * gridDim.x;

  for(c_int i = idx; i < nnz; i += grid_size) {
    c_int column = col_ind[i];
    data_out[i] = data_in[i] * diag[column];
  }
}

__global__ void vec_abs_kernel(c_float *a,
                               c_int    n) {

  c_int i  = threadIdx.x + blockDim.x * blockIdx.x;

  if (i < n) {
#ifdef DFLOAT
    a[i] = fabsf(a[i]);
#else
    a[i] = fabs(a[i]);
#endif
  }
}

__global__ void scatter_kernel(c_float       *out,
                               const c_float *in,
                               const c_int   *ind,
                               c_int          n) {

  c_int idx = threadIdx.x + blockDim.x * blockIdx.x;
  c_int grid_size = blockDim.x * gridDim.x;

  for(c_int i = idx; i < n; i += grid_size) {
    c_int j = ind[i];
    out[j] = in[i];
  }
}

/*
 * This code complements the cublasITamax routine which only returns the 
 * one-based index to the maximum absolute value in d_x. 
*/
__global__ void abs_kernel(const c_int   *index_one_based,
                           const c_float *d_x,
                           c_float       *res) {

  /* cublasITamax returns one-based index */
  (*res) = abs(d_x[(*index_one_based)-1]);
}


/*******************************************************************************
 *                         Private functions                                   *
 *******************************************************************************/

/*
 *  out[j] = in[i], where j = ind[i] for i in [0,n-1]
 */
void scatter(c_float       *out,
             const c_float *in,
             const c_int   *ind,
             c_int          n) {

  c_int num_blocks = (n / THREADS_PER_BLOCK) + 1;
  scatter_kernel<<<num_blocks, THREADS_PER_BLOCK>>>(out, in, ind, n);
}


/*******************************************************************************
 *                          Thrust-related functions                           *
 *******************************************************************************/

template<typename BinaryFunction>
void Segmented_reduce(const c_int    *key_start,
                      c_int           number_of_keys,
                      c_int           num_segments,
                      const c_float  *values,
                      void           *buffer,
                      c_float        *result,
                      BinaryFunction  binary_op) {
 
  c_int num_nnz_rows;

 /*  Memory layout of buffer:
  *  [ m*sizeof(c_float) Bytes | m*sizeof(c_int) Bytes]
  *  where m = "number of rows"
  */
  c_float *intermediate_result = (c_float*) buffer; 
  c_int   *nnz_rows            = (c_int*) (&intermediate_result[num_segments]);

  thrust::pair<c_int*,c_float*> new_end;
  thrust::equal_to<c_int> binary_pred;
  
  new_end = thrust::reduce_by_key(thrust::device,
                                  key_start,
                                  key_start + number_of_keys,
                                  values,
                                  nnz_rows,
                                  intermediate_result,
                                  binary_pred,
                                  binary_op);

  num_nnz_rows = new_end.first - nnz_rows;
  checkCudaErrors(cudaMemset(result, 0, num_segments * sizeof(c_float)));
  scatter(result, intermediate_result, nnz_rows, num_nnz_rows);
}

template<typename T>
struct abs_maximum {
  typedef T first_argument_type;
  typedef T second_argument_type;
  typedef T result_type;
  __host__ __device__ T operator()(const T &lhs, const T &rhs) const {return max(abs(lhs), abs(rhs));}
 };

template void Segmented_reduce<abs_maximum<c_float>>(const c_int          *key_start,
                                                     c_int                 number_of_keys,
                                                     c_int                 number_of_segments,
                                                     const c_float        *values,
                                                     void                 *buffer,
                                                     c_float              *result,
                                                     abs_maximum<c_float>  binary_op);


/*******************************************************************************
 *                           API Functions                                     *
 *******************************************************************************/

void cuda_vec_copy_d2d(c_float       *d_y,
                       const c_float *d_x,
                       c_int          n) {

  checkCudaErrors(cudaMemcpy(d_y, d_x, n * sizeof(c_float), cudaMemcpyDeviceToDevice));
}

void cuda_vec_copy_h2d(c_float       *d_y,
                       const c_float *h_x,
                       c_int          n) {

  checkCudaErrors(cudaMemcpy(d_y, h_x, n * sizeof(c_float), cudaMemcpyHostToDevice));
}

void cuda_vec_copy_d2h(c_float       *h_y,
                       const c_float *d_x,
                       c_int          n) {

  checkCudaErrors(cudaMemcpy(h_y, d_x, n * sizeof(c_float), cudaMemcpyDeviceToHost));
}

void cuda_vec_int_copy_h2d(c_int       *d_y,
                           const c_int *h_x,
                           c_int        n) {

  checkCudaErrors(cudaMemcpy(d_y, h_x, n * sizeof(c_int), cudaMemcpyHostToDevice));
}

void cuda_vec_int_copy_d2h(c_int       *h_y,
                           const c_int *d_x,
                           c_int        n) {

  checkCudaErrors(cudaMemcpy(h_y, d_x, n * sizeof(c_int), cudaMemcpyDeviceToHost));
}

void cuda_vec_set_sc(c_float *d_a,
                     c_float  sc,
                     c_int    n) {

  c_int number_of_blocks = (n / THREADS_PER_BLOCK) + 1;
  vec_set_sc_kernel<<<number_of_blocks, THREADS_PER_BLOCK>>>(d_a, sc, n);
}

void cuda_vec_set_sc_cond(c_float     *d_a,
                          const c_int *d_test,
                          c_float      sc_if_neg,
                          c_float      sc_if_zero,
                          c_float      sc_if_pos,
                          c_float      n) {

  c_int number_of_blocks = (n / THREADS_PER_BLOCK) + 1;

  vec_set_sc_cond_kernel<<<number_of_blocks, THREADS_PER_BLOCK>>>(d_a, d_test, sc_if_neg, sc_if_zero, sc_if_pos, n);
}

void cuda_vec_mult_sc(c_float *d_a,
                      c_float  sc,
                      c_int    n) {

  checkCudaErrors(cublasTscal(CUDA_handle->cublasHandle, n, &sc, d_a, 1));
}

void cuda_vec_add_scaled(c_float       *d_x,
                         const c_float *d_a,
                         const c_float *d_b,
                         c_float        sca,
                         c_float        scb,
                         c_int          n) {

  if (d_x != d_a || sca != 1.0) {
    if (sca == 1.0) {
      /* d_x = d_a */
      checkCudaErrors(cudaMemcpy(d_x, d_a, n * sizeof(c_float), cudaMemcpyDeviceToDevice));
    }
    else if (d_x == d_a) {
      /* d_x *= sca */
      checkCudaErrors(cublasTscal(CUDA_handle->cublasHandle, n, &sca, d_x, 1));
    }
    else {
      /* d_x = 0 */
      checkCudaErrors(cudaMemset(d_x, 0, n * sizeof(c_float)));

      /* d_x += sca * d_a */
      checkCudaErrors(cublasTaxpy(CUDA_handle->cublasHandle, n, &sca, d_a, 1, d_x, 1));
    }
  }

  /* d_x += scb * d_b */
  checkCudaErrors(cublasTaxpy(CUDA_handle->cublasHandle, n, &scb, d_b, 1, d_x, 1));
}

void cuda_vec_add_scaled3(c_float       *d_x,
                          const c_float *d_a,
                          const c_float *d_b,
                          const c_float *d_c,
                          c_float        sca,
                          c_float        scb,
                          c_float        scc,
                          c_int          n) {

  if (d_x != d_a || sca != 1.0) {
    if (sca == 1.0) {
      /* d_x = d_a */
      checkCudaErrors(cudaMemcpy(d_x, d_a, n * sizeof(c_float), cudaMemcpyDeviceToDevice));
    }
    else if (d_x == d_a) {
      /* d_x *= sca */
      checkCudaErrors(cublasTscal(CUDA_handle->cublasHandle, n, &sca, d_x, 1));
    }
    else {
      /* d_x = 0 */
      checkCudaErrors(cudaMemset(d_x, 0, n * sizeof(c_float)));

      /* d_x += sca * d_a */
      checkCudaErrors(cublasTaxpy(CUDA_handle->cublasHandle, n, &sca, d_a, 1, d_x, 1));
    }
  }

  /* d_x += scb * d_b */
  checkCudaErrors(cublasTaxpy(CUDA_handle->cublasHandle, n, &scb, d_b, 1, d_x, 1));

  /* d_x += scc * d_c */
  checkCudaErrors(cublasTaxpy(CUDA_handle->cublasHandle, n, &scc, d_c, 1, d_x, 1));
}

void cuda_vec_norm_inf(const c_float *d_x,
                       c_int          n,
                       c_float       *h_res) {

  cublasPointerMode_t mode;
  checkCudaErrors(cublasGetPointerMode(CUDA_handle->cublasHandle, &mode));

  if (mode == CUBLAS_POINTER_MODE_DEVICE) {
    checkCudaErrors(cublasITamax(CUDA_handle->cublasHandle, n, d_x, 1, CUDA_handle->d_index));
    abs_kernel<<<1,1>>>(CUDA_handle->d_index, d_x, h_res);  /* d_res actually */
  }
  else {
    c_int idx;
    checkCudaErrors(cublasITamax(CUDA_handle->cublasHandle, n, d_x, 1, &idx));
    checkCudaErrors(cudaMemcpy(h_res, d_x + (idx-1), sizeof(c_float), cudaMemcpyDeviceToHost));
    (*h_res) = abs(*h_res);
  }
}

void cuda_vec_norm_1(const c_float *d_x,
                     c_int          n,
                     c_float       *h_res) {

  cublasTasum(CUDA_handle->cublasHandle, n, d_x, 1, h_res);
}

void cuda_vec_norm_2(const c_float *d_x,
                     c_int          n,
                     c_float       *h_res) {

  cublasTnrm2(CUDA_handle->cublasHandle, n, d_x, 1, h_res);
}

void cuda_vec_scaled_norm_inf(const c_float *d_S,
                              const c_float *d_v,
                              c_int          n,
                              c_float       *h_res) {

  c_float *d_v_scaled;

  cuda_malloc((void **) &d_v_scaled, n * sizeof(c_float));

  /* d_v_scaled = d_S * d_v */
  cuda_vec_ew_prod(d_v_scaled, d_S, d_v, n);

  /* (*h_res) = |d_v_scaled|_inf */
  cuda_vec_norm_inf(d_v_scaled, n, h_res);

  cuda_free((void **) &d_v_scaled);
}

void cuda_vec_diff_norm_inf(const c_float *d_a,
                            const c_float *d_b,
                            c_int          n,
                            c_float       *h_res) {

  c_float *d_diff;

  cuda_malloc((void **) &d_diff, n * sizeof(c_float));

  /* d_diff = d_a - d_b */
  cuda_vec_add_scaled(d_diff, d_a, d_b, 1.0, -1.0, n);

  /* (*h_res) = |d_diff|_inf */
  cuda_vec_norm_inf(d_diff, n, h_res);

  cuda_free((void **) &d_diff);
}

void cuda_vec_mean(const c_float *d_x,
                   c_int          n,
                   c_float       *h_res) {

  cublasTasum(CUDA_handle->cublasHandle, n, d_x, 1, h_res);
  (*h_res) /= n;
}

void cuda_vec_prod(const c_float *d_a,
                   const c_float *d_b,
                   c_int          n,
                   c_float       *h_res) {

  checkCudaErrors(cublasTdot(CUDA_handle->cublasHandle, n, d_a, 1, d_b, 1, h_res));
}

void cuda_vec_prod_signed(const c_float *d_a,
                          const c_float *d_b,
                          c_int          sign,
                          c_int          n,
                          c_float       *h_res) {

  c_float *d_res;
  c_int number_of_blocks = (n / THREADS_PER_BLOCK) + 1;

  cuda_calloc((void **) &d_res, sizeof(c_float));

  if (sign == 1) {
    vec_prod_pos_kernel<<<number_of_blocks, THREADS_PER_BLOCK>>>(d_a, d_b, d_res, n);
    checkCudaErrors(cudaMemcpy(h_res, d_res, sizeof(c_float), cudaMemcpyDeviceToHost));
  }
  else if (sign == -1) {
    vec_prod_neg_kernel<<<number_of_blocks, THREADS_PER_BLOCK>>>(d_a, d_b, d_res, n);
    checkCudaErrors(cudaMemcpy(h_res, d_res, sizeof(c_float), cudaMemcpyDeviceToHost));
  }
  else {
    checkCudaErrors(cublasTdot(CUDA_handle->cublasHandle, n, d_a, 1, d_b, 1, h_res));
  }

  cuda_free((void **) &d_res);
}

void cuda_vec_ew_prod(c_float       *d_c,
                      const c_float *d_a,
                      const c_float *d_b,
                      c_int          n) {

  c_int number_of_blocks = (n / THREADS_PER_BLOCK) + 1;

  vec_ew_prod_kernel<<<number_of_blocks, THREADS_PER_BLOCK>>>(d_c, d_a, d_b, n);
}

void cuda_vec_leq(const c_float *d_l,
                   const c_float *d_u,
                   c_int          n,
                   c_int         *h_res) {

  c_int *d_res;
  c_int number_of_blocks = (n / THREADS_PER_BLOCK) + 1;

  cuda_malloc((void **) &d_res, sizeof(c_int));

  /* Initialize d_res to 1 */
  *h_res = 1;
  checkCudaErrors(cudaMemcpy(d_res, h_res, sizeof(c_int), cudaMemcpyHostToDevice));

  vec_leq_kernel<<<number_of_blocks, THREADS_PER_BLOCK>>>(d_l, d_u, d_res, n);

  checkCudaErrors(cudaMemcpy(h_res, d_res, sizeof(c_int), cudaMemcpyDeviceToHost));

  cuda_free((void **) &d_res);
}

void cuda_vec_bound(c_float       *d_x,
                    const c_float *d_z,
                    const c_float *d_l,
                    const c_float *d_u,
                    c_int          n) {

  c_int number_of_blocks = (n / THREADS_PER_BLOCK) + 1;

  vec_bound_kernel<<<number_of_blocks, THREADS_PER_BLOCK>>>(d_x, d_z, d_l, d_u, n);
}

void cuda_vec_project_polar_reccone(c_float       *d_y,
                                    const c_float *d_l,
                                    const c_float *d_u,
                                    c_float        infval,
                                    c_int          n) {

  c_int number_of_blocks = (n / THREADS_PER_BLOCK) + 1;

  vec_project_polar_reccone_kernel<<<number_of_blocks, THREADS_PER_BLOCK>>>(d_y, d_l, d_u, infval, n);
}

void cuda_vec_in_reccone(const c_float *d_y,
                         const c_float *d_l,
                         const c_float *d_u,
                         c_float        infval,
                         c_float        tol,
                         c_int          n,
                         c_int         *h_res) {

  c_int *d_res;
  c_int number_of_blocks = (n / THREADS_PER_BLOCK) + 1;

  cuda_malloc((void **) &d_res, sizeof(c_int));

  /* Initialize d_res to 1 */
  *h_res = 1;
  checkCudaErrors(cudaMemcpy(d_res, h_res, sizeof(c_int), cudaMemcpyHostToDevice));

  vec_in_reccone_kernel<<<number_of_blocks, THREADS_PER_BLOCK>>>(d_y, d_l, d_u, infval, tol, d_res, n);

  checkCudaErrors(cudaMemcpy(h_res, d_res, sizeof(c_int), cudaMemcpyDeviceToHost));

  cuda_free((void **) &d_res);
}

void cuda_vec_reciprocal(c_float       *d_b,
                         const c_float *d_a,
                         c_int          n) {

  c_int number_of_blocks = (n / THREADS_PER_BLOCK) + 1;

  vec_reciprocal_kernel<<<number_of_blocks, THREADS_PER_BLOCK>>>(d_b, d_a, n);
}

void cuda_vec_sqrt(c_float *d_a,
                   c_int    n) {

  c_int number_of_blocks = (n / THREADS_PER_BLOCK) + 1;

  vec_sqrt_kernel<<<number_of_blocks, THREADS_PER_BLOCK>>>(d_a, n);
}

void cuda_vec_max(c_float       *d_c,
                  const c_float *d_a,
                  const c_float *d_b,
                  c_int          n) {

  c_int number_of_blocks = (n / THREADS_PER_BLOCK) + 1;

  vec_max_kernel<<<number_of_blocks, THREADS_PER_BLOCK>>>(d_c, d_a, d_b, n);
}

void cuda_vec_min(c_float       *d_c,
                  const c_float *d_a,
                  const c_float *d_b,
                  c_int          n) {

  c_int number_of_blocks = (n / THREADS_PER_BLOCK) + 1;

  vec_min_kernel<<<number_of_blocks, THREADS_PER_BLOCK>>>(d_c, d_a, d_b, n);
}

void cuda_vec_bounds_type(c_int         *d_iseq,
                          const c_float *d_l,
                          const c_float *d_u,
                          c_float        infval,
                          c_float        tol,
                          c_int          n,
                          c_int         *h_has_changed) {

  c_int *d_has_changed;
  c_int number_of_blocks = (n / THREADS_PER_BLOCK) + 1;

  /* Initialize d_has_changed to zero */
  cuda_calloc((void **) &d_has_changed, sizeof(c_int));

  vec_bounds_type_kernel<<<number_of_blocks, THREADS_PER_BLOCK>>>(d_iseq, d_l, d_u, infval, tol, d_has_changed, n);

  checkCudaErrors(cudaMemcpy(h_has_changed, d_has_changed, sizeof(c_int), cudaMemcpyDeviceToHost));

  cuda_free((void **) &d_has_changed);
}

void cuda_vec_set_sc_if_lt(c_float       *d_x,
                           const c_float *d_z,
                           c_float        testval,
                           c_float        newval,
                           c_int          n) {

  c_int number_of_blocks = (n / THREADS_PER_BLOCK) + 1;

  vec_set_sc_if_lt_kernel<<<number_of_blocks, THREADS_PER_BLOCK>>>(d_x, d_z, testval, newval, n);
}

void cuda_vec_set_sc_if_gt(c_float       *d_x,
                           const c_float *d_z,
                           c_float        testval,
                           c_float        newval,
                           c_int          n) {

  c_int number_of_blocks = (n / THREADS_PER_BLOCK) + 1;

  vec_set_sc_if_gt_kernel<<<number_of_blocks, THREADS_PER_BLOCK>>>(d_x, d_z, testval, newval, n);
}

void cuda_vec_segmented_sum(const c_float *d_values,
                            const c_int   *d_keys,
                            c_float       *d_res,
                            void          *d_buffer,
                            c_int          num_segments,
                            c_int          num_elements) {

  thrust::plus<c_float> binary_op;
  Segmented_reduce(d_keys, num_elements, num_segments, d_values, d_buffer, d_res, binary_op);
}

void cuda_mat_mult_sc(csr     *S,
                      csr     *At,
                      c_int    symmetric,
                      c_float  sc) {

  checkCudaErrors(cublasTscal(CUDA_handle->cublasHandle, S->nnz, &sc, S->val, 1));

  if (!symmetric) {
    /* Update At as well */
    checkCudaErrors(cublasTscal(CUDA_handle->cublasHandle, At->nnz, &sc, At->val, 1));
  }
}

void cuda_mat_lmult_diag(csr           *S,
                         csr           *At,
                         c_int          symmetric,
                         const c_float *d_diag) {

  c_int nnz = S->nnz;
  c_int number_of_blocks = (nnz / THREADS_PER_BLOCK) / ELEMENTS_PER_THREAD + 1;

  mat_lmult_diag_kernel<<<number_of_blocks,THREADS_PER_BLOCK>>>(S->row_ind, d_diag, S->val, nnz);

  if (!symmetric) {
    /* Multiply At from right */
    mat_rmult_diag_kernel<<<number_of_blocks,THREADS_PER_BLOCK>>>(At->col_ind, d_diag, At->val, nnz);
  }
}

void cuda_mat_rmult_diag(csr           *S,
                         csr           *At,
                         c_int          symmetric,
                         const c_float *d_diag) {

  c_int nnz = S->nnz;
  c_int number_of_blocks = (nnz / THREADS_PER_BLOCK) / ELEMENTS_PER_THREAD + 1;

  mat_rmult_diag_kernel<<<number_of_blocks,THREADS_PER_BLOCK>>>(S->col_ind, d_diag, S->val, nnz);

  if (!symmetric) {
    /* Multiply At from left */
    mat_lmult_diag_kernel<<<number_of_blocks,THREADS_PER_BLOCK>>>(At->row_ind, d_diag, At->val, nnz);
  }
}

void cuda_mat_rmult_diag_new(const csr     *S,
                             c_float       *d_buffer,
                             const c_float *d_diag) {

  c_int nnz = S->nnz;
  c_int number_of_blocks = (nnz / THREADS_PER_BLOCK) / ELEMENTS_PER_THREAD + 1;

  mat_rmult_diag_new_kernel<<<number_of_blocks,THREADS_PER_BLOCK>>>(S->col_ind, d_diag, S->val, d_buffer, nnz);
}

void cuda_mat_Axpy(const csr     *A,
                   const c_float *d_x,
                   c_float       *d_y,
                   c_float        alpha,
                   c_float        beta) {

  if (A->nnz == 0 || alpha == 0.0) {
    /* d_y = beta * d_y */
    cuda_vec_mult_sc(d_y, beta, A->m);
    return;
  }

  checkCudaErrors(cusparseCsrmv(CUDA_handle->cusparseHandle, A->alg, A->m, A->n, A->nnz, &alpha, A->MatDescription, A->val, A->row_ptr, A->col_ind, d_x, &beta, d_y, A->buffer));
}

void cuda_mat_quad_form(const csr     *P,
                        const c_float *d_x,
                        c_float       *h_res) {

  c_int n = P->n;
  c_float *d_Px;

  cuda_malloc((void **) &d_Px, n * sizeof(c_float));

  /* d_Px = P * x */
  cuda_mat_Axpy(P, d_x, d_Px, 1.0, 0.0);

  /* h_res = d_Px' * d_x */
  cuda_vec_prod(d_Px, d_x, n, h_res);

  /* h_res *= 0.5 */
  (*h_res) *= 0.5;

  cuda_free((void **) &d_Px);
}

void cuda_mat_row_norm_inf(const csr *S,
                           c_float   *d_res) {

  c_int nnz      = S->nnz;
  c_int num_rows = S->m;

  if (nnz == 0) return;

  abs_maximum<c_float> binary_op;
  void *d_buffer;
  cuda_malloc(&d_buffer, num_rows * (sizeof(c_float) + sizeof(c_int)));

  /* 
  *  For rows with only one element, the element itself is returned.
  *  Therefore, we have to take the absolute value to get the inf-norm.
  */
  Segmented_reduce(S->row_ind, nnz, num_rows, S->val, d_buffer, d_res, binary_op);
  vec_abs_kernel<<<num_rows/THREADS_PER_BLOCK+1,THREADS_PER_BLOCK>>>(d_res, num_rows);

  cuda_free(&d_buffer);
}
