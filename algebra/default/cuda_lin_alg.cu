#include "cuda_lin_alg.h"
#include "cuda_configure.h"
#include "cuda_handler.h"
#include "cuda_malloc.h"
#include "cuda_wrapper.h"
#include "helper_cuda.h"    /* --> checkCudaErrors */

#include "glob_opts.h"


extern CUDA_Handle_t *CUDA_handle;

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
    a[i] = __frsqrt_rn(a[i]);
#else
    a[i] = __drcp_rn(__dsqrt_rn(a[i]));
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

  c_int idx;

  checkCudaErrors(cublasITamax(CUDA_handle->cublasHandle, n, d_x, 1, &idx));
  checkCudaErrors(cudaMemcpy(h_res, d_x + (idx-1), sizeof(c_float), cudaMemcpyDeviceToHost));
  (*h_res) = abs(*h_res);
}

void cuda_vec_norm_1(const c_float *d_x,
                     c_int          n,
                     c_float       *h_res) {

  cublasTasum(CUDA_handle->cublasHandle, n, d_x, 1, h_res);
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

  checkCudaErrors(cudaMemcpy(&h_res, d_res, sizeof(c_int), cudaMemcpyDeviceToHost));

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
