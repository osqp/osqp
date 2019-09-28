#include "cuda_lin_alg.h"
#include "cuda_configure.h"
#include "cuda_handler.h"
#include "cuda_wrapper.h"
#include "helper_cuda.h"    /* --> checkCudaErrors */


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

void cuda_norm_inf(const c_float *d_x,
                   c_int          n,
                   c_float       *h_res) {

  c_int   idx;

  checkCudaErrors(cublasITamax(CUDA_handle->cublasHandle, n, d_x, 1, &idx));
  checkCudaErrors(cudaMemcpy(h_res, d_x + (idx-1), sizeof(c_float), cudaMemcpyDeviceToHost));
  (*h_res) = abs(*h_res);
}
