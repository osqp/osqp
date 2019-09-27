#include "cuda_lin_alg.h"
#include "cuda_configure.h"
#include "helper_cuda.h"    /* --> checkCudaErrors */


/*******************************************************************************
 *                              GPU Kernels                                    *
 *******************************************************************************/

 __global__ void set_scalar_kernel(c_float *a,
                                   c_float  sc,
                                   c_int    n) {

  c_int idx = threadIdx.x + blockDim.x * blockIdx.x;
  c_int grid_size = blockDim.x * gridDim.x;

  for(c_int i = idx; i < n; i += grid_size) {
    a[i] = sc;
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

void cuda_set_scalar(c_float *d_a,
                     c_float  sc,
                     c_int    n) {

  c_int number_of_blocks = (n / THREADS_PER_BLOCK) + 1;
  set_scalar_kernel<<<number_of_blocks, THREADS_PER_BLOCK>>>(d_a, sc, n);
}

