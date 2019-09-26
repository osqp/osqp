#include "cuda_handler.h"
#include "helper_cuda.h"


CUDA_Handle_t CUDA_handle;
CUDA_handle.cublasHandle   = NULL;
CUDA_handle.cusparseHandle = NULL;
CUDA_handle.d_index        = NULL;


void CUDA_init_libs(void) {

  int deviceCount = 0;

  cudaGetDeviceCount(&deviceCount);
  if (!deviceCount) printf("No GPU detected.\n");

  checkCudaErrors(cudaSetDevice(0));
  checkCudaErrors(cusparseCreate(&CUDA_handle.cusparseHandle));
  checkCudaErrors(cublasCreate(&CUDA_handle.cublasHandle));
  checkCudaErrors(cudaMalloc(&CUDA_handle.d_index, sizeof(int)));
}


void CUDA_free_libs() {
  cusparseDestroy(CUDA_handle.cusparseHandle);
  cublasDestroy(CUDA_handle.cublasHandle);
  cudaFree(CUDA_handle.d_index);
}
