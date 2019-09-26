#include "cuda_handler.h"
#include "helper_cuda.h"


CUDA_Handle_t* CUDA_init_libs(void) {

  int deviceCount = 0;

  cudaGetDeviceCount(&deviceCount);
  if (deviceCount == 0) {
    printf("No GPU detected.\n");
    return NULL;
  }

  CUDA_Handle_t *CUDA_handle = (CUDA_Handle_t*) malloc(sizeof(CUDA_Handle_t));
  if (!CUDA_handle) {
    printf("Memory allocation error.\n");
    return NULL;
  }

  checkCudaErrors(cudaSetDevice(0));
  checkCudaErrors(cusparseCreate(&CUDA_handle->cusparseHandle));
  checkCudaErrors(cublasCreate(&CUDA_handle->cublasHandle));
  checkCudaErrors(cudaMalloc(&CUDA_handle->d_index, sizeof(int)));

  return CUDA_handle;
}


void CUDA_free_libs(CUDA_Handle_t *CUDA_handle) {
  cusparseDestroy(CUDA_handle->cusparseHandle);
  cublasDestroy(CUDA_handle->cublasHandle);
  cudaFree(CUDA_handle->d_index);
  free(CUDA_handle);
}

