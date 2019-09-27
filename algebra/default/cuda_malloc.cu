#include "cuda_malloc.h"
#include "helper_cuda.h"    /* --> checkCudaErrors */


void cuda_malloc(void** devPtr, size_t size) {
  checkCudaErrors(cudaMalloc(devPtr, size));
}

void cuda_calloc(void** devPtr, size_t size) {
  cudaError_t cudaCalloc_er = cudaMalloc(devPtr, size);
  if (cudaCalloc_er == cudaSuccess) {
    return cudaMemset(*devPtr, 0, size);
  }
  else {
    return cudaCalloc_er;
  }
}

void cuda_free(void** devPtr) {
  cudaError_t cuda_error = cudaFree(*devPtr);
  *devPtr = NULL;
  return cuda_error;
}
