#include "cuda_malloc.h"
#include "helper_cuda.h"    /* --> checkCudaErrors */


#define c_cudaMalloc cudaMalloc

template<typename T>
inline cudaError_t  c_cudaCalloc(T** devPtr, size_t size) {
  cudaError_t cudaCalloc_er = cudaMalloc(devPtr, size);
  if (cudaCalloc_er == cudaSuccess) {
    return cudaMemset(*devPtr, 0, size);
  }
  else {
    return cudaCalloc_er;
  }
}

template<typename T>
inline cudaError_t c_cudaFree(T** devPtr) {
  cudaError_t cuda_error = cudaFree(*devPtr);
  *devPtr = NULL;
  return cuda_error;
}


void cuda_malloc(void** devPtr, size_t size) {
  checkCudaErrors(c_cudaMalloc(devPtr, size));
}

void cuda_calloc(void** devPtr, size_t size) {
  checkCudaErrors(c_cudaCalloc(devPtr, size));
}

void cuda_free(void** devPtr, size_t size) {
  c_cudaFree(devPtr);
}
