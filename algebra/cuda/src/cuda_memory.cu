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

#include "cuda_memory.h"
#include "helper_cuda.h"    /* --> checkCudaErrors */


#define c_cudaMalloc cudaMalloc
#define c_cudaMallocHost cudaMallocHost


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

template<typename T>
inline cudaError_t c_cudaFreeHost(T** devPtr) {
  cudaError_t cuda_error = cudaFreeHost(*devPtr);
  *devPtr = NULL;
  return cuda_error;
}


void cuda_malloc(void** devPtr, size_t size) {
  checkCudaErrors(c_cudaMalloc(devPtr, size));
}

void cuda_malloc_host(void** devPtr, size_t size) {
  checkCudaErrors(c_cudaMallocHost(devPtr, size));
}

void cuda_calloc(void** devPtr, size_t size) {
  checkCudaErrors(c_cudaCalloc(devPtr, size));
}

void cuda_free(void** devPtr) {
  checkCudaErrors(c_cudaFree(devPtr));
}

void cuda_free_host(void** devPtr) {
  checkCudaErrors(c_cudaFreeHost(devPtr));
}

bool cuda_isdeviceptr(const void* ptr) {
  struct cudaPointerAttributes attributes;

  cudaError_t err = cudaPointerGetAttributes(&attributes, ptr);

  // It may be possible for host allocated memory to return an error code of
  // cudaErrorInvalidValue instead of cudaSuccess, so we don't want to treat
  // that error code as an actual error, just as a sign the memory is on the host.
  if (err == cudaErrorInvalidValue) {
    cudaGetLastError();
    return false;
  }
  else if (err != cudaSuccess) {
    checkCudaErrors(err);
    return false;
  }

  // This memory has been allocated on the device for sure.
  if (attributes.type == cudaMemoryTypeDevice) {
    return true;
  }

  // We don't handle the cudaMemoryTypeUnregistered, cudaMemoryTypeHost or cudaMemoryTypeManaged
  // possibilities explicitly, because we are curious about if the memory is on the device.
  // cudaMemoryTypeUnregistered and cudaMemoryTypeHost imply the memory is on the host, and
  // cudaMemoryTypeManaged means the data is in memory that could be accessible from either the CPU
  // or device directly, however we always want data on the GPU (so we will do a manual copy from that
  // memory to the GPU when initializing the matrices/vectors).

  // Clear out any errors that may have happened (just in case)
  cudaGetLastError();

  return false;
}

cudaError_t cuda_memcpy_hd2d(void* dst, const void* src, size_t count) {
  if (cuda_isdeviceptr(src))
    return cudaMemcpy(dst, src, count, cudaMemcpyDeviceToDevice);
  else
    return cudaMemcpy(dst, src, count, cudaMemcpyHostToDevice);
}
