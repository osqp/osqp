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

#include "cuda_handler.h"
#include "helper_cuda.h"


CUDA_Handle_t* cuda_init_libs(int device) {

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

  checkCudaErrors(cudaSetDevice(device));
  checkCudaErrors(cusparseCreate(&CUDA_handle->cusparseHandle));
  checkCudaErrors(cublasCreate(&CUDA_handle->cublasHandle));
  checkCudaErrors(cudaMalloc(&CUDA_handle->d_index, sizeof(int)));

  return CUDA_handle;
}


void cuda_free_libs(CUDA_Handle_t *CUDA_handle) {
  cusparseDestroy(CUDA_handle->cusparseHandle);
  cublasDestroy(CUDA_handle->cublasHandle);
  cudaFree(CUDA_handle->d_index);
  free(CUDA_handle);
}

