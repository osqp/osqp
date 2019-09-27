#ifndef CUDA_MALLOC_H
# define CUDA_MALLOC_H

// #include "cuda.h"
// #include <cuda_runtime.h>

extern "C" {

void cuda_malloc(void** devPtr, size_t size);

void cuda_calloc(void** devPtr, size_t size);

void cuda_free(void** devPtr);

}

#endif /* ifndef CUDA_MALLOC_H */
