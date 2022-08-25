#ifndef ALGEBRA_MEMORY_H_
#define ALGEBRA_MEMORY_H_

#include <mkl.h>

/* Align to 64-bytes */
#define MEMORY_ALIGNMENT 64

/* Define the memory management functions for the MKL algebra */
#define blas_malloc(alloc_size)  mkl_malloc(alloc_size, MEMORY_ALIGNMENT)
#define blas_calloc(num, size)   mkl_calloc(num, size, MEMORY_ALIGNMENT)
#define blas_free    mkl_free
#define blas_realloc mkl_realloc

#endif
