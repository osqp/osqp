#ifndef ALGEBRA_MEMORY_H_
#define ALGEBRA_MEMORY_H_

/* Define the memory management functions for the builtin algebra */
#define blas_malloc  c_malloc
#define blas_calloc  c_calloc
#define blas_free    c_free
#define blas_realloc c_realloc

#endif
