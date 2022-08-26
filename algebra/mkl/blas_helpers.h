#ifndef BLAS_HELPERS_H_
#define BLAS_HELPERS_H_

#include "osqp_configure.h"

#include <mkl_blas.h>
#include <mkl_spblas.h>

#include <mkl.h>

/* Let the user override the MKL memory alignment if they really want,
   but default to 64-bytes alignment if nothing is specified. */
#ifndef OSQP_MKL_MEMORY_ALIGNMENT
# define OSQP_MKL_MEMORY_ALIGNMENT 64
#endif

/* Define the memory management functions for the MKL algebra */
#define blas_malloc(alloc_size)  mkl_malloc(alloc_size, OSQP_MKL_MEMORY_ALIGNMENT)
#define blas_calloc(num, size)   mkl_calloc(num, size, OSQP_MKL_MEMORY_ALIGNMENT)
#define blas_free                mkl_free
#define blas_realloc             mkl_realloc

/* Define the blas functions based on the data type we are using */
#ifdef OSQP_USE_FLOAT
  #define blas_copy  scopy
  #define blas_dot   sdot
  #define blas_scale sscal
  #define blas_swap  sswap
  #define blas_axpy  saxpy
  #define blas_2norm snrm2

  #define spblas_create_csc mkl_sparse_s_create_csc
  #define spblas_set_value  mkl_sparse_s_set_value
  #define spblas_export_csc mkl_sparse_s_export_csc
  #define spblas_mv         mkl_sparse_s_mv
#else
  #define blas_copy  dcopy
  #define blas_dot   ddot
  #define blas_scale dscal
  #define blas_swap  dswap
  #define blas_axpy  daxpy
  #define blas_2norm dnrm2

  #define spblas_create_csc mkl_sparse_d_create_csc
  #define spblas_set_value  mkl_sparse_d_set_value
  #define spblas_export_csc mkl_sparse_d_export_csc
  #define spblas_mv         mkl_sparse_d_mv
#endif /* OSQP_USE_FLOAT */

#endif
