#ifndef BLAS_HELPERS_H_
#define BLAS_HELPERS_H_

#include "osqp_configure.h"

#include <mkl.h>
#include <mkl_cblas.h>
#include <mkl_spblas.h>
#include <mkl_vml.h>

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
  // MKL Level 1 BLAS functions
  #define blas_copy  cblas_scopy
  #define blas_dot   cblas_sdot
  #define blas_scale cblas_sscal
  #define blas_swap  cblas_sswap
  #define blas_axpy  cblas_saxpy
  #define blas_2norm cblas_snrm2
  #define blas_asum  cblas_sasum
  #define blas_iamax cblas_isamax

  // MKL Vector Math functions
  #define vml_add    vsAdd
  #define vml_sub    vsSub
  #define vml_mul    vsMul
  #define vml_max    vsFmax
  #define vml_maxinc vsFmaxI
  #define vml_min    vsFmin
  #define vml_mininc vsFminI
  #define vml_inv    vsInv
  #define vml_sqrt   vsSqrt

  // MKL Sparse BLAS functions
  #define spblas_create_csc mkl_sparse_s_create_csc
  #define spblas_set_value  mkl_sparse_s_set_value
  #define spblas_export_csc mkl_sparse_s_export_csc
  #define spblas_mv         mkl_sparse_s_mv
#else
  // MKL Level 1 BLAS functions
  #define blas_copy  cblas_dcopy
  #define blas_dot   cblas_ddot
  #define blas_scale cblas_dscal
  #define blas_swap  cblas_dswap
  #define blas_axpy  cblas_daxpy
  #define blas_2norm cblas_dnrm2
  #define blas_asum  cblas_dasum
  #define blas_iamax cblas_idamax

  // MKL Vector Math functions
  #define vml_add    vdAdd
  #define vml_sub    vdSub
  #define vml_mul    vdMul
  #define vml_max    vdFmax
  #define vml_maxinc vdFmaxI
  #define vml_min    vdFmin
  #define vml_mininc vdFminI
  #define vml_inv    vdInv
  #define vml_sqrt   vdSqrt

  // MKL Sparse BLAS functions
  #define spblas_create_csc mkl_sparse_d_create_csc
  #define spblas_set_value  mkl_sparse_d_set_value
  #define spblas_export_csc mkl_sparse_d_export_csc
  #define spblas_mv         mkl_sparse_d_mv
#endif /* OSQP_USE_FLOAT */

#endif
