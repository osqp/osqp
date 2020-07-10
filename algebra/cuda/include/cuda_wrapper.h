/**
 *  Copyright (c) 2019-2020 ETH Zurich, Automatic Control Lab,
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

/********************************************************************
 *       Wrapper functions to abstract floating point type          *
 *                                                                  *
 *  They make the code work when either single or double precision  *
 *  floating-point type is used.                                    *
 ********************************************************************/

#ifndef CUDA_WRAPPER_H
# define CUDA_WRAPPER_H

#include <cusparse.h>
#include <cublas_v2.h>

#include "osqp_api_types.h"


static cublasStatus_t cublasTaxpy(cublasHandle_t  handle,
                                  c_int           n,
                                  const c_float  *alpha,
                                  const c_float  *x,
                                  c_int           incx,
                                  c_float        *y,
                                  c_int           incy) {

#ifdef DFLOAT
  return cublasSaxpy(handle, n, alpha, x, incx, y, incy);
#else
  return cublasDaxpy(handle, n, alpha, x, incx, y, incy);
#endif
}


static cublasStatus_t cublasTscal(cublasHandle_t  handle,
                                  c_int           n,
                                  const c_float  *alpha,
                                  c_float        *x,
                                  c_int           incx) {

#ifdef DFLOAT
  return cublasSscal(handle, n, alpha, x, incx);
#else
  return cublasDscal(handle, n, alpha, x, incx);
#endif
}


static cublasStatus_t cublasTdot(cublasHandle_t  handle,
                                 c_int           n,
                                 const c_float  *x,
                                 c_int           incx,
                                 const c_float  *y,
                                 c_int           incy,
                                 c_float        *result) {

#ifdef DFLOAT
  return cublasSdot (handle, n, x, incx, y, incy, result);
#else
  return cublasDdot (handle, n, x, incx, y, incy, result);
#endif
}


static cublasStatus_t cublasITamax(cublasHandle_t  handle,
                                   c_int           n,
                                   const c_float  *x,
                                   c_int           incx,
                                   c_int          *result) {

#ifdef DFLOAT
  return cublasIsamax(handle, n, x, incx, result);
#else
  return cublasIdamax(handle, n, x, incx, result);
#endif
}


static cublasStatus_t cublasTasum(cublasHandle_t  handle,
                                  c_int           n,
                                  const c_float  *x,
                                  c_int           incx,
                                  c_float        *result) {

#ifdef DFLOAT
  return cublasSasum(handle, n, x, incx, result);
#else
  return cublasDasum(handle, n, x, incx, result);
#endif
}


static cusparseStatus_t cusparseTgthr(cusparseHandle_t  handle,
                                      c_int             nnz,
                                      const c_float    *y,
                                      c_float          *xVal,
                                      const c_int      *xInd) {

#ifdef DFLOAT
  return cusparseSgthr(handle, nnz, y, xVal, xInd, CUSPARSE_INDEX_BASE_ZERO);
#else
  return cusparseDgthr(handle, nnz, y, xVal, xInd, CUSPARSE_INDEX_BASE_ZERO);
#endif
}


static cublasStatus_t cublasTnrm2(cublasHandle_t  handle,
                                  c_int           n,
                                  const c_float  *x,
                                  c_int           incx,
                                  c_float        *result) {

#ifdef DFLOAT
  return cublasSnrm2(handle, n, x, incx, result);
#else
  return cublasDnrm2(handle, n, x, incx, result);
#endif
}


#endif /* ifndef CUDA_WRAPPER */

