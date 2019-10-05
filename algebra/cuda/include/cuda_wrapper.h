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


static cusparseStatus_t cusparseTcsrmv(cusparseHandle_t          handle,
                                       cusparseOperation_t       transA,
                                       c_int                     m,
                                       c_int                     n,
                                       c_int                     nnz,
                                       const c_float            *alpha,
                                       const cusparseMatDescr_t  descrA,
                                       const c_float            *csrValA,
                                       const c_int              *csrRowPtrA,
                                       const c_int              *csrColIndA,
                                       const c_float            *x,
                                       const c_float            *beta,
                                       c_float                  *y) {

#ifdef DFLOAT
  return cusparseScsrmv(handle, transA, m, n, nnz, alpha,
                        descrA,  csrValA, csrRowPtrA, csrColIndA,
                        x, beta, y);
#else
  return cusparseDcsrmv(handle, transA, m, n, nnz, alpha,
                        descrA,  csrValA, csrRowPtrA, csrColIndA,
                        x, beta, y);
#endif
}


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


static cublasStatus_t cublasTtbmv(cublasHandle_t     handle,
                                  cublasFillMode_t   uplo,
                                  cublasOperation_t  trans,
                                  cublasDiagType_t   diag,
                                  c_int              n,
                                  c_int              k,
                                  const c_float     *A,
                                  c_int              lda,
                                  c_float           *x,
                                  c_int              incx) {

#ifdef DFLOAT
  return cublasStbmv(handle, uplo, trans, diag, n, k, A, lda, x, incx);
#else
  return cublasDtbmv(handle, uplo, trans, diag, n, k, A, lda, x, incx);
#endif
}


static cusparseStatus_t cusparseTgthr(cusparseHandle_t     handle,
                                      c_int                nnz,
                                      const c_float       *y,
                                      c_float             *xVal,
                                      const c_int         *xInd,
                                      cusparseIndexBase_t  idxBase) {

#ifdef DFLOAT
  return cusparseSgthr(handle, nnz, y, xVal, xInd, idxBase);
#else
  return cusparseDgthr(handle, nnz, y, xVal, xInd, idxBase);
#endif
}


static cusparseStatus_t cusparseTcsr2csc(cusparseHandle_t     handle,
                                         c_int                m,
                                         c_int                n,
                                         c_int                nnz,
                                         const c_float       *csrVal,
                                         const c_int         *csrRowPtr,
                                         const c_int         *csrColInd,
                                         c_float             *cscVal,
                                         c_int               *cscRowInd,
                                         c_int               *cscColPtr,
                                         cusparseAction_t     copyValues,
                                         cusparseIndexBase_t  idxBase) {

#ifdef DFLOAT
  return cusparseScsr2csc(handle, m, n, nnz, csrVal, csrRowPtr, csrColInd,
                          cscVal, cscRowInd, cscColPtr, copyValues, idxBase);
#else
  return cusparseDcsr2csc(handle, m, n, nnz, csrVal, csrRowPtr, csrColInd,
                          cscVal, cscRowInd, cscColPtr, copyValues, idxBase);
#endif
}


static cublasStatus_t cublasTcopy(cublasHandle_t  handle,
                                  c_int           n,
                                  const c_float  *x,
                                  c_int           incx,
                                  c_float        *y,
                                  c_int           incy) {

#ifdef DFLOAT
  return cublasScopy(handle, n, x, incx, y, incy);
#else
  return cublasDcopy(handle, n, x, incx, y, incy);
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


static cusparseStatus_t cusparseCsrmv(cusparseHandle_t          handle,
                                      cusparseAlgMode_t         alg,
                                      c_int                     m,
                                      c_int                     n,
                                      c_int                     nnz,
                                      const c_float            *alpha,
                                      const cusparseMatDescr_t  descrA,
                                      const c_float            *csrValA,
                                      const c_int              *csrRowPtrA,
                                      const c_int              *csrColIndA,
                                      const c_float            *x,
                                      const c_float            *beta,
                                      c_float                  *y,
                                      void                     *buffer) {

#ifdef DFLOAT
  return cusparseCsrmvEx(handle, alg, CUSPARSE_OPERATION_NON_TRANSPOSE, m, n, nnz, alpha,
                         CUDA_R_32F, descrA, csrValA, CUDA_R_32F, csrRowPtrA, csrColIndA, x,
                         CUDA_R_32F, beta, CUDA_R_32F, y, CUDA_R_32F, CUDA_R_32F, buffer);
#else
  return cusparseCsrmvEx(handle, alg, CUSPARSE_OPERATION_NON_TRANSPOSE, m, n, nnz, alpha,
                         CUDA_R_64F, descrA, csrValA, CUDA_R_64F, csrRowPtrA, csrColIndA, x,
                         CUDA_R_64F, beta, CUDA_R_64F, y, CUDA_R_64F, CUDA_R_64F, buffer);
#endif
} 


static cusparseStatus_t cusparseCsrmv_bufferSize(cusparseHandle_t          handle,
                                                 cusparseAlgMode_t         alg,
                                                 c_int                     m,
                                                 c_int                     n,
                                                 c_int                     nnz,
                                                 const c_float            *alpha,
                                                 const cusparseMatDescr_t  descrA,
                                                 const c_float            *csrValA,
                                                 const c_int              *csrRowPtrA,
                                                 const c_int              *csrColIndA,
                                                 const c_float            *x,
                                                 const c_float            *beta,
                                                 c_float                  *y,
                                                 size_t                   *bufferSizeInBytes) {

#ifdef DFLOAT
  return cusparseCsrmvEx_bufferSize(handle, alg, CUSPARSE_OPERATION_NON_TRANSPOSE, m, n, nnz, alpha,
                                    CUDA_R_32F, descrA, csrValA, CUDA_R_32F, csrRowPtrA, csrColIndA, x,
                                    CUDA_R_32F, beta, CUDA_R_32F, y, CUDA_R_32F, CUDA_R_32F, bufferSizeInBytes);
#else
  return cusparseCsrmvEx_bufferSize(handle, alg, CUSPARSE_OPERATION_NON_TRANSPOSE, m, n, nnz, alpha,
                                    CUDA_R_64F, descrA, csrValA, CUDA_R_64F, csrRowPtrA, csrColIndA, x,
                                    CUDA_R_64F, beta, CUDA_R_64F, y, CUDA_R_64F, CUDA_R_64F, bufferSizeInBytes);
#endif 
}


static cusparseStatus_t cusparseCsr2csc_bufferSize(cusparseHandle_t      handle,
                                                   c_int                 m,
                                                   c_int                 n,
                                                   c_int                 nnz,
                                                   const void           *csrVal,
                                                   const c_int          *csrRowPtr,
                                                   const c_int          *csrColInd,
                                                   void                 *cscVal,
                                                   c_int                *cscColPtr,
                                                   c_int                *cscRowInd,
                                                   cusparseAction_t      copyValues,
                                                   cusparseIndexBase_t   idxBase,
                                                   cusparseCsr2CscAlg_t  alg,
                                                   size_t               *bufferSize) {

#ifdef DFLOAT
  return cusparseCsr2cscEx2_bufferSize(handle, m, n, nnz, csrVal, csrRowPtr, csrColInd,
                                       cscVal, cscColPtr, cscRowInd, CUDA_R_32F,
                                       copyValues, idxBase, alg, bufferSize);
#else
  return cusparseCsr2cscEx2_bufferSize(handle, m, n, nnz, csrVal, csrRowPtr, csrColInd,
                                       cscVal, cscColPtr, cscRowInd, CUDA_R_64F,
                                       copyValues, idxBase, alg, bufferSize);
#endif
}


static cusparseStatus_t cusparseCsr2csc(cusparseHandle_t      handle,
                                        c_int                 m,
                                        c_int                 n,
                                        c_int                 nnz,
                                        const void           *csrVal,
                                        const c_int          *csrRowPtr,
                                        const c_int          *csrColInd,
                                        void                 *cscVal,
                                        c_int                *cscColPtr,
                                        c_int                *cscRowInd,
                                        cusparseAction_t      copyValues,
                                        cusparseIndexBase_t   idxBase,
                                        cusparseCsr2CscAlg_t  alg,
                                        void                 *buffer) {

#ifdef DFLOAT
  return cusparseCsr2cscEx2(handle, m, n, nnz, csrVal, csrRowPtr, csrColInd,
                            cscVal, cscColPtr, cscRowInd, CUDA_R_32F,
                            copyValues, idxBase, alg, buffer);
#else
  return cusparseCsr2cscEx2(handle, m, n, nnz, csrVal, csrRowPtr, csrColInd,
                            cscVal, cscColPtr, cscRowInd, CUDA_R_64F,
                            copyValues, idxBase, alg, buffer);
#endif
}

#endif /* ifndef CUDA_WRAPPER */
