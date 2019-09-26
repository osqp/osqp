/********************************************************************
 *       Wrapper functions to abstract floating point type          *
 *                                                                  *
 *  They make the code work when either single or double precision  *
 *  floating-point type is used.                                    *
 ********************************************************************/

#ifndef CUDA_WRAPPER_H
# define CUDA_WRAPPER_H

#include "cuda_configure.h"
#include <cusparse.h>
#include <cublas_v2.h>


static cusparseStatus_t cusparseTcsrmv(cusparseHandle_t          handle,
                                       cusparseOperation_t       transA,
                                       GPU_int                   m,
                                       GPU_int                   n,
                                       GPU_int                   nnz,
                                       const GPU_float          *alpha,
                                       const cusparseMatDescr_t  descrA,
                                       const GPU_float          *csrValA,
                                       const GPU_int            *csrRowPtrA,
                                       const GPU_int            *csrColIndA,
                                       const GPU_float          *x,
                                       const GPU_float          *beta,
                                       GPU_float                *y) {

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


static cublasStatus_t cublasTaxpy(cublasHandle_t   handle,
                                  GPU_int          n,
                                  const GPU_float *alpha,
                                  const GPU_float *x,
                                  GPU_int          incx,
                                  GPU_float       *y,
                                  GPU_int          incy) {

#ifdef DFLOAT
  return cublasSaxpy(handle, n, alpha, x, incx, y, incy);
#else
  return cublasDaxpy(handle, n, alpha, x, incx, y, incy);
#endif
}

static cublasStatus_t cublasTscal(cublasHandle_t   handle,
                                  GPU_int          n,
                                  const GPU_float *alpha,
                                  GPU_float        *x,
                                  GPU_int           incx) {

#ifdef DFLOAT
  return cublasSscal(handle, n, alpha, x, incx);
#else
  return cublasDscal(handle, n, alpha, x, incx);
#endif
}


static cublasStatus_t cublasTdot(cublasHandle_t   handle,
                                 GPU_int          n,
                                 const GPU_float *x,
                                 GPU_int          incx,
                                 const GPU_float *y,
                                 GPU_int          incy,
                                 GPU_float       *result) {

#ifdef DFLOAT
  return cublasSdot (handle, n, x, incx, y, incy, result);
#else
  return cublasDdot (handle, n, x, incx, y, incy, result);
#endif
}


static cublasStatus_t cublasITamax(cublasHandle_t   handle,
                                   GPU_int          n,
                                   const GPU_float *x,
                                   GPU_int          incx,
                                   GPU_int         *result) {

#ifdef DFLOAT
  return cublasIsamax(handle, n, x, incx, result);
#else
  return cublasIdamax(handle, n, x, incx, result);
#endif
}


static cublasStatus_t cublasTasum(cublasHandle_t   handle,
                                  GPU_int          n,
                                  const GPU_float *x,
                                  GPU_int          incx,
                                  GPU_float       *result) {

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
                                  GPU_int            n,
                                  GPU_int            k,
                                  const GPU_float   *A,
                                  GPU_int            lda,
                                  GPU_float         *x,
                                  GPU_int            incx) {

#ifdef DFLOAT
  return cublasStbmv(handle, uplo, trans, diag, n, k, A, lda, x, incx);
#else
  return cublasDtbmv(handle, uplo, trans, diag, n, k, A, lda, x, incx);
#endif
}


static cusparseStatus_t cusparseTgthr(cusparseHandle_t     handle,
                                      GPU_int              nnz,
                                      const GPU_float     *y,
                                      GPU_float           *xVal,
                                      const GPU_int       *xInd,
                                      cusparseIndexBase_t  idxBase) {

#ifdef DFLOAT
  return cusparseSgthr(handle, nnz, y, xVal, xInd, idxBase);
#else
  return cusparseDgthr(handle, nnz, y, xVal, xInd, idxBase);
#endif
}


static cusparseStatus_t cusparseTcsr2csc(cusparseHandle_t handle,
                                         GPU_int m,
                                         GPU_int n,
                                         GPU_int nnz,
                                        const GPU_float *csrVal,
                                        const GPU_int *csrRowPtr,
                                        const GPU_int *csrColInd,
                                        GPU_float           *cscVal,
                                        GPU_int *cscRowInd,
                                        GPU_int *cscColPtr,
                                        cusparseAction_t copyValues,
                                        cusparseIndexBase_t idxBase) {

#ifdef DFLOAT
  return cusparseScsr2csc(handle, m, n, nnz, csrVal, csrRowPtr, csrColInd,
                          cscVal, cscRowInd, cscColPtr, copyValues, idxBase);
#else
  return cusparseDcsr2csc(handle, m, n, nnz, csrVal, csrRowPtr, csrColInd,
                          cscVal, cscRowInd, cscColPtr, copyValues, idxBase);
#endif
}


static cublasStatus_t cublasTcopy(cublasHandle_t   handle,
                                  GPU_int          n,
                                  const GPU_float *x,
                                  GPU_int          incx,
                                  GPU_float       *y,
                                  GPU_int          incy) {

#ifdef DFLOAT
  return cublasScopy(handle, n, x, incx, y, incy);
#else
  return cublasDcopy(handle, n, x, incx, y, incy);
#endif
}


static cublasStatus_t cublasTnrm2(cublasHandle_t   handle,
                                  GPU_int          n,
                                  const GPU_float *x,
                                  GPU_int          incx,
                                  GPU_float       *result) {

#ifdef DFLOAT
  return cublasSnrm2(handle, n, x, incx, result);
#else
  return cublasDnrm2(handle, n, x, incx, result);
#endif
}


static cusparseStatus_t cusparseCsrmv(cusparseHandle_t          handle,
                                      cusparseAlgMode_t         alg,
                                      GPU_int                   m,
                                      GPU_int                   n,
                                      GPU_int                   nnz,
                                      const GPU_float          *alpha,
                                      const cusparseMatDescr_t  descrA,
                                      const GPU_float          *csrValA,
                                      const GPU_int            *csrRowPtrA,
                                      const GPU_int            *csrColIndA,
                                      const GPU_float          *x,
                                      const GPU_float          *beta,
                                      GPU_float                *y,
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
                                                 GPU_int                   m,
                                                 GPU_int                   n,
                                                 GPU_int                   nnz,
                                                 const GPU_float          *alpha,
                                                 const cusparseMatDescr_t  descrA,
                                                 const GPU_float          *csrValA,
                                                 const GPU_int            *csrRowPtrA,
                                                 const GPU_int            *csrColIndA,
                                                 const GPU_float          *x,
                                                 const GPU_float          *beta,
                                                 GPU_float                *y,
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
                                                   GPU_int               m,
                                                   GPU_int               n,
                                                   GPU_int               nnz,
                                                   const void           *csrVal,
                                                   const GPU_int        *csrRowPtr,
                                                   const GPU_int        *csrColInd,
                                                   void                 *cscVal,
                                                   GPU_int              *cscColPtr,
                                                   GPU_int              *cscRowInd,
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
                                        GPU_int               m,
                                        GPU_int               n,
                                        GPU_int               nnz,
                                        const void           *csrVal,
                                        const GPU_int        *csrRowPtr,
                                        const GPU_int        *csrColInd,
                                        void                 *cscVal,
                                        GPU_int              *cscColPtr,
                                        GPU_int              *cscRowInd,
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
