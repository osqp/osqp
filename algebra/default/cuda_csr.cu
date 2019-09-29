#include "cuda_csr.h"
#include "cuda_handler.h"
#include "helper_cuda.h"    /* --> checkCudaErrors */

#include "glob_opts.h"


extern CUDA_Handle_t *CUDA_handle;


/*******************************************************************************
 *                         Private Functions                                   *
 *******************************************************************************/

 /*
 *  Update the size of buffer used for the merge path based
 *  sparse matrix-vector product (spmv).
 */
void update_mp_buffer(csr *P) {

  size_t bufferSizeInBytes = 0;
  c_float alpha = 1.0;

  checkCudaErrors(cusparseCsrmv_bufferSize(CUDA_Handle->cusparseHandle,
                                           P->alg, P->m, P->n, P->nnz,
                                           &alpha,
                                           P->MatDescription, P->val, P->row_ptr, P->col_ind,
                                           NULL,
                                           &alpha,
                                           NULL,
                                           &bufferSizeInBytes));
  
  if (bufferSizeInBytes > P->bufferSizeInBytes) {
    cuda_free((void **) &P->buffer);                            
    checkCudaErrors(cuda_malloc((void **) &P->buffer, bufferSizeInBytes));
    P->bufferSizeInBytes = bufferSizeInBytes;
  }
}

 /*
 *  Creates a CSR matrix with the specified dimension (m,n,nnz).
 *  
 *  If specified, it allocates proper amount of device memory
 *  allocate_on_device = 1: device memory for CSR
 *  allocate_on_device = 2: device memory for CSR (+ col_ind)  
 */
csr* csr_alloc(c_int m,
               c_int n,
               c_int nnz,
               c_int allocate_on_device) {

  csr *dev_mat = (csr*) c_calloc(1, sizeof(csr));

  if (!dev_mat) return NULL;

  dev_mat->m   = m;
  dev_mat->n   = n;
  dev_mat->nnz = nnz;

  dev_mat->val     = NULL;
  dev_mat->row_ptr = NULL;
  dev_mat->col_ind = NULL;
  dev_mat->row_ind = NULL;
      
#ifdef IS_WINDOWS
  /* MERGE_PATH is not working properly on WINDOWS */
  dev_mat->alg = CUSPARSE_ALG_NAIVE;
#else
  dev_mat->alg = CUSPARSE_ALG_MERGE_PATH;
#endif

  dev_mat->buffer = NULL;
  dev_mat->bufferSizeInBytes = 0;

  checkCudaErrors(cusparseCreateMatDescr(&dev_mat->MatDescription));
  cusparseSetMatType(dev_mat->MatDescription, CUSPARSE_MATRIX_TYPE_GENERAL);
  cusparseSetMatIndexBase(dev_mat->MatDescription, CUSPARSE_INDEX_BASE_ZERO);

  if (allocate_on_device > 0) {
    checkCudaErrors(cuda_calloc((void **) &dev_mat->val, (dev_mat->nnz + 1) * sizeof(c_float)));
    checkCudaErrors(cuda_malloc((void **) &dev_mat->row_ptr, (dev_mat->m + 1) * sizeof(c_int))); 
    checkCudaErrors(cuda_malloc((void **) &dev_mat->col_ind, dev_mat->nnz * sizeof(c_int)));

    if (allocate_on_device > 1) {
      checkCudaErrors(cuda_malloc((void **) &dev_mat->row_ind, dev_mat->nnz * sizeof(c_int)));
    } 
  }
  return dev_mat;
}

/*
 *  Copy CSR matrix from host to device.
 *  The device memory should be pre-allocated.
 */
void csr_copy_h2d(csr           *dev_mat,
                  const c_int   *h_row_ptr,
                  const c_int   *h_col_ind,
                  const c_float *h_val) {

  checkCudaErrors(cudaMemcpy(dev_mat->row_ptr, h_row_ptr, (dev_mat->m + 1) * sizeof(c_int), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(dev_mat->col_ind, h_col_ind, dev_mat->nnz * sizeof(c_int), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(dev_mat->val, h_val, dev_mat->nnz * sizeof(c_float), cudaMemcpyHostToDevice));
}

csr* csr_init(c_int          m,
              c_int          n,
              const c_int   *h_row_ptr,
              const c_int   *h_col_ind,
              const c_float *h_val) {
    
  csr *dev_mat = csr_alloc(m, n, h_row_ptr[m], 1);
  
  if (!dev_mat) return NULL;
  
  if (m == 0) return dev_mat;

  /* copy_matrix_to_device */
  csr_copy_h2d(dev_mat, h_row_ptr, h_col_ind, h_val);
  update_mp_buffer(dev_mat);

  return dev_mat;
}


/*******************************************************************************
 *                           API Functions                                     *
 *******************************************************************************/

void cuda_mat_init_P(const csc  *mat,
                     csr       **P,
                     c_float   **d_P_triu_val,
                     c_int     **d_P_triu_to_full_ind,
                     c_int     **d_P_diag_ind) {

  c_int n   = mat->n;
  c_int m   = mat->m;
  c_int nnz = mat->p[n];
  
  /* Initialize upper triangular part of P */
  *P = csr_init(n, n, mat->p, mat->i, mat->x);

  /* Convert P to a full matrix. Store indices of diagonal and triu elements. */
  csr_triu_to_full(*P, d_P_triu_to_full_ind, d_P_diag_ind);
  csr_expand_row_ind(*P);

  /* We need 0.0 at val[nzz] -> nnz+1 elements */
  checkCudaErrors(cuda_calloc(d_P_triu_val, (nnz+1) * sizeof(c_float)));

  /* Store triu elements */
  checkCudaErrors(cudaMemcpy(*d_P_triu_val, mat->x, nnz * sizeof(c_float), cudaMemcpyHostToDevice));
}
