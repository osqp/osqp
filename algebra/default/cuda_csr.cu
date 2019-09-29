#include "cuda_csr.h"
#include "cuda_handler.h"
#include "cuda_malloc.h"
#include "cuda_wrapper.h"
#include "helper_cuda.h"    /* --> checkCudaErrors */

#include "glob_opts.h"


extern CUDA_Handle_t *CUDA_handle;


/*******************************************************************************
 *                            GPU Kernels                                      *
 *******************************************************************************/

 /*
 * Expand an upper triangular matrix given in COO format to a symetric
 * matrix. Each entry is duplicated with its column- and row index switched.
 * In the case of a diagonal element we set the indices to a value  that is
 * larger than n to easily remove it later. This is done to keep the memory
 * patern one to one (MAP operation).
 * 
 * Additionally, it adds additional n diagonal elements to have a full 
 * diagonal.
 * 
 * The output arrays row_ind_out and col_ind_out have to be of size 2*nnz+n.
 */
__global__ void fill_full_matrix_kernel(c_int       *row_ind_out,
                                        c_int       *col_ind_out,
                                        c_int       *nnz_on_diag,
                                        c_int       *has_non_zero_diag_element,
                                        const c_int *__restrict__ row_ind_in,
                                        const c_int *__restrict__ col_ind_in,
                                        c_int        nnz,
                                        c_int        n) {

  c_int idx = threadIdx.x + blockDim.x * blockIdx.x;
  c_int grid_size = blockDim.x * gridDim.x;

  for(c_int i = idx; i < nnz; i += grid_size) {
    c_int row = row_ind_in[i];
    c_int column = col_ind_in[i];

    row_ind_out[i] = row;
    col_ind_out[i] = column;

    if (row == column) {
      has_non_zero_diag_element[row] = 1;
      row_ind_out[i + nnz] = column + n; /* dummy value for sorting and removal later on */
      col_ind_out[i + nnz] = row + n;
      atomicAdd(nnz_on_diag, 1);
    }
    else {
      row_ind_out[i + nnz] = column;
      col_ind_out[i + nnz] = row;
    }
  }
}

/**
 * Insert elements at structural zeros on the diagonal of the sparse matrix
 * specified by row and column index (COO format). To keep a one-to-one memory
 * patern we add n new elements to the matrix. In case where there already is a
 * diagonal element we add a dummy entry. The dummy entries will be removed later.
 */
__global__ void add_diagonal_kernel(c_int       *row_ind,
                                    c_int       *col_ind,
                                    const c_int *has_non_zero_diag_element,
                                    c_int        n) {

  c_int idx = threadIdx.x + blockDim.x * blockIdx.x;
  c_int grid_size = blockDim.x * gridDim.x;

  for(c_int row = idx; row < n; row += grid_size) {
    if (has_non_zero_diag_element[row] == 0) {
      row_ind[row] = row; 
      col_ind[row] = row;
    }
    else {
      row_ind[row] = row + n; /* dummy value, for easy removal after sorting */
      col_ind[row] = row + n;
    }
  }
}

/*
 * Permutation in: (size n, range 2*nnz+n):
 * 
 * Gathers from the following array to create the full matrix :
 * 
 *       |P_lower->val|P_lower->val|zeros(n)|
 *
 *       
 * Permutation out: (size n, range new_range)
 * 
 * Gathers from the following array to create the full matrix :
 * 
 *          |P_lower->val|zeros(1)|
 *                             
 *          | x[i] mod new_range    if x[i] <  2 * new_range
 * x[i] ->  | new_range             if x[i] >= 2 * new_range   
 * 
 */
__global__ void reduce_permutation_kernel(c_int *permutation,
                                          c_int  new_range,
                                          c_int  n) {

  c_int idx = threadIdx.x + blockDim.x * blockIdx.x;
  c_int grid_size = blockDim.x * gridDim.x;

  for(c_int i = idx; i < n; i += grid_size) {
    if (permutation[i] < 2 * new_range) {
      permutation[i] = permutation[i] % new_range;
    }
    else {
      permutation[i] = new_range; /* gets the 0 element at nnz+1 of the value array */
    }
  }
}

__global__ void get_diagonal_indices_kernel(c_int *row_ind,
                                            c_int *col_ind,
                                            c_int  nnz,
                                            c_int *diag_index) {

  c_int idx = threadIdx.x + blockDim.x * blockIdx.x;
  c_int grid_size = blockDim.x * gridDim.x;

  for (c_int index = idx; index < nnz; index += grid_size) {
    c_int row = row_ind[index];
    c_int column = col_ind[index];

    if (row == column) {
      diag_index[row] = index;
    }
  }
}


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

  checkCudaErrors(cusparseCsrmv_bufferSize(CUDA_handle->cusparseHandle,
                                           P->alg, P->m, P->n, P->nnz,
                                           &alpha,
                                           P->MatDescription, P->val, P->row_ptr, P->col_ind,
                                           NULL,
                                           &alpha,
                                           NULL,
                                           &bufferSizeInBytes));
  
  if (bufferSizeInBytes > P->bufferSizeInBytes) {
    cuda_free((void **) &P->buffer);                            
    cuda_malloc((void **) &P->buffer, bufferSizeInBytes);
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
    cuda_calloc((void **) &dev_mat->val, (dev_mat->nnz + 1) * sizeof(c_float));
    cuda_malloc((void **) &dev_mat->row_ptr, (dev_mat->m + 1) * sizeof(c_int)); 
    cuda_malloc((void **) &dev_mat->col_ind, dev_mat->nnz * sizeof(c_int));

    if (allocate_on_device > 1) {
      cuda_malloc((void **) &dev_mat->row_ind, dev_mat->nnz * sizeof(c_int));
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

/*
 *  Compress row indices from the COO format to the row pointer
 *  of the CSR format.
 */
void compress_row_ind(csr *mat) {

  cuda_free((void** ) &mat->row_ptr);
  cuda_malloc((void** ) &mat->row_ptr, (mat->m + 1) * sizeof(c_float));
  checkCudaErrors(cusparseXcoo2csr(CUDA_handle->cusparseHandle, mat->row_ind, mat->nnz, mat->m, mat->row_ptr, CUSPARSE_INDEX_BASE_ZERO));
}

void csr_expand_row_ind(csr *mat) {

  if (!mat->row_ind) {
    cuda_malloc((void** ) &mat->row_ind, mat->nnz * sizeof(c_float));
    checkCudaErrors(cusparseXcsr2coo(CUDA_handle->cusparseHandle, mat->row_ptr, mat->nnz, mat->m, mat->row_ind, CUSPARSE_INDEX_BASE_ZERO));
  }
}

/*
 *  Sorts matrix in COO format by row. It returns a permutation
 *  vector that describes reordering of the elements.
 */
c_int* sort_coo(csr *A) {

  c_int *A_to_At_permutation;  
  char *pBuffer;
  size_t pBufferSizeInBytes;

  checkCudaErrors(cuda_malloc((void **) &A_to_At_permutation, A->nnz * sizeof(c_int)));
  checkCudaErrors(cusparseCreateIdentityPermutation(CUDA_handle->cusparseHandle, A->nnz, A_to_At_permutation));

  checkCudaErrors(cusparseXcoosort_bufferSizeExt(CUDA_handle->cusparseHandle, A->m, A->n, A->nnz, A->row_ind, A->col_ind, &pBufferSizeInBytes));

  checkCudaErrors(cuda_malloc((void **) &pBuffer, pBufferSizeInBytes * sizeof(char)));

  checkCudaErrors(cusparseXcoosortByRow(CUDA_handle->cusparseHandle, A->m, A->n, A->nnz, A->row_ind, A->col_ind, A_to_At_permutation, pBuffer));

  cuda_free((void **) &pBuffer);

  return A_to_At_permutation;
}

/*
 *  Copy the values and pointers form target to the source matrix.
 *  The device memory of source has to be freed first to avoid a
 *  memory leak in case it holds allocated memory.
 *  
 *  The MatrixDescription has to be destroyed first since it is a
 *  pointer hidded by a typedef.
 *  
 *  The pointers of source matrix are set to NULL to avoid
 *  accidental freeing of the associated memory blocks.
 */
void copy_csr(csr* target,
              csr* source) {

  target->m                 = source->m;
  target->n                 = source->n;
  target->nnz               = source->nnz;
  target->bufferSizeInBytes = source->bufferSizeInBytes;
  target->alg               = source->alg;

  cusparseDestroyMatDescr(target->MatDescription);
  cuda_free((void **) &target->val);
  cuda_free((void **) &target->row_ind);
  cuda_free((void **) &target->row_ptr);
  cuda_free((void **) &target->col_ind);
  cuda_free((void **) &target->buffer);

  target->val            = source->val;
  target->row_ind        = source->row_ind;
  target->row_ptr        = source->row_ptr;
  target->col_ind        = source->col_ind;
  target->buffer         = source->buffer;
  target->MatDescription = source->MatDescription; 

  source->val            = NULL;
  source->row_ind        = NULL;
  source->row_ptr        = NULL;
  source->col_ind        = NULL;
  source->buffer         = NULL;
  source->MatDescription = NULL;
}

void csr_triu_to_full(csr    *P_triu,
                      c_int **P_triu_to_full_permutation,
                      c_int **P_diag_indices) {

  c_int number_of_blocks;
  c_int *has_non_zero_diag_element, *d_nnz_diag;
  c_int h_nnz_diag, Full_nnz, nnz_triu, n, nnz_max_Full;
  c_int offset;

  nnz_triu     = P_triu->nnz;
  n            = P_triu->n;
  nnz_max_Full = 2*nnz_triu + n;

  csr *Full_P = csr_alloc(n, n, nnz_max_Full, 2);
  checkCudaErrors(c_cudaCalloc(&has_non_zero_diag_element, n * sizeof(c_int)));
  checkCudaErrors(c_cudaCalloc(&d_nnz_diag, sizeof(c_int)));

  csr_expand_row_ind(P_triu);

  number_of_blocks = (nnz_triu / THREADS_PER_BLOCK) + 1;
  fill_full_matrix_kernel<<<number_of_blocks, THREADS_PER_BLOCK>>>(Full_P->row_ind, Full_P->col_ind, d_nnz_diag, has_non_zero_diag_element, P_triu->row_ind, P_triu->col_ind, nnz_triu, n);

  offset = 2 * nnz_triu;
  number_of_blocks = (n / THREADS_PER_BLOCK) + 1;
  add_diagonal_kernel<<<number_of_blocks, THREADS_PER_BLOCK>>>(Full_P->row_ind + offset, Full_P->col_ind + offset, has_non_zero_diag_element, n);

  /* The Full matrix now is of size (2n)x(2n)
    *                  [P 0]
    *                  [0 D]
    * where P is the desired full matrix and D is
    * a diagonal that contains dummy values
  */
  
  checkCudaErrors(cudaMemcpy(&h_nnz_diag, d_nnz_diag, sizeof(c_int), cudaMemcpyDeviceToHost));

  Full_nnz = (2 * (nnz_triu - h_nnz_diag)) + n;
  c_int *d_P = sort_coo(Full_P);

  number_of_blocks = (nnz_triu / THREADS_PER_BLOCK) + 1;
  reduce_permutation_kernel<<<number_of_blocks,THREADS_PER_BLOCK>>>(d_P, nnz_triu, Full_nnz);

  permute_vector(Full_P->val, P_triu->val, d_P, Full_nnz);


  checkCudaErrors(c_cudaMalloc(P_triu_to_full_permutation, Full_nnz * sizeof(c_int)));
  checkCudaErrors(cudaMemcpy(*P_triu_to_full_permutation, d_P, Full_nnz * sizeof(c_int), cudaMemcpyDeviceToDevice));
  checkCudaErrors(c_cudaMalloc(P_diag_indices, n * sizeof(c_int)));

  number_of_blocks = (Full_nnz / THREADS_PER_BLOCK) + 1;
  get_diagonal_indices_kernel<<<number_of_blocks, THREADS_PER_BLOCK>>>(Full_P->row_ind, Full_P->col_ind, Full_nnz, *P_diag_indices);

  Full_P->nnz = Full_nnz;
  compress_row_ind(Full_P);
  update_mp_buffer(Full_P); 
  copy_csr(P_triu, Full_P);

  cuda_mat_free(Full_P);
  cuda_free((void **) &d_P);
  cuda_free((void **) &d_nnz_diag);
  cuda_free((void **) &has_non_zero_diag_element);
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
  c_int nnz = mat->p[n];
  
  /* Initialize upper triangular part of P */
  *P = csr_init(n, n, mat->p, mat->i, mat->x);

  /* Convert P to a full matrix. Store indices of diagonal and triu elements. */
  csr_triu_to_full(*P, d_P_triu_to_full_ind, d_P_diag_ind);
  csr_expand_row_ind(*P);

  /* We need 0.0 at val[nzz] -> nnz+1 elements */
  cuda_calloc((void **) d_P_triu_val, (nnz+1) * sizeof(c_float));

  /* Store triu elements */
  checkCudaErrors(cudaMemcpy(*d_P_triu_val, mat->x, nnz * sizeof(c_float), cudaMemcpyHostToDevice));
}

void cuda_mat_free(csr *dev_mat) {
  if (dev_mat) {
    cuda_free((void **) &dev_mat->val);
    cuda_free((void **) &dev_mat->row_ptr);
    cuda_free((void **) &dev_mat->col_ind);
    cuda_free((void **) &dev_mat->buffer);
    cuda_free((void **) &dev_mat->row_ind);
    cusparseDestroyMatDescr(dev_mat->MatDescription);
    c_free(dev_mat);
  }
}
