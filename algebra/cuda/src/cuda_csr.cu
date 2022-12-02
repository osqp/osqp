/**
 *  Copyright (c) 2019-2021 ETH Zurich, Automatic Control Lab,
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

#include "cuda_csr.h"
#include "cuda_configure.h"
#include "cuda_handler.h"
#include "cuda_lin_alg.h"   /* --> cuda_vec_gather */
#include "cuda_malloc.h"
#include "helper_cuda.h"    /* --> checkCudaErrors */

#include "csr_type.h"
#include "glob_opts.h"

#include <thrust/scan.h>
#include <thrust/execution_policy.h>

extern CUDA_Handle_t *CUDA_handle;

/* This function is implemented in cuda_lin_alg.cu */
extern void scatter(OSQPFloat *out, const OSQPFloat *in, const OSQPInt *ind, OSQPInt n);


/*******************************************************************************
 *                            GPU Kernels                                      *
 *******************************************************************************/

 /*
 * Expand an upper triangular matrix given in COO format to a symmetric
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
__global__ void fill_full_matrix_kernel(OSQPInt*       row_ind_out,
                                        OSQPInt*       col_ind_out,
                                        OSQPInt*       nnz_on_diag,
                                        OSQPInt*       has_non_zero_diag_element,
                                        const OSQPInt* __restrict__ row_ind_in,
                                        const OSQPInt* __restrict__ col_ind_in,
                                        OSQPInt        nnz,
                                        OSQPInt        n) {

  OSQPInt idx = threadIdx.x + blockDim.x * blockIdx.x;
  OSQPInt grid_size = blockDim.x * gridDim.x;

  for(OSQPInt i = idx; i < nnz; i += grid_size) {
    OSQPInt row = row_ind_in[i];
    OSQPInt column = col_ind_in[i];

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
__global__ void add_diagonal_kernel(OSQPInt*       row_ind,
                                    OSQPInt*       col_ind,
                                    const OSQPInt* has_non_zero_diag_element,
                                    OSQPInt        n) {

  OSQPInt idx = threadIdx.x + blockDim.x * blockIdx.x;
  OSQPInt grid_size = blockDim.x * gridDim.x;

  for(OSQPInt row = idx; row < n; row += grid_size) {
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
__global__ void reduce_permutation_kernel(OSQPInt* permutation,
                                          OSQPInt  new_range,
                                          OSQPInt  n) {

  OSQPInt idx = threadIdx.x + blockDim.x * blockIdx.x;
  OSQPInt grid_size = blockDim.x * gridDim.x;

  for(OSQPInt i = idx; i < n; i += grid_size) {
    if (permutation[i] < 2 * new_range) {
      permutation[i] = permutation[i] % new_range;
    }
    else {
      permutation[i] = new_range; /* gets the 0 element at nnz+1 of the value array */
    }
  }
}

__global__ void get_diagonal_indices_kernel(OSQPInt* row_ind,
                                            OSQPInt* col_ind,
                                            OSQPInt  nnz,
                                            OSQPInt* diag_index) {

  OSQPInt idx = threadIdx.x + blockDim.x * blockIdx.x;
  OSQPInt grid_size = blockDim.x * gridDim.x;

  for (OSQPInt index = idx; index < nnz; index += grid_size) {
    OSQPInt row = row_ind[index];
    OSQPInt column = col_ind[index];

    if (row == column) {
      diag_index[row] = index;
    }
  }
}

__global__ void predicate_generator_kernel(const OSQPInt* row_ind,
                                           const OSQPInt* row_predicate,
                                                 OSQPInt* predicate,
                                                 OSQPInt  nnz) {

  OSQPInt idx = threadIdx.x + blockDim.x * blockIdx.x;
  OSQPInt grid_stride = gridDim.x * blockDim.x;

  for(OSQPInt i = idx; i < nnz; i += grid_stride) {
    OSQPInt row = row_ind[i];
    predicate[i] = row_predicate[row];
  }
}

template<typename T>
__global__ void compact(const T*       data_in,
                              T*       data_out,
                              OSQPInt* predicate,
                              OSQPInt* scatter_addres,
                              OSQPInt  n) {

  int idx = threadIdx.x + blockDim.x * blockIdx.x;

  if(idx < n) {
    if(predicate[idx]) {
      int write_ind = scatter_addres[idx] - 1;
      data_out[write_ind] = data_in[idx];
    }
  }
}

__global__ void compact_rows(const OSQPInt* row_ind,
                                   OSQPInt* data_out,
                                   OSQPInt* new_row_number,
                                   OSQPInt* predicate,
                                   OSQPInt* scatter_addres,
                                   OSQPInt  n) {

  int idx = threadIdx.x + blockDim.x * blockIdx.x;

  if(idx < n) {
    if(predicate[idx]) {
      OSQPInt write_ind = scatter_addres[idx] - 1;
      OSQPInt row = row_ind[idx];
      data_out[write_ind] = new_row_number[row]-1;
    }
  }
}

__global__ void vector_init_abs_kernel(const OSQPInt* a,
                                             OSQPInt* b,
                                             OSQPInt  n) {

  OSQPInt i  = threadIdx.x + blockDim.x * blockIdx.x;

  if (i < n) {
    b[i] = abs(a[i]);
  }
}

__global__ void csr_eq_kernel(const OSQPInt*   A_row_ptr,
                              const OSQPInt*   A_col_ind,
                              const OSQPFloat* A_val,
                              const OSQPInt*   B_row_ptr,
                              const OSQPInt*   B_col_ind,
                              const OSQPFloat* B_val,
                                    OSQPInt    m,
                                    OSQPFloat  tol,
                                    OSQPInt*   res) {
  OSQPInt i = 0;
  OSQPInt j = 0;
  OSQPFloat diff = 0.0;

  *res = 1;

  for (j = 0; j < m; j++) { // Cycle over rows j
    // if row pointer of next row does not coincide, they are not equal
    // NB: first row always has A->p[0] = B->p[0] = 0 by construction.
    if (A_row_ptr[j+1] != B_row_ptr[j+1]) {
        *res = 0;
        return;
    }

    for (i = A_row_ptr[j]; i < A_row_ptr[j + 1]; i++) { // Cycle columns i in row j
      if (A_col_ind[i] != B_col_ind[i]) {   // Different column indices
        *res = 0;
        return;
      }

#ifdef OSQP_USE_FLOAT
      diff = fabsf(A_val[i] - B_val[i]);
#else
      diff = fabs(A_val[i] - B_val[i]);
#endif

      if (diff > tol) {  // The actual matrix values are different
        *res = 0;
        return;
      }
    }
  }
}


/*******************************************************************************
 *                         Private Functions                                   *
 *******************************************************************************/

static void init_SpMV_interface(csr *M) {

  OSQPFloat* d_x;
  OSQPFloat* d_y;
  cusparseDnVecDescr_t vecx, vecy;

  OSQPFloat alpha = 1.0;
  OSQPInt   m = M->m;
  OSQPInt   n = M->n;

  /* Only create the matrix if it has non-zero dimensions.
   * Some versions of CUDA don't allow creating matrices with rows/columns of
   * size 0 and assert instead. So we don't create the matrix object, and instead
   * will never perform any operations on it.
   */
  if ((m > 0) && (n > 0)) {
    /* Wrap raw data into cuSPARSE API matrix */
    checkCudaErrors(cusparseCreateCsr(
      &M->SpMatDescr, m, n, M->nnz,
      (void*)M->row_ptr, (void*)M->col_ind, (void*)M->val,
      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
      CUSPARSE_INDEX_BASE_ZERO, CUDA_FLOAT));

    if (!M->SpMatBufferSize) {
      cuda_malloc((void **) &d_x, n * sizeof(OSQPFloat));
      cuda_malloc((void **) &d_y, m * sizeof(OSQPFloat));

      cuda_vec_create(&vecx, d_x, n);
      cuda_vec_create(&vecy, d_y, m);

      /* Allocate workspace for cusparseSpMV */
      checkCudaErrors(cusparseSpMV_bufferSize(
        CUDA_handle->cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, M->SpMatDescr, vecx, &alpha, vecy,
        CUDA_FLOAT, CUSPARSE_SPMV_ALG_DEFAULT, &M->SpMatBufferSize));

      if (M->SpMatBufferSize)
        cuda_malloc((void **) &M->SpMatBuffer, M->SpMatBufferSize);

      cuda_vec_destroy(vecx);
      cuda_vec_destroy(vecy);

      cuda_free((void **) &d_x);
      cuda_free((void **) &d_y);
    }
  }
}

 /*
 *  Creates a CSR matrix with the specified dimension (m,n,nnz).
 *  
 *  If specified, it allocates proper amount of device memory
 *  allocate_on_device = 1: device memory for CSR
 *  allocate_on_device = 2: device memory for CSR (+ col_ind)  
 */
csr* csr_alloc(OSQPInt m,
               OSQPInt n,
               OSQPInt nnz,
               OSQPInt allocate_on_device) {

  csr* dev_mat = (csr*) c_calloc(1, sizeof(csr));

  if (!dev_mat) return NULL;

  dev_mat->m   = m;
  dev_mat->n   = n;
  dev_mat->nnz = nnz;

  if (allocate_on_device > 0) {
    cuda_calloc((void **) &dev_mat->val,     (dev_mat->nnz + 1) * sizeof(OSQPFloat));
    cuda_malloc((void **) &dev_mat->row_ptr, (dev_mat->m + 1) * sizeof(OSQPInt));
    cuda_malloc((void **) &dev_mat->col_ind, dev_mat->nnz * sizeof(OSQPInt));

    if (allocate_on_device > 1)
      cuda_malloc((void **) &dev_mat->row_ind, dev_mat->nnz * sizeof(OSQPInt));
  }

  dev_mat->SpMatBufferSize = 0;
  dev_mat->SpMatBuffer = NULL;

  return dev_mat;
}

csr* csr_init(OSQPInt          m,
              OSQPInt          n,
              const OSQPInt*   h_row_ptr,
              const OSQPInt*   h_col_ind,
              const OSQPFloat* h_val) {
    
  csr* dev_mat = csr_alloc(m, n, h_row_ptr[m], 1);
  
  if (!dev_mat) return NULL;
  
  if (m == 0) return dev_mat;

  /* copy_matrix_to_device */
  checkCudaErrors(cudaMemcpy(dev_mat->row_ptr, h_row_ptr, (dev_mat->m + 1) * sizeof(OSQPInt), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(dev_mat->col_ind, h_col_ind, dev_mat->nnz * sizeof(OSQPInt),     cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(dev_mat->val,     h_val,     dev_mat->nnz * sizeof(OSQPFloat),   cudaMemcpyHostToDevice));

  return dev_mat;
}

/*
 *  Compress row indices from the COO format to the row pointer
 *  of the CSR format.
 */
void compress_row_ind(csr* mat) {

  cuda_free((void** ) &mat->row_ptr);
  cuda_malloc((void** ) &mat->row_ptr, (mat->m + 1) * sizeof(OSQPFloat));
  checkCudaErrors(cusparseXcoo2csr(CUDA_handle->cusparseHandle, mat->row_ind, mat->nnz, mat->m, mat->row_ptr, CUSPARSE_INDEX_BASE_ZERO));
}

void csr_expand_row_ind(csr* mat) {

  if (!mat->row_ind) {
    cuda_malloc((void** ) &mat->row_ind, mat->nnz * sizeof(OSQPFloat));
    checkCudaErrors(cusparseXcsr2coo(CUDA_handle->cusparseHandle, mat->row_ptr, mat->nnz, mat->m, mat->row_ind, CUSPARSE_INDEX_BASE_ZERO));
  }
}

/*
 *  Sorts matrix in COO format by row. It returns a permutation
 *  vector that describes reordering of the elements.
 */
OSQPInt* coo_sort(csr* A) {

  OSQPInt* A_to_At_permutation;
  char*    pBuffer;
  size_t   pBufferSizeInBytes;

  cuda_malloc((void **) &A_to_At_permutation, A->nnz * sizeof(OSQPInt));
  checkCudaErrors(cusparseCreateIdentityPermutation(CUDA_handle->cusparseHandle, A->nnz, A_to_At_permutation));

  checkCudaErrors(cusparseXcoosort_bufferSizeExt(CUDA_handle->cusparseHandle, A->m, A->n, A->nnz, A->row_ind, A->col_ind, &pBufferSizeInBytes));

  cuda_malloc((void **) &pBuffer, pBufferSizeInBytes * sizeof(char));

  checkCudaErrors(cusparseXcoosortByRow(CUDA_handle->cusparseHandle, A->m, A->n, A->nnz, A->row_ind, A->col_ind, A_to_At_permutation, pBuffer));

  cuda_free((void **) &pBuffer);

  return A_to_At_permutation;
}

/*
 * Compute transpose of a matrix in COO format.
 */
void coo_tranpose(csr* A) {
  OSQPInt m = A->m;
  A->m = A->n;
  A->n = m;

  OSQPInt *row_ind = A->row_ind;
  A->row_ind = A->col_ind;
  A->col_ind = row_ind;
}

/*
 *  values[i] = values[permutation[i]] for i in [0,n-1]
 */
void permute_vector(OSQPFloat*     values,
                    const OSQPInt* permutation,
                    OSQPInt        n) {

  OSQPFloat* permuted_values;
  cuda_malloc((void **) &permuted_values, n * sizeof(OSQPFloat));

  cuda_vec_gather(n, values, permuted_values, permutation);

  checkCudaErrors(cudaMemcpy(values, permuted_values, n * sizeof(OSQPFloat), cudaMemcpyDeviceToDevice));
  cuda_free((void **) &permuted_values);
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

  target->m   = source->m;
  target->n   = source->n;
  target->nnz = source->nnz;

  cuda_free((void **) &target->val);
  cuda_free((void **) &target->row_ind);
  cuda_free((void **) &target->row_ptr);
  cuda_free((void **) &target->col_ind);

  target->val     = source->val;
  target->row_ind = source->row_ind;
  target->row_ptr = source->row_ptr;
  target->col_ind = source->col_ind;

  source->val     = NULL;
  source->row_ind = NULL;
  source->row_ptr = NULL;
  source->col_ind = NULL;
}

void csr_triu_to_full(csr*      P_triu,
                      OSQPInt** P_triu_to_full_permutation,
                      OSQPInt** P_diag_indices) {

  OSQPInt number_of_blocks;
  OSQPInt* has_non_zero_diag_element;
  OSQPInt* d_nnz_diag;
  OSQPInt h_nnz_diag, Full_nnz, nnz_triu, n, nnz_max_Full;
  OSQPInt offset;

  nnz_triu     = P_triu->nnz;
  n            = P_triu->n;
  nnz_max_Full = 2*nnz_triu + n;

  csr* Full_P = csr_alloc(n, n, nnz_max_Full, 2);
  cuda_calloc((void **) &has_non_zero_diag_element, n * sizeof(OSQPInt));
  cuda_calloc((void **) &d_nnz_diag, sizeof(OSQPInt));

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
  
  checkCudaErrors(cudaMemcpy(&h_nnz_diag, d_nnz_diag, sizeof(OSQPInt), cudaMemcpyDeviceToHost));

  Full_nnz = (2 * (nnz_triu - h_nnz_diag)) + n;
  OSQPInt* d_P = coo_sort(Full_P);

  number_of_blocks = (nnz_triu / THREADS_PER_BLOCK) + 1;
  reduce_permutation_kernel<<<number_of_blocks,THREADS_PER_BLOCK>>>(d_P, nnz_triu, Full_nnz);

  /* permute vector */
  cuda_vec_gather(Full_nnz, P_triu->val, Full_P->val, d_P);

  cuda_malloc((void **) P_triu_to_full_permutation, Full_nnz * sizeof(OSQPInt));
  checkCudaErrors(cudaMemcpy(*P_triu_to_full_permutation, d_P, Full_nnz * sizeof(OSQPInt), cudaMemcpyDeviceToDevice));
  cuda_malloc((void **) P_diag_indices, n * sizeof(OSQPInt));

  number_of_blocks = (Full_nnz / THREADS_PER_BLOCK) + 1;
  get_diagonal_indices_kernel<<<number_of_blocks, THREADS_PER_BLOCK>>>(Full_P->row_ind, Full_P->col_ind, Full_nnz, *P_diag_indices);

  Full_P->nnz = Full_nnz;
  compress_row_ind(Full_P);
  copy_csr(P_triu, Full_P);

  cuda_mat_free(Full_P);
  cuda_free((void **) &d_P);
  cuda_free((void **) &d_nnz_diag);
  cuda_free((void **) &has_non_zero_diag_element);
}

/**
 * Matrix A is converted from CSC to CSR. The data in A is interpreted as
 * being in CSC format, even if it is in CSR.
 * This operation is equivalent to a transpose. We temporarily allocate space
 * for the new matrix since this operation cannot be done inplace.
 * Additionally, a gather indices vector is generated to perform the conversion
 * from A to A' faster during a matrix update.
 */
void csr_transpose(csr*      A,
                   OSQPInt** A_to_At_permutation) {

  (*A_to_At_permutation) = NULL;

  if (A->nnz == 0) {
    OSQPInt tmp = A->n;
    A->n = A->m;
    A->m = tmp;
    return;
  }

  csr_expand_row_ind(A);
  coo_tranpose(A);
  (*A_to_At_permutation) = coo_sort(A);
  compress_row_ind(A);

  permute_vector(A->val, *A_to_At_permutation, A->nnz);
}


/*******************************************************************************
 *                           API Functions                                     *
 *******************************************************************************/

void cuda_mat_init_P(const OSQPCscMatrix* mat,
                           csr**          P,
                           OSQPFloat**    d_P_triu_val,
                           OSQPInt**      d_P_triu_to_full_ind,
                           OSQPInt**      d_P_diag_ind) {

  OSQPInt n   = mat->n;
  OSQPInt nnz = mat->p[n];
  
  /* Initialize upper triangular part of P */
  *P = csr_init(n, n, mat->p, mat->i, mat->x);

  /* Convert P to a full matrix. Store indices of diagonal and triu elements. */
  csr_triu_to_full(*P, d_P_triu_to_full_ind, d_P_diag_ind);
  csr_expand_row_ind(*P);

  /* We need 0.0 at val[nzz] -> nnz+1 elements */
  cuda_calloc((void **) d_P_triu_val, (nnz+1) * sizeof(OSQPFloat));

  /* Store triu elements */
  checkCudaErrors(cudaMemcpy(*d_P_triu_val, mat->x, nnz * sizeof(OSQPFloat), cudaMemcpyHostToDevice));

  init_SpMV_interface(*P);
}

void cuda_mat_init_A(const OSQPCscMatrix* mat,
                           csr**          A,
                           csr**          At,
                           OSQPInt**      d_A_to_At_ind) {

  OSQPInt m = mat->m;
  OSQPInt n = mat->n;

  /* Initializing At is easy since it is equal to A in CSC */
  *At = csr_init(n, m, mat->p, mat->i, mat->x);
  csr_expand_row_ind(*At);

  /* We need to take transpose of At to get A */
  *A = csr_init(n, m, mat->p, mat->i, mat->x);
  csr_transpose(*A, d_A_to_At_ind);
  csr_expand_row_ind(*A);

  init_SpMV_interface(*A);
  init_SpMV_interface(*At);
}

void cuda_mat_update_P(const OSQPFloat*  Px,
                       const OSQPInt*    Px_idx,
                             OSQPInt     Px_n,
                             csr**       P,
                             OSQPFloat*  d_P_triu_val,
                             OSQPInt*    d_P_triu_to_full_ind,
                             OSQPInt*    d_P_diag_ind,
                             OSQPInt     P_triu_nnz) {

  if (!Px_idx) { /* Update whole P */
    OSQPFloat* d_P_val_new;

    /* Allocate memory */
    cuda_malloc((void **) &d_P_val_new, (P_triu_nnz + 1) * sizeof(OSQPFloat));

    /* Copy new values from host to device */
    checkCudaErrors(cudaMemcpy(d_P_val_new, Px, P_triu_nnz * sizeof(OSQPFloat), cudaMemcpyHostToDevice));

    cuda_vec_gather((*P)->nnz, d_P_val_new, (*P)->val, d_P_triu_to_full_ind);

    cuda_free((void **) &d_P_val_new);
  }
  else { /* Update P partially */
    OSQPFloat* d_P_val_new;
    OSQPInt*   d_P_ind_new;

    /* Allocate memory */
    cuda_malloc((void **) &d_P_val_new, Px_n * sizeof(OSQPFloat));
    cuda_malloc((void **) &d_P_ind_new, Px_n * sizeof(OSQPInt));

    /* Copy new values and indices from host to device */
    checkCudaErrors(cudaMemcpy(d_P_val_new, Px,     Px_n * sizeof(OSQPFloat), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_P_ind_new, Px_idx, Px_n * sizeof(OSQPInt),   cudaMemcpyHostToDevice));

    /* Update d_P_triu_val */
    scatter(d_P_triu_val, d_P_val_new, d_P_ind_new, Px_n);

    /* Gather from d_P_triu_val to update full P */
    cuda_vec_gather((*P)->nnz, d_P_triu_val, (*P)->val, d_P_triu_to_full_ind);

    cuda_free((void **) &d_P_val_new);
    cuda_free((void **) &d_P_ind_new);
  }
}

void cuda_mat_update_A(const OSQPFloat* Ax,
                       const OSQPInt*   Ax_idx,
                             OSQPInt    Ax_n,
                             csr**      A,
                             csr**      At,
                             OSQPInt*   d_A_to_At_ind) {

  OSQPInt    Annz  = (*A)->nnz;
  OSQPFloat* Aval  = (*A)->val;
  OSQPFloat* Atval = (*At)->val;

  if (!Ax_idx) { /* Update whole A */
    /* Updating At is easy since it is equal to A in CSC */
    checkCudaErrors(cudaMemcpy(Atval, Ax, Annz * sizeof(OSQPFloat), cudaMemcpyHostToDevice));

    /* Updating A requires transpose of A_new */
    cuda_vec_gather(Annz, Atval, Aval, d_A_to_At_ind);
  }
  else { /* Update A partially */
    OSQPFloat* d_At_val_new;
    OSQPInt*   d_At_ind_new;

    /* Allocate memory */
    cuda_malloc((void **) &d_At_val_new, Ax_n * sizeof(OSQPFloat));
    cuda_malloc((void **) &d_At_ind_new, Ax_n * sizeof(OSQPInt));

    /* Copy new values and indices from host to device */
    checkCudaErrors(cudaMemcpy(d_At_val_new, Ax,     Ax_n * sizeof(OSQPFloat), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_At_ind_new, Ax_idx, Ax_n * sizeof(OSQPInt),   cudaMemcpyHostToDevice));

    /* Update At first since it is equal to A in CSC */
    scatter(Atval, d_At_val_new, d_At_ind_new, Ax_n);

    cuda_free((void **) &d_At_val_new);
    cuda_free((void **) &d_At_ind_new);

    /* Gather from Atval to construct Aval */
    cuda_vec_gather(Annz, Atval, Aval, d_A_to_At_ind);
  }
}

void cuda_mat_free(csr* mat) {
  if (mat) {
    cuda_free((void **) &mat->val);
    cuda_free((void **) &mat->row_ptr);
    cuda_free((void **) &mat->col_ind);
    cuda_free((void **) &mat->row_ind);

    cuda_free((void **) &mat->SpMatBuffer);
    checkCudaErrors(cusparseDestroySpMat(mat->SpMatDescr));

    c_free(mat);
  }
}

OSQPInt cuda_csr_is_eq(const csr*      A,
                       const csr*      B,
                             OSQPFloat tol) {

  OSQPInt h_res = 0;
  OSQPInt *d_res;

  // If number of columns, rows and non-zeros are not the same, they are not equal.
  if ((A->n != B->n) || (A->m != B->m) || (A->nnz != B->nnz)) {
      return 0;
  }

  OSQPInt nnz = A->nnz;
  OSQPInt number_of_blocks = (nnz / THREADS_PER_BLOCK) / ELEMENTS_PER_THREAD + 1;


  cuda_malloc((void **) &d_res, sizeof(OSQPInt));

  /* Initialize d_res to 1 */
  h_res = 1;
  checkCudaErrors(cudaMemcpy(d_res, &h_res, sizeof(OSQPInt), cudaMemcpyHostToDevice));

  csr_eq_kernel<<<number_of_blocks, THREADS_PER_BLOCK>>>(A->row_ptr, A->col_ind, A->val,
                                                         B->row_ptr, B->col_ind, B->val,
                                                         A->m, tol, d_res);

  checkCudaErrors(cudaMemcpy(&h_res, d_res, sizeof(OSQPInt), cudaMemcpyDeviceToHost));

  cuda_free((void **) &d_res);

  return h_res;
}

void cuda_submat_byrows(const csr* A,
                        const OSQPInt* d_rows,
                              csr**    Ared,
                              csr**    Aredt) {

  OSQPInt new_m = 0;

  OSQPInt n   = A->n;
  OSQPInt m   = A->m;
  OSQPInt nnz = A->nnz;

  OSQPInt* d_predicate;
  OSQPInt* d_compact_address;
  OSQPInt* d_row_predicate;
  OSQPInt* d_new_row_number;

  cuda_malloc((void **) &d_row_predicate,  m * sizeof(OSQPInt));
  cuda_malloc((void **) &d_new_row_number, m * sizeof(OSQPInt));

  cuda_malloc((void **) &d_predicate,       nnz * sizeof(OSQPInt));
  cuda_malloc((void **) &d_compact_address, nnz * sizeof(OSQPInt));

  // Copy rows array to device and set -1s to ones
  checkCudaErrors(cudaMemcpy(d_row_predicate, d_rows, m * sizeof(OSQPInt), cudaMemcpyDeviceToDevice));
  vector_init_abs_kernel<<<(m/THREADS_PER_BLOCK) + 1,THREADS_PER_BLOCK>>>(d_row_predicate, d_row_predicate, m);

  // Calculate new row numbering and get new number of rows
  thrust::inclusive_scan(thrust::device, d_row_predicate, d_row_predicate + m, d_new_row_number);
  if (m) {
    checkCudaErrors(cudaMemcpy(&new_m, &d_new_row_number[m-1], sizeof(OSQPInt), cudaMemcpyDeviceToHost));
  }
  else {
    (*Ared) = (csr *) c_calloc(1, sizeof(csr));
    (*Ared)->n = n;

    (*Aredt) = (csr *) c_calloc(1, sizeof(csr));
    (*Aredt)->m = n;

    return;
  }

  // Generate predicates per element from per row predicate
  predicate_generator_kernel<<<(nnz/THREADS_PER_BLOCK) + 1, THREADS_PER_BLOCK>>>(A->row_ind, d_row_predicate, d_predicate, nnz);

  // Get array offset for compacting and new nnz
  thrust::inclusive_scan(thrust::device, d_predicate, d_predicate + nnz, d_compact_address);
  OSQPInt nnz_new;
  if (nnz) checkCudaErrors(cudaMemcpy(&nnz_new, &d_compact_address[nnz-1], sizeof(OSQPInt), cudaMemcpyDeviceToHost));

  // allocate new matrix (2 -> allocate row indices as well)
  (*Ared) = csr_alloc(new_m, n, nnz_new, 2);

  // Compact arrays according to given predicates, special care has to be taken for the rows
  compact_rows<<<(nnz/THREADS_PER_BLOCK) + 1, THREADS_PER_BLOCK>>>(A->row_ind, (*Ared)->row_ind, d_new_row_number, d_predicate, d_compact_address, nnz);
  compact<<<(nnz/THREADS_PER_BLOCK) + 1, THREADS_PER_BLOCK>>>(A->col_ind, (*Ared)->col_ind, d_predicate, d_compact_address, nnz);
  compact<<<(nnz/THREADS_PER_BLOCK) + 1, THREADS_PER_BLOCK>>>(A->val, (*Ared)->val, d_predicate, d_compact_address, nnz);

  // Generate row pointer
  compress_row_ind(*Ared);

  // We first make a copy of Ared
  *Aredt = csr_alloc(new_m, n, nnz_new, 1);
  checkCudaErrors(cudaMemcpy((*Aredt)->val,     (*Ared)->val,     nnz_new   * sizeof(OSQPFloat), cudaMemcpyDeviceToDevice));
  checkCudaErrors(cudaMemcpy((*Aredt)->row_ptr, (*Ared)->row_ptr, (new_m+1) * sizeof(OSQPInt),   cudaMemcpyDeviceToDevice));
  checkCudaErrors(cudaMemcpy((*Aredt)->col_ind, (*Ared)->col_ind, nnz_new   * sizeof(OSQPInt),   cudaMemcpyDeviceToDevice));

  OSQPInt* d_A_to_At_ind;
  csr_transpose(*Aredt, &d_A_to_At_ind);

  csr_expand_row_ind(*Ared);
  csr_expand_row_ind(*Aredt);

  init_SpMV_interface(*Ared);
  init_SpMV_interface(*Aredt);

  cuda_free((void**)&d_A_to_At_ind);
  cuda_free((void**)&d_predicate);
  cuda_free((void**)&d_compact_address);
  cuda_free((void**)&d_row_predicate);
  cuda_free((void**)&d_new_row_number);
}

