#ifndef CSC_MATRIX_H
# define CSC_MATRIX_H

# ifdef __cplusplus
extern "C" {
# endif // ifdef __cplusplus

# include "types.h"   // CSC matrix type
# include "lin_alg.h" // Vector copy operations

/**
 *  Matrix in compressed-column or triplet form.  The same structure
 *  is used for csc, csr and sparse triplet form
 */
struct SparseMatrix_ {
  c_int    nzmax; ///< maximum number of entries.
  c_int    m;     ///< number of rows
  c_int    n;     ///< number of columns
  c_int   *p;     ///< column or row pointers (size n+1) (col indices (size nzmax)
                  // start from 0 when using triplet format (direct KKT matrix
                  // formation))
  c_int   *i;     ///< row indices, size nzmax starting from 0
  c_float *x;     ///< numerical values, size nzmax
  c_int    nnz;    ///< # of entries in triplet matrix, -1 for csc/csr
};

typedef struct SparseMatrix_ CscMatrix; // Compressed sparse column matrix

/*****************************************************************************
* Create and free CSC Matrices                                              *
*****************************************************************************/

/**
 * Create Compressed-Column-Sparse matrix from existing arrays
    (no MALLOC to create inner arrays x, i, p)
 * @param  m     First dimension
 * @param  n     Second dimension
 * @param  nzmax Maximum number of nonzero elements
 * @param  x     Vector of data
 * @param  i     Vector of row indices
 * @param  p     Vector of column pointers
 * @return       New matrix pointer
 */
CscMatrix* CscMatrix_matrix(c_int    m,
                c_int    n,
                c_int    nzmax,
                c_float *x,
                c_int   *i,
                c_int   *p);


/**
 * Create uninitialized CscMatrix matrix atricture
    (uses MALLOC to create inner arrays x, i, p)
 * @param  m       First dimension
 * @param  n       Second dimension
 * @param  nzmax   Maximum number of nonzero elements
 * @param  values  Allocate values (0/1)
 * @param  triplet Allocate CscMatrix or triplet format matrix (1/0)
 * @return         Matrix pointer
 */
CscMatrix* CscMatrix_spalloc(c_int m,
                 c_int n,
                 c_int nzmax,
                 c_int values,
                 c_int triplet);


/**
 * Free sparse matrix
    (uses FREE to free inner arrays x, i, p)
 * @param  A Matrix in CscMatrix format
 */
void CscMatrix_spfree(CscMatrix *A);


/**
 * free workspace and return a sparse matrix result
 * @param  C  CscMatrix matrix
 * @param  w  Workspace vector
 * @param  x  Workspace vector
 * @param  ok flag
 * @return    Return result C if OK, otherwise free it
 */
CscMatrix* CscMatrix_done(CscMatrix  *C,
              void *w,
              void *x,
              c_int ok);

/*****************************************************************************
* Copy Matrices                                                             *
*****************************************************************************/

/**
 *  Copy sparse CscMatrix matrix A to output.
 *  output is allocated by this function (uses MALLOC)
 */
CscMatrix* copy_CscMatrix_mat(const CscMatrix *A);


/**
 *  Copy sparse CscMatrix matrix A to B (B is preallocated, NO MALOC)
 */
void prea_copy_CscMatrix_mat(const CscMatrix *A,
                       CscMatrix       *B);


/*****************************************************************************
* Matrices Conversion                                                       *
*****************************************************************************/


/**
 * C = compressed-column CscMatrix from matrix T in triplet form
 *
 * TtoC stores the vector of indices from T to C
 *  -> C[TtoC[i]] = T[i]
 *
 * @param  T    matrix in triplet format
 * @param  TtoC vector of indices from triplet to CscMatrix format
 * @return      matrix in CscMatrix format
 */
CscMatrix* triplet_to_CscMatrix(const CscMatrix *T,
                    c_int     *TtoC);


/**
 * C = compressed-row CSR from matrix T in triplet form
 *
 * TtoC stores the vector of indices from T to C
 *  -> C[TtoC[i]] = T[i]
 *
 * @param  T    matrix in triplet format
 * @param  TtoC vector of indices from triplet to CSR format
 * @return      matrix in CSR format
 */
CscMatrix* triplet_to_csr(const CscMatrix *T,
                    c_int     *TtoC);


/**
 * Convert sparse to dense
 */
c_float* CscMatrix_to_dns(CscMatrix *M);


/**
 * Convert square CscMatrix matrix into upper triangular one
 *
 * @param  M         Matrix to be converted
 * @return           Upper triangular matrix in CscMatrix format
 */
CscMatrix* CscMatrix_to_triu(CscMatrix *M);


/*****************************************************************************
* Helper operations                                                          *
*****************************************************************************/

/**
 * p [0..n] = cumulative sum of c [0..n-1], and then copy p [0..n-1] into c
 *
 * @param  p Create cumulative sum into p
 * @param  c Vector of which we compute cumulative sum
 * @param  n Number of elements
 * @return   Exitflag
 */
c_int CscMatrix_cumsum(c_int *p,
                 c_int *c,
                 c_int  n);

/**
 * Compute inverse of permutation matrix stored in the vector p.
 * The computed inverse is also stored in a vector.
 */
c_int* CscMatrix_pinv(c_int const *p,
                c_int        n);

/**
 * C = A(p,p)= PAP' where A and C are symmetric the upper part stored;
 *  NB: pinv not p!
 * @param  A      Original matrix (upper-triangular)
 * @param  pinv   Inverse of permutation vector
 * @param  AtoC   Mapping from indices of A-x to C->x
 * @param  values Are values of A allocated?
 * @return        New matrix (allocated)
 */
CscMatrix* CscMatrix_symperm(const CscMatrix   *A,
                 const c_int *pinv,
                 c_int       *AtoC,
                 c_int        values);



/*****************************************************************************
* matrix algebra                                         ******************************************************************************/

//DEBUG : ADD documentation

/* matrix times scalar */

void CscMatrix_scale(CscMatrix* A, c_float sc);

void CscMatrix_lmult_diag(CscMatrix* A, const OSQPVectorf *L);

void OSQPMatrix_scale(OSQPMatrix* A, c_float sc);

void CscMatrix_rmult_diag(CscMatrix* A, const OSQPVectorf *R);

//y = alpha*A*x + beta*y
void CscMatrix_Axpy(const CscMatrix   *A,
                    const OSQPVectorf *x,
                    OSQPVectorf *y,
                    c_float alpha,
                    c_float beta);


void CscMatrix_Atxpy(const CscMatrix *A,
                     const OSQPVectorf *x,
                     OSQPVectorf *y,
                     c_float alpha,
                     c_float beta,
                     c_int skip_diag);

c_float CscMatrix_quad_form(const CscMatrix *P, const OSQPVectorf *x);


#if EMBEDDED != 1

void CscMatrix_col_norm_inf(const CscMatrix *M, OSQPVectorf *E);

void CscMatrix_row_norm_inf(const CscMatrix *M, OSQPVectorf *E);

void CscMatrix_row_norm_inf_sym_triu(const CscMatrix *M, OSQPVectorf *E);

#endif /* if EMBEDDED != 1 */


# ifdef __cplusplus
}
# endif // ifdef __cplusplus

#endif // ifndef CSC_MATRIX
