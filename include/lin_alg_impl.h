#ifndef LIN_ALG_IMPL_H
# define LIN_ALG_IMPL_H

# ifdef __cplusplus
extern "C" {
# endif // ifdef __cplusplus

/***********************************
* Concrete Matrix and Vector Types *
***********************************/

/**
 *  An enum used to indicate whether a matrix stores
 *  real values or only the locations of non-zeros
 */
typedef enum OSQPMatrix_value_type {REAL,LOGICAL} SparseMatrix_value_type;

/**
 *  An enum used to indicate whether a matrix is symmetric.   Options
 *  NONE : matrix is not fully populated
 *  TRUI : matrix is symmetric and only upper triangle is stored
 *  TRIL : matrix is symmetric and only lower triangle is stored
 */
typedef enum OSQPMatrix_symmetry_type {NONE,TRIU,TRIL} OSQPMatrix_symmetry_type;

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
  c_int   nnz;    ///< # of entries in triplet matrix, -1 for csc/csr
  SparseMatrix_value_type    datatype; /// REAL or LOGICAL.  If Logical, then x = NULL
};

typedef struct SparseMatrix_ CscMatrix; // Compressed sparse column matrix
typedef struct SparseMatrix_ CsrMatrix; // Compressed sparse row matrix
typedef struct SparseMatrix_ TripletMatrix; // Sparse Triplet format matrix

typedef struct OSQPMatrix_ {
  CscMatrix* csc; //sparse column representation (NULL if unused)
  CsrMatrix* csr; //sparse row representation (NULL if unused)
  OSQPMatrix_symmetry_type symmetry; // NONE (if full)
                                     // TRIL or TRIU if symmetric and only
                                     // upper/lower triangle is stored
} OSQPMatrix;


typedef struct OSQPVectori_ {
  c_int* values;
  c_int length;
} OSQPVectori;

typedef struct OSQPVectorf_ {
  c_float* values;
  c_int length;
} OSQPVectorf;


/* MATRIX /VECTOR FUNCTIONS that allocate memory------------------------------*/

# ifndef EMBEDDED



/* MATRIX FUNCTIONS ----------------------------------------------------------*/

/* multiply matrix by scalar */
void CscMatrix_mult_scalar(CscMatrix *A, c_float sc);
void CsrMatrix_mult_scalar(CsrMatrix *A, c_float sc);

/* Premultiply (i.e. left) matrix A by diagonal matrix with diagonal d,
   i.e. scale the rows of A by d
 */
void CscMatrix_premult_diag(CscMatrix *A, const OSQPVectorf *d);
void CsrMatrix_premult_diag(CsrMatrix *A, const OSQPVectorf *d);

/* Postmultiply (i.e. right) matrix A by diagonal matrix with diagonal d,
   i.e. scale the columns of A by d
 */
void CscMatrix_postmult_diag(CscMatrix *A, const OSQPVectorf *d);
void CsrMatrix_postmult_diag(CsrMatrix *A, const OSQPVectorf *d);


/* Matrix-vector multiplication
 *    y  =  A*x  (if plus_eq == 0)
 *    y +=  A*x  (if plus_eq == 1)
 *    y -=  A*x  (if plus_eq == -1)
 */
void CscMatrix_Ax(const CscMatrix  *A,
                  const OSQPVectorf *x,
                  OSQPVectorf       *y,
                  c_int             sign,
                  c_int             skip_diag);
void CsrMatrix_Ax(const CsrMatrix  *A,
                  const OSQPVectorf *x,
                  OSQPVectorf       *y,
                  c_int             sign,
                  c_int             skip_diag);


/* Matrix-transpose-vector multiplication
 *    y  =  A'*x  (if plus_eq == 0)
 *    y +=  A'*x  (if plus_eq == 1)
 *    y -=  A'*x  (if plus_eq == -1)
 */
 void CscMatrix_Atx(const CscMatrix  *A,
                    const OSQPVectorf *x,
                    OSQPVectorf       *y,
                    c_int             sign,
                    c_int             skip_diag);
 void CsrMatrix_Atx(const CsrMatrix  *A,
                    const OSQPVectorf *x,
                    OSQPVectorf       *y,
                    c_int             sign,
                    c_int             skip_diag);


# if EMBEDDED != 1

/**
 * Infinity norm of each matrix column
 * @param M	Input matrix
 * @param E     Vector of infinity norms
 *
 */
void CscMatrix_inf_norm_cols(const CscMatrix *M,OSQPVectorf *E);
void CsrMatrix_inf_norm_cols(const CsrMatrix *M,OSQPVectorf *E);

/**
 * Infinity norm of each matrix row
 * @param M	Input matrix
 * @param E     Vector of infinity norms
 *
 */
void CscMatrix_inf_norm_rows(const CscMatrix *M,OSQPVectorf *E);
void CsrMatrix_inf_norm_rows(const CsrMatrix *M,OSQPVectorf *E);

# endif // EMBEDDED != 1

/**
 * Compute quadratic form f(x) = 1/2 x' P x
 * @param  P symmetric matrix (symmetry is TRIU or TRIL)
 * @param  x argument float vector
 * @return   quadratic form value
 */
c_float CscMatrix_quad_form(const CscMatrix  *P, const OSQPVectorf *x);
c_float CsrMatrix_quad_form(const CsrMatrix  *P, const OSQPVectorf *x);

# ifdef __cplusplus
}
# endif // ifdef __cplusplus

#endif // ifndef LIN_ALG_IMPL_H
