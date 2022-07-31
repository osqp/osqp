#include "osqp.h"
#include "lin_alg.h"
#include "algebra_impl.h"
#include "algebra_memory.h"
#include "csc_math.h"
#include "csc_utils.h"

#include <mkl.h>
#include <mkl_spblas.h>
#ifdef DFLOAT
  #define spblas_create_csc mkl_sparse_s_create_csc
  #define spblas_set_value mkl_sparse_s_set_value
  #define spblas_export_csc mkl_sparse_s_export_csc
  #define spblas_mv mkl_sparse_s_mv
#else
  #define spblas_create_csc mkl_sparse_d_create_csc
  #define spblas_set_value mkl_sparse_d_set_value
  #define spblas_export_csc mkl_sparse_d_export_csc
  #define spblas_mv mkl_sparse_d_mv
#endif //dfloat endif

/*  logical test functions ----------------------------------------------------*/

c_int OSQPMatrix_is_eq(const OSQPMatrix *A,
                       const OSQPMatrix *B,
                       c_float           tol){

  return (A->symmetry == B->symmetry &&
          csc_is_eq(A->csc, B->csc, tol) );
}

//Make a copy from a csc matrix.  Returns OSQP_NULL on failure
OSQPMatrix* OSQPMatrix_new_from_csc(const csc *A,
                                    c_int      is_triu){

  c_int i = 0;
  c_int n = A->n;   /* Number of columns */
  c_int m = A->m;   /* Number of rows */

  MKL_INT *p;
  MKL_INT retval = 0;
  MKL_INT nzmax = A->p[A->n];

  OSQPMatrix* out = c_malloc(sizeof(OSQPMatrix));

  if (!out)
   return OSQP_NULL;

  if (is_triu)
    out->symmetry = TRIU;
  else
    out->symmetry = NONE;

  out->csc = csc_copy(A);

  if(!out->csc){
    c_free(out);
    return OSQP_NULL;
  }

  out->shifted_p = blas_malloc(n+1);

  for (i = 0; i <= n; i++) {
    if(A->p[i] > 0)
      out->shifted_p[i] = A->p[i] - 1;
    else
      out->shifted_p[i] = A->p[i];
  }

  retval = spblas_create_csc(&out->mkl_mat,
                             SPARSE_INDEX_BASE_ZERO,
                             out->csc->m,      /* Number of rows */
                             out->csc->n,      /* Number of columns */
                             out->shifted_p,   /* Array of column start indices (this will only look at the first n entries in p, skipping the last one) */
                             out->shifted_p+1, /* Array of column end indices (this will skip the first entry to only look at the last n) */
                             out->csc->i,      /* Array of row indices */
                             out->csc->x);     /* The actual data */

  if (retval != SPARSE_STATUS_SUCCESS) {
    blas_free(p);
    OSQPMatrix_free(out);
    return OSQP_NULL;
  }

  return out;
}

/*  direct data access functions ---------------------------------------------*/

void OSQPMatrix_update_values(OSQPMatrix  *M,
                            const c_float *Mx_new,
                            const c_int   *Mx_new_idx,
                            c_int          M_new_n) {
  c_int i;

  /* Get a view into the data inside the current matrix */
  sparse_index_base_t idx_method = 0;
  MKL_INT numrows = 0;
  MKL_INT numcols = 0;

  MKL_INT *p_start;
  MKL_INT *p_end;
  MKL_INT *row_idx;
  c_float *vals;

  spblas_export_csc(M->mkl_mat, &idx_method, &numrows, &numcols, &p_start, &p_end, &row_idx, &vals);

  /* Update subset of elements */
  /* This operates on the assumption that we get a pointer to the actual data back from the above
     call to spblas_export_csc, which seems to be the case in all the testing done. */
  if (Mx_new_idx) {
    /* Change only Mx_new_idx */
    for (i = 0; i < M_new_n; i++) {
      vals[Mx_new_idx[i]] = Mx_new[i];
    }
  }
  else {
    /* Change whole M.  Assumes M_new_n == nnz(M) */
    for (i = 0; i < (p_end[numcols-1] + 1); i++) {
      vals[i] = Mx_new[i];
    }
  }
}

/* Matrix dimensions and data access */
c_int    OSQPMatrix_get_m(const OSQPMatrix *M){return M->csc->m;}
c_int    OSQPMatrix_get_n(const OSQPMatrix *M){return M->csc->n;}
c_float* OSQPMatrix_get_x(const OSQPMatrix *M){return M->csc->x;}
c_int*   OSQPMatrix_get_i(const OSQPMatrix *M){return M->csc->i;}
c_int*   OSQPMatrix_get_p(const OSQPMatrix *M){return M->csc->p;}
c_int    OSQPMatrix_get_nz(const OSQPMatrix *M){return M->csc->p[M->csc->n];}

csc*     OSQPMatrix_get_csc(const OSQPMatrix *M) {
  /* Values returned from the MKL object */
  sparse_index_base_t idx_method = 0;
  MKL_INT numrows = 0;
  MKL_INT numcols = 0;

  MKL_INT *p_start;
  MKL_INT *p_end;
  MKL_INT *row_idx;
  c_float *vals;

  /* Computed values */
  csc *B;
  c_int i = 0;
  c_int nnz = 0;

  spblas_export_csc(M->mkl_mat, &idx_method, &numrows, &numcols, &p_start, &p_end, &row_idx, &vals);

  /* Create the CSC using the returned data */
  nnz = p_end[numcols-1]+1;
  B = csc_spalloc(numcols, numrows, nnz, 1, 0);

  /* MKL doesn't give back the actual p we need, we need to take the last value from p_end and concatenate
     it onto the array returned in p_start */
  for (i = 0; i < numcols; i++) {
    if (i == 0)
      B->p[i] = p_start[i];
    else
      B->p[i] = p_start[i] + 1;
  }
  B->p[numcols] = p_end[numcols-1] + 1;

  for (i=0; i < nnz; i++) {
    B->i[i] = row_idx[i];
    B->x[i] = vals[i];
  }

  return B;
}


/* math functions ----------------------------------------------------------*/

//A = sc*A
void OSQPMatrix_mult_scalar(OSQPMatrix *A,
                            c_float     sc){
  /* This operates on the assumption that the stored shadow csc matrix is the backing memory for
     the actual MKL matrix handle, which seems to be the case in all the testing done. */
  csc_scale(A->csc, sc);
}

void OSQPMatrix_lmult_diag(OSQPMatrix        *A,
                           const OSQPVectorf *L) {
  /* This operates on the assumption that the stored shadow csc matrix is the backing memory for
     the actual MKL matrix handle, which seems to be the case in all the testing done. */
  csc_lmult_diag(A->csc, OSQPVectorf_data(L));
}

void OSQPMatrix_rmult_diag(OSQPMatrix        *A,
                           const OSQPVectorf *R) {
  /* This operates on the assumption that the stored shadow csc matrix is the backing memory for
     the actual MKL matrix handle, which seems to be the case in all the testing done. */
  csc_rmult_diag(A->csc, OSQPVectorf_data(R));
}

//y = alpha*A*x + beta*y
void OSQPMatrix_Axpy(const OSQPMatrix  *A,
                     const OSQPVectorf *x,
                     OSQPVectorf       *y,
                     c_float            alpha,
                     c_float            beta) {

  struct matrix_descr descr;

  if(A->symmetry == NONE){
    descr.type = SPARSE_MATRIX_TYPE_GENERAL;

    /* These don't actually matter for this mode, but put them to a known value */
    descr.mode = SPARSE_FILL_MODE_UPPER;
    descr.diag = SPARSE_DIAG_NON_UNIT;
  }
  else{
    /* Assumed to be TRIU if not NONE */
    descr.type = SPARSE_MATRIX_TYPE_SYMMETRIC;
    descr.mode = SPARSE_FILL_MODE_UPPER;
    descr.diag = SPARSE_DIAG_NON_UNIT;
  }

  spblas_mv(SPARSE_OPERATION_NON_TRANSPOSE, alpha, A->mkl_mat, descr, x->values, beta, y->values);
}

void OSQPMatrix_Atxpy(const OSQPMatrix  *A,
                      const OSQPVectorf *x,
                      OSQPVectorf       *y,
                      c_float            alpha,
                      c_float            beta) {
  struct matrix_descr descr;

  if(A->symmetry == NONE){
    descr.type = SPARSE_MATRIX_TYPE_GENERAL;

    /* These don't actually matter for this mode, but put them to a known value */
    descr.mode = SPARSE_FILL_MODE_UPPER;
    descr.diag = SPARSE_DIAG_NON_UNIT;
  }
  else{
    /* Assumed to be TRIU if not NONE */
    descr.type = SPARSE_MATRIX_TYPE_SYMMETRIC;
    descr.mode = SPARSE_FILL_MODE_UPPER;
    descr.diag = SPARSE_DIAG_NON_UNIT;
  }

  spblas_mv(SPARSE_OPERATION_TRANSPOSE, alpha, A->mkl_mat, descr, x->values, beta, y->values);
}

void OSQPMatrix_col_norm_inf(const OSQPMatrix *M,
                             OSQPVectorf      *E) {
  /* This operates on the assumption that the stored shadow csc matrix is the backing memory for
     the actual MKL matrix handle, which seems to be the case in all the testing done. */
   csc_col_norm_inf(M->csc, OSQPVectorf_data(E));
}

void OSQPMatrix_row_norm_inf(const OSQPMatrix *M,
                             OSQPVectorf      *E) {
  /* This operates on the assumption that the stored shadow csc matrix is the backing memory for
     the actual MKL matrix handle, which seems to be the case in all the testing done. */
   if(M->symmetry == NONE) csc_row_norm_inf(M->csc, OSQPVectorf_data(E));
   else                    csc_row_norm_inf_sym_triu(M->csc, OSQPVectorf_data(E));
}

void OSQPMatrix_free(OSQPMatrix *M){
  if (M) {
    if(M->mkl_mat)
      mkl_sparse_destroy(M->mkl_mat);

    if(M->csc)
      csc_spfree(M->csc);

    if(M->shifted_p)
      blas_free(M->shifted_p);
  };

  c_free(M);
}

OSQPMatrix* OSQPMatrix_submatrix_byrows(const OSQPMatrix  *A,
                                        const OSQPVectori *rows){
  /* This operates on the assumption that the stored shadow csc matrix is the backing memory for
     the actual MKL matrix handle, which seems to be the case in all the testing done. */
  csc        *M;
  OSQPMatrix *out;

  if(A->symmetry == TRIU){
    c_eprint("row selection not implemented for partially filled matrices");
    return OSQP_NULL;
  }

  M = csc_submatrix_byrows(A->csc, rows->values);

  if(!M) return OSQP_NULL;

  out = c_malloc(sizeof(OSQPMatrix));

  if(!out){
    csc_spfree(M);
    return OSQP_NULL;
  }

  out->symmetry = NONE;
  out->csc      = M;

  return out;
}
