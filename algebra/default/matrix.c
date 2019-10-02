#include "osqp_api_types.h"
#include "algebra_types.h"
#include "lin_alg.h"
#include "csc_math.h"
#include "csc_utils.h"


/*******************************************************************************
 *                       External CUDA Functions                               *
 *******************************************************************************/

/* cuda_malloc.h */
extern void cuda_free(void** devPtr);

/* cuda_csr.h */
extern void cuda_mat_init_P(const csc *mat, csr **P, c_float **d_P_triu_val, c_int **d_P_triu_to_full_ind, c_int **d_P_diag_ind);
extern void cuda_mat_init_A(const csc *mat, csr **A, csr **At, c_int **d_A_to_At_ind);
extern void cuda_mat_update_P(const c_float *Px, const c_int *Px_idx, c_int Px_n, csr **P, c_float *d_P_triu_val, c_int *d_P_triu_to_full_ind, c_int *d_P_diag_ind, c_int P_triu_nnz);
extern void cuda_mat_update_A(const c_float *Ax, const c_int *Ax_idx, c_int Ax_n, csr **A, csr **At, c_int *d_A_to_At_ind);
extern void cuda_mat_free(csr *mat);
extern void cuda_submat_byrows(const csr *A, const c_int *d_rows, csr **Ared, csr **Aredt);
extern void cuda_mat_get_m(const csr *mat, c_int *m);
extern void cuda_mat_get_n(const csr *mat, c_int *n);
extern void cuda_mat_get_nnz(const csr *mat, c_int *nnz);

/* cuda_lin_alg.h */
extern void cuda_mat_mult_sc(csr *S, csr *At, c_int symmetric, c_float sc);
extern void cuda_mat_lmult_diag(csr *S, csr *At, c_int symmetric, const c_float *d_diag);
extern void cuda_mat_rmult_diag(csr *S, csr *At, c_int symmetric, const c_float *d_diag);
extern void cuda_mat_Axpy(const csr *A, const c_float *d_x, c_float *d_y, c_float alpha, c_float beta);
extern void cuda_mat_quad_form(const csr *P, const c_float *d_x, c_float *h_res);
extern void cuda_mat_row_norm_inf(const csr *S, c_float *d_res);


/*  logical test functions ----------------------------------------------------*/

c_int OSQPMatrix_is_eq(OSQPMatrix *A, OSQPMatrix* B, c_float tol){
  return (A->symmetry == B->symmetry &&
          csc_is_eq(A->csc, B->csc, tol) );
}


/*******************************************************************************
 *                           API Functions                                     *
 *******************************************************************************/

OSQPMatrix* OSQPMatrix_new_from_csc(const csc *M,
                                    c_int      is_triu) {

  OSQPMatrix* out = c_calloc(1, sizeof(OSQPMatrix));
  if (!out) return OSQP_NULL;

  if (is_triu) {
    /* Initialize P */
    out->symmetric = 1;
    out->P_triu_nnz = M->p[M->n];
    cuda_mat_init_P(M, &out->S, &out->d_P_triu_val, &out->d_P_triu_to_full_ind, &out->d_P_diag_ind);
  }
  else {
    /* Initialize A */
    out->symmetric = 0;
    cuda_mat_init_A(M, &out->S, &out->At, &out->d_A_to_At_ind);
  }

  if (is_triu) out->symmetry = TRIU;
  else         out->symmetry = NONE;

  out->csc = csc_copy(M);

  if(!out->csc){
    c_free(out);
    return OSQP_NULL;
  }
  else{
    return out;
  }
}

void OSQPMatrix_update_values(OSQPMatrix    *mat,
                              const c_float *Mx_new,
                              const c_int   *Mx_new_idx,
                              c_int          Mx_new_n) {

  csc_update_values(mat->csc, Mx_new, Mx_new_idx, Mx_new_n);

  if (mat->symmetric) {
    cuda_mat_update_P(Mx_new, Mx_new_idx, Mx_new_n, &mat->S, mat->d_P_triu_val,
                      mat->d_P_triu_to_full_ind, mat->d_P_diag_ind, mat->P_triu_nnz);
  }
  else {
    cuda_mat_update_A(Mx_new, Mx_new_idx, Mx_new_n, &mat->S, &mat->At, mat->d_A_to_At_ind);
  }
}

c_int OSQPMatrix_get_m( const OSQPMatrix *mat) {
  c_int m;

  cuda_mat_get_nnz(mat->S, &m);

  return mat->csc->m;
}

c_int OSQPMatrix_get_n( const OSQPMatrix *mat) {
  c_int n;

  cuda_mat_get_nnz(mat->S, &n);

  return mat->csc->n;
}

c_int OSQPMatrix_get_nz(const OSQPMatrix *mat) {

  c_int nnz;


  if (mat->symmetric) nnz = mat->P_triu_nnz;
  else                cuda_mat_get_nnz(mat->S, &nnz);

  return mat->csc->p[mat->csc->n];
}

// GB: These are only used by direct solver interfaces. We will not need them in the PCG solver.
//     Anyway, they should be implemented by allocating c_float array and copying values there.
c_float* OSQPMatrix_get_x( const OSQPMatrix *mat){return mat->csc->x;}
c_int*   OSQPMatrix_get_i( const OSQPMatrix *mat){return mat->csc->i;}
c_int*   OSQPMatrix_get_p( const OSQPMatrix *mat){return mat->csc->p;}


void OSQPMatrix_mult_scalar(OSQPMatrix *mat,
                            c_float     sc) {

  csc_scale(mat->csc, sc);

  cuda_mat_mult_sc(mat->S, mat->At, mat->symmetric, sc);
}

void OSQPMatrix_lmult_diag(OSQPMatrix        *mat,
                           const OSQPVectorf *D) {

  csc_lmult_diag(mat->csc, OSQPVectorf_data(D));

  cuda_mat_lmult_diag(mat->S, mat->At, mat->symmetric, D->d_val);
}

void OSQPMatrix_rmult_diag(OSQPMatrix        *mat,
                           const OSQPVectorf *D) {

  csc_rmult_diag(mat->csc, OSQPVectorf_data(D));

  cuda_mat_rmult_diag(mat->S, mat->At, mat->symmetric, D->d_val);
}

// y = alpha*A*x + beta*y
void OSQPMatrix_Axpy(const OSQPMatrix  *mat,
                     const OSQPVectorf *x,
                     OSQPVectorf       *y,
                     c_float            alpha,
                     c_float            beta) {

  c_float* xf = OSQPVectorf_data(x);
  c_float* yf = OSQPVectorf_data(y);

  if(mat->symmetry == NONE){
    // full matrix
    csc_Axpy(mat->csc, xf, yf, alpha, beta);
  }
  else{
    // should be TRIU here, but not directly checked
    csc_Axpy_sym_triu(mat->csc, xf, yf, alpha, beta);
  }

  cuda_mat_Axpy(mat->S, x->d_val, y->d_val, alpha, beta);
}

void OSQPMatrix_Atxpy(const OSQPMatrix  *mat,
                      const OSQPVectorf *x,
                      OSQPVectorf       *y,
                      c_float            alpha,
                      c_float            beta) {

  if (mat->symmetry == NONE){
    csc_Atxpy(mat->csc, OSQPVectorf_data(x), OSQPVectorf_data(y), alpha, beta);
  }
  else{
    csc_Axpy_sym_triu(mat->csc, OSQPVectorf_data(x), OSQPVectorf_data(y), alpha, beta);
  }

  cuda_mat_Axpy(mat->At, x->d_val, y->d_val, alpha, beta);
}

c_float OSQPMatrix_quad_form(const OSQPMatrix  *mat,
                             const OSQPVectorf *x) {

  c_float res;

  if (mat->symmetric) {
    cuda_mat_quad_form(mat->S, x->d_val, &res);
    return csc_quad_form(mat->csc, OSQPVectorf_data(x));
  }
  else {
#ifdef PRINTING
    c_eprint("quad_form matrix is not upper triangular");
#endif /* ifdef PRINTING */
    return -1.0;
  }
}


void OSQPMatrix_col_norm_inf(const OSQPMatrix *mat,
                             OSQPVectorf      *res) {

  if (mat->symmetry == NONE) csc_col_norm_inf(mat->csc, OSQPVectorf_data(res));
  else                       csc_row_norm_inf_sym_triu(mat->csc, OSQPVectorf_data(res));

  if (mat->symmetric) cuda_mat_row_norm_inf(mat->S,  res->d_val);
  else                cuda_mat_row_norm_inf(mat->At, res->d_val);
}

void OSQPMatrix_row_norm_inf(const OSQPMatrix *mat,
                             OSQPVectorf      *res) {

  if (mat->symmetry == NONE) csc_row_norm_inf(mat->csc, OSQPVectorf_data(res));
  else                       csc_row_norm_inf_sym_triu(mat->csc, OSQPVectorf_data(res));

  cuda_mat_row_norm_inf(mat->S, res->d_val);
}



void OSQPMatrix_free(OSQPMatrix *mat){
  if (mat) {
    csc_spfree(mat->csc);
    cuda_mat_free(mat->S);
    cuda_mat_free(mat->At);
    cuda_free((void **) &mat->d_A_to_At_ind);
    cuda_free((void **) &mat->d_P_triu_to_full_ind);
    cuda_free((void **) &mat->d_P_diag_ind);
    cuda_free((void **) &mat->d_P_triu_val);
    c_free(mat);
  }
}

OSQPMatrix* OSQPMatrix_submatrix_byrows(const OSQPMatrix  *mat,
                                        const OSQPVectori *rows) {

  OSQPMatrix *out;

  if (mat->symmetry == TRIU) {
#ifdef PRINTING
    c_eprint("row selection not implemented for partially filled matrices");
#endif
    return OSQP_NULL;
  }

  out = c_calloc(1, sizeof(OSQPMatrix));

  if (!out) return OSQP_NULL;

  out->symmetry = NONE;
  out->csc = csc_submatrix_byrows(mat->csc, rows->values);

  out->symmetric = 0;
  cuda_submat_byrows(mat->S, rows->d_val, &out->S, &out->At);

  return out;
}

