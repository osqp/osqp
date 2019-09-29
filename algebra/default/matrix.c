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
extern void cuda_mat_free(csr *dev_mat);


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


/*  direct data access functions ---------------------------------------------*/

void OSQPMatrix_update_values(OSQPMatrix  *M,
                            const c_float *Mx_new,
                            const c_int   *Mx_new_idx,
                            c_int          M_new_n){
  csc_update_values(M->csc, Mx_new, Mx_new_idx, M_new_n);
}

/* Matrix dimensions and data access */
c_int    OSQPMatrix_get_m(const OSQPMatrix *M){return M->csc->m;}
c_int    OSQPMatrix_get_n(const OSQPMatrix *M){return M->csc->n;}
c_float* OSQPMatrix_get_x(const OSQPMatrix *M){return M->csc->x;}
c_int*   OSQPMatrix_get_i(const OSQPMatrix *M){return M->csc->i;}
c_int*   OSQPMatrix_get_p(const OSQPMatrix *M){return M->csc->p;}
c_int    OSQPMatrix_get_nz(const OSQPMatrix *M){return M->csc->p[M->csc->n];}


/* math functions ----------------------------------------------------------*/

//A = sc*A
void OSQPMatrix_mult_scalar(OSQPMatrix *A, c_float sc){
  csc_scale(A->csc,sc);
}

void OSQPMatrix_lmult_diag(OSQPMatrix *A, const OSQPVectorf *L) {
  csc_lmult_diag(A->csc, OSQPVectorf_data(L));
}

void OSQPMatrix_rmult_diag(OSQPMatrix *A, const OSQPVectorf *R) {
  csc_rmult_diag(A->csc, OSQPVectorf_data(R));
}

//y = alpha*A*x + beta*y
void OSQPMatrix_Axpy(const OSQPMatrix *A,
                     const OSQPVectorf *x,
                     OSQPVectorf *y,
                     c_float alpha,
                     c_float beta) {

  c_float* xf = OSQPVectorf_data(x);
  c_float* yf = OSQPVectorf_data(y);

  if(A->symmetry == NONE){
    //full matrix
    csc_Axpy(A->csc, xf, yf, alpha, beta);
  }
  else{
    //should be TRIU here, but not directly checked
    csc_Axpy_sym_triu(A->csc, xf, yf, alpha, beta);
  }
}

void OSQPMatrix_Atxpy(const OSQPMatrix *A,
                      const OSQPVectorf *x,
                      OSQPVectorf *y,
                      c_float alpha,
                      c_float beta) {

   if(A->symmetry == NONE) csc_Atxpy(A->csc, OSQPVectorf_data(x), OSQPVectorf_data(y), alpha, beta);
   else            csc_Axpy_sym_triu(A->csc, OSQPVectorf_data(x), OSQPVectorf_data(y), alpha, beta);
}


c_float OSQPMatrix_quad_form(const OSQPMatrix *P, const OSQPVectorf *x) {
   if(P->symmetry == TRIU) return csc_quad_form(P->csc, OSQPVectorf_data(x));
   else {
#ifdef PRINTING
     c_eprint("quad_form matrix is not upper triangular");
#endif /* ifdef PRINTING */
     return -1.0;
   }
}


void OSQPMatrix_col_norm_inf(const OSQPMatrix *M, OSQPVectorf *E) {
   csc_col_norm_inf(M->csc, OSQPVectorf_data(E));
}

void OSQPMatrix_row_norm_inf(const OSQPMatrix *M, OSQPVectorf *E) {
   if(M->symmetry == NONE) csc_row_norm_inf(M->csc, OSQPVectorf_data(E));
   else                    csc_row_norm_inf_sym_triu(M->csc, OSQPVectorf_data(E));
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

OSQPMatrix* OSQPMatrix_submatrix_byrows(const OSQPMatrix* A, const OSQPVectori* rows){

  csc        *M;
  OSQPMatrix *out;


  if(A->symmetry == TRIU){
#ifdef PRINTING
    c_eprint("row selection not implemented for partially filled matrices");
#endif
    return OSQP_NULL;
  }


  M = csc_submatrix_byrows(A->csc, OSQPVectori_data(rows));

  if(!M) return OSQP_NULL;

  out = c_calloc(1, sizeof(OSQPMatrix));

  if(!out){
    csc_spfree(M);
    return OSQP_NULL;
  }

  out->symmetry = NONE;
  out->csc      = M;

  return out;

}

