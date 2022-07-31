#include "osqp.h"
#include "lin_alg.h"
#include "algebra_impl.h"
#include "csc_math.h"


#ifndef EMBEDDED

#include "csc_utils.h"

/*  logical test functions ----------------------------------------------------*/

c_int OSQPMatrix_is_eq(const OSQPMatrix *A,
                       const OSQPMatrix *B,
                       c_float           tol){

  return (A->symmetry == B->symmetry &&
          csc_is_eq(A->csc, B->csc, tol) );
}


/*  Non-embeddable functions (using malloc) ----------------------------------*/

//Make a copy from a csc matrix.  Returns OSQP_NULL on failure
OSQPMatrix* OSQPMatrix_new_from_csc(const csc *A,
                                    c_int      is_triu){

  OSQPMatrix* out = c_malloc(sizeof(OSQPMatrix));
  if(!out) return OSQP_NULL;

  if(is_triu) out->symmetry = TRIU;
  else        out->symmetry = NONE;

  out->csc = csc_copy(A);

  if(!out->csc){
    c_free(out);
    return OSQP_NULL;
  }
  else{
    return out;
  }
}

csc*     OSQPMatrix_get_csc(const OSQPMatrix *M) {return csc_copy(M->csc);}

#endif //EMBEDDED

/*  direct data access functions ---------------------------------------------*/

void OSQPMatrix_update_values(OSQPMatrix    *M,
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
void OSQPMatrix_mult_scalar(OSQPMatrix *A,
                            c_float     sc){
  csc_scale(A->csc,sc);
}

void OSQPMatrix_lmult_diag(OSQPMatrix        *A,
                           const OSQPVectorf *L) {
  csc_lmult_diag(A->csc, OSQPVectorf_data(L));
}

void OSQPMatrix_rmult_diag(OSQPMatrix *A, const OSQPVectorf *R) {
  csc_rmult_diag(A->csc, R->values);
}

//y = alpha*A*x + beta*y
void OSQPMatrix_Axpy(const OSQPMatrix  *A,
                     const OSQPVectorf *x,
                     OSQPVectorf       *y,
                     c_float            alpha,
                     c_float            beta) {

  if(A->symmetry == NONE){
    //full matrix
    csc_Axpy(A->csc, x->values, y->values, alpha, beta);
  }
  else{
    //should be TRIU here, but not directly checked
    csc_Axpy_sym_triu(A->csc, x->values, y->values, alpha, beta);
  }
}

void OSQPMatrix_Atxpy(const OSQPMatrix  *A,
                      const OSQPVectorf *x,
                      OSQPVectorf       *y,
                      c_float            alpha,
                      c_float            beta) {

   if(A->symmetry == NONE) csc_Atxpy(A->csc, x->values, y->values, alpha, beta);
   else            csc_Axpy_sym_triu(A->csc, x->values, y->values, alpha, beta);
}

// c_float OSQPMatrix_quad_form(const OSQPMatrix  *P,
//                              const OSQPVectorf *x) {

//    if (P->symmetry == TRIU) return csc_quad_form(P->csc, OSQPVectorf_data(x));
//    else {
//      c_eprint("quad_form matrix is not upper triangular");
//      return -1.0;
//    }
// }

#if EMBEDDED != 1

void OSQPMatrix_col_norm_inf(const OSQPMatrix *M,
                             OSQPVectorf      *E) {
   csc_col_norm_inf(M->csc, OSQPVectorf_data(E));
}

void OSQPMatrix_row_norm_inf(const OSQPMatrix *M,
                             OSQPVectorf      *E) {
   if(M->symmetry == NONE) csc_row_norm_inf(M->csc, OSQPVectorf_data(E));
   else                    csc_row_norm_inf_sym_triu(M->csc, OSQPVectorf_data(E));
}

#endif // endef EMBEDDED

#ifndef EMBEDDED

void OSQPMatrix_free(OSQPMatrix *M){
  if (M) csc_spfree(M->csc);
  c_free(M);
}

OSQPMatrix* OSQPMatrix_submatrix_byrows(const OSQPMatrix  *A,
                                        const OSQPVectori *rows){

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

#endif /* if EMBEDDED != 1 */
