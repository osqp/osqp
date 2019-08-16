#include "constants.h"
#include "lin_alg.h"
#include "./algebra_bare.h"

/* MATRIX FUNCTIONS ----------------------------------------------------------*/


void OSQPMatrix_lmult_diag(OSQPMatrix *A, const OSQPVectorf *L) {
  CscMatrix_lmult_diag(A->csc,L);
}

void OSQPMatrix_rmult_diag(OSQPMatrix *A, const OSQPVectorf *R) {
  CscMatrix_rmult_diag(A->csc,R);
}

//y = A*x + beta*y
void OSQPMatrix_Axpy(const OSQPMatrix *A,
                      const OSQPVectorf *x,
                      OSQPVectorf *y,
                      c_float alpha,
                      c_float beta) {

  if(A->symmetry == NONE){
    //full matrix
    CscMatrix_Axpy(A->csc, x, y, alpha, beta);
  }
  else{
    //should be TRIU here, but not directly checked
    CscMatrix_Axpy(A->csc, x, y, alpha, beta);    // y = Ax + by
    CscMatrix_Atxpy(A->csc, x, y, alpha, 1.0, 1); // y = (A^T - D)x + y
  }
}

void OSQPMatrix_Atxpy(const OSQPMatrix *A,
                      const OSQPVectorf *x,
                      OSQPVectorf *y,
                      c_float alpha,
                      c_float beta) {

   if(A->symmetry == NONE) CscMatrix_Atxpy(A->csc, x, y, alpha, beta, 0);
   else                    OSQPMatrix_Axpy(A,x,y,alpha,beta);
}


c_float OSQPMatrix_quad_form(const OSQPMatrix *P, const OSQPVectorf *x) {
   if(P->symmetry == TRIU) return CscMatrix_quad_form(P->csc, x);
   else {
#ifdef PRINTING
     c_eprint("quad_form matrix is not upper triangular");
#endif /* ifdef PRINTING */
     return -1.0;
   }
}

#if EMBEDDED != 1

void OSQPMatrix_col_norm_inf(const OSQPMatrix *M, OSQPVectorf *E) {
   CscMatrix_col_norm_inf(M->csc, E);
}

void OSQPMatrix_row_norm_inf(const OSQPMatrix *M, OSQPVectorf *E) {
   if(M->symmetry == NONE) CscMatrix_row_norm_inf(M->csc, E);
   else                    CscMatrix_row_norm_inf_sym_triu(M->csc, E);
}

#endif /* if EMBEDDED != 1 */


#ifndef EMBEDDED

OSQPMatrix* OSQPMatrix_symperm(const OSQPMatrix *A, const OSQPVectori *pinv, OSQPVectori *AtoC, c_int values){

  OSQPMatrix* out = c_malloc(sizeof(OSQPMatrix));
  if (!out) return OSQP_NULL;

  out->csc = CscMatrix_symperm(A->csc,pinv->values,AtoC->values,values);

  if(!out->csc){
    c_free(out); return OSQP_NULL;
  }
  out->symmetry = TRIU;
  return out;
}

#endif // ndef EMBEDDED
