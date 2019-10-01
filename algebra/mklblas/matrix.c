#include "osqp.h"
#include "lin_alg.h"
#include "algebra_impl.h"
#include "csc_math.h"
#include "csc_utils.h"
//#include "PG_debug.h"
//#include "assert.h"

#define mkl
#ifdef mkl
  #ifdef DFLOAT
    #define cscmv mkl_scscmv
  #else
    #define cscmv mkl_dcscmv
  #endif //float or double

#endif // mkl def


/*  logical test functions ----------------------------------------------------*/

c_int OSQPMatrix_is_eq(OSQPMatrix *A, OSQPMatrix* B, c_float tol){
  return (A->symmetry == B->symmetry &&
          csc_is_eq(A->csc, B->csc, tol) );
}


/*  Non-embeddable functions (using malloc) ----------------------------------*/

#ifndef EMBEDDED

//Make a copy from a csc matrix.  Returns OSQP_NULL on failure

// TODO: add the struc matdescra variables
OSQPMatrix* OSQPMatrix_new_from_csc(const csc* A, c_int is_triu){

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

#endif //EMBEDDED

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
#ifdef mkl
void OSQPMatrix_mult_scalar(OSQPMatrix *A, c_float sc){
  csc_scale(A->csc,sc);
}

#else
void OSQPMatrix_mult_scalar(OSQPMatrix *A, c_float sc){
  csc_scale(A->csc,sc);
}

#endif // mult scalar

#ifdef mkl
void OSQPMatrix_lmult_diag(OSQPMatrix *A, const OSQPVectorf *L) {
  csc_lmult_diag(A->csc, OSQPVectorf_data(L));
}

#else
void OSQPMatrix_lmult_diag(OSQPMatrix *A, const OSQPVectorf *L) {
  csc_lmult_diag(A->csc, OSQPVectorf_data(L));
}

#endif // L*A mult

#ifdef mkl
void OSQPMatrix_rmult_diag(OSQPMatrix *A, const OSQPVectorf *R) {
  csc_rmult_diag(A->csc, OSQPVectorf_data(R));
}

#else
void OSQPMatrix_rmult_diag(OSQPMatrix *A, const OSQPVectorf *R) {
  csc_rmult_diag(A->csc, OSQPVectorf_data(R));
}

#endif // A*R mult

//y = alpha*A*x + beta*y
#ifdef mkl
void OSQPMatrix_Axpy(/*const*/ OSQPMatrix *A,
                     const OSQPVectorf *x,
                     OSQPVectorf *y,
                     c_float alpha,
                     c_float beta) {
  const c_float* xf = x->values;
  c_float* yf = y->values;
  const MKL_INT m = A->csc->m; // row
  const MKL_INT k = A->csc->n; // columns
  char transa = 'n';
  A->matdescra[0] = 'g';
  A->matdescra[1] = 'u';
  A->matdescra[2] = 'n';
  A->matdescra[3] = 'c';
  const c_float* val = A->csc->x; // numerical values
  const MKL_INT* indx = A->csc->i; // row indices
  const MKL_INT* pntrb = (A->csc->p); // column pointer starting with zero
  const MKL_INT* pntre = (A->csc->p + 1); // column pointer ending with 'k' (number of columns) 

  // printf("sizeof int = %d, sizeof MKLINT = %d\n", sizeof(c_int), sizeof(MKL_INT));
  //assert(x->length == k); 

  if (A->symmetry == NONE){
    // OSQPMatrix_print(A, "A1");
    // OSQPVectorf_print(x, "x");
    // OSQPVectorf_print(y, "y");
    // printf("%f %f are the const vals\n", alpha, beta);
    cscmv (&transa , &m , &k , &alpha , A->matdescra , val , indx , pntrb , pntre , xf , &beta , yf );
    // OSQPMatrix_print(A, "A2");
    // OSQPVectorf_print(x, "x");
    // OSQPVectorf_print(y, "y");
    // printf("%f %f are the const vals\n", alpha, beta);
  }
  else{ 
    A->matdescra[0] = 's';
    // OSQPMatrix_print(A, "A1_");
    // OSQPVectorf_print(x, "x_");
    // OSQPVectorf_print(y, "y_");
    // printf("%f %f are the const vals\n", alpha, beta);
    cscmv (&transa , &m , &k , &alpha , A->matdescra , val , indx , pntrb , pntre , xf , &beta , yf );
    // OSQPMatrix_print(A, "A2_");
    // OSQPVectorf_print(x, "x_");
    // OSQPVectorf_print(y, "y_");
    // printf("%f %f are the const vals\n", alpha, beta);
  } 
} 
 #else
 
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


#endif // Axpy matrix sparse blas implementation

#ifdef mkl// It has a bug Im not sure how to fix yet.

void OSQPMatrix_Atxpy(/*const*/ OSQPMatrix *A,
                      const OSQPVectorf *x,
                      OSQPVectorf *y,
                      c_float alpha,
                      c_float beta) {
    char transa = 't'; // Treating the transpose of the matrix
    const c_float* xf = x->values;
    c_float* yf = y->values;
    const MKL_INT m = A->csc->m; // row
    const MKL_INT k = A->csc->n; // columns
    A->matdescra[1] = 'u';
    A->matdescra[2] = 'n';
    A->matdescra[3] = 'c';
    const c_float* val = A->csc->x; // numerical values
    const MKL_INT* indx = A->csc->i; // row indices
    const MKL_INT* pntrb = (A->csc->p); // column pointer starting with zero
    const MKL_INT* pntre = (A->csc->p + 1); // column pointer ending with 'k' (number of columns) 

   if(A->symmetry == NONE){
    A->matdescra[0] = 'g';
    cscmv (&transa , &m , &k , &alpha , A->matdescra , val , indx , pntrb , pntre , xf , &beta , yf );
   }
   else{
    A->matdescra[0] = 's';
    cscmv (transa , &m , &k , &alpha , A->matdescra , val , indx , pntrb , pntre , xf , &beta , yf );
   }
}

#else

void OSQPMatrix_Atxpy(const OSQPMatrix *A,
                      const OSQPVectorf *x,
                      OSQPVectorf *y,
                      c_float alpha,
                      c_float beta) {

   if(A->symmetry == NONE) csc_Atxpy(A->csc, OSQPVectorf_data(x), OSQPVectorf_data(y), alpha, beta);
   else            csc_Axpy_sym_triu(A->csc, OSQPVectorf_data(x), OSQPVectorf_data(y), alpha, beta);
}


#endif // transposed AXPY 


#ifdef mkl 
c_float OSQPMatrix_quad_form(const OSQPMatrix* P, const OSQPVectorf* x) {
  if (P->symmetry == TRIU) {
    OSQPVectorf* y;
    y = OSQPVectorf_malloc(x->length);
    OSQPMatrix_Axpy(P, x, y, 1, 0); // Performing y=(P'*x)
    return 0.5*OSQPVectorf_dot_prod(y, x); // Performing y'*x = (P'*x)'*y = x'*P*x (quad form)
  }
  else {
#ifdef PRINTING
    c_eprint("quad_form matrix is not upper triangular");
#endif /* ifdef PRINTING */
    return -1.0;
  }
}

#else

c_float OSQPMatrix_quad_form(const OSQPMatrix* P, const OSQPVectorf* x) {
  if (P->symmetry == TRIU) return csc_quad_form(P->csc, OSQPVectorf_data(x));
  else {
#ifdef PRINTING
    c_eprint("quad_form matrix is not upper triangular");
#endif /* ifdef PRINTING */
    return -1.0;
  }
}
#endif // quad form

#if EMBEDDED != 1

void OSQPMatrix_col_norm_inf(const OSQPMatrix *M, OSQPVectorf *E) {
   csc_col_norm_inf(M->csc, OSQPVectorf_data(E));
}

void OSQPMatrix_row_norm_inf(const OSQPMatrix *M, OSQPVectorf *E) {
   if(M->symmetry == NONE) csc_row_norm_inf(M->csc, OSQPVectorf_data(E));
   else                    csc_row_norm_inf_sym_triu(M->csc, OSQPVectorf_data(E));
}

#endif // endef EMBEDDED

#ifndef EMBEDDED

void OSQPMatrix_free(OSQPMatrix *M){
  if (M) csc_spfree(M->csc);
  c_free(M);
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
