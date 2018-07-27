#include "lin_alg_impl.h"
#include <assert.h>

/* a macro for swapping variable pairs */
#define SWAP(x, y, T) do { T SWAP = x; *(T *)(&x) = y; *(T *)(&y) = SWAP; } while (0)

/* VECTOR FUNCTIONS ----------------------------------------------------------*/


#ifndef EMBEDDED

OSQPVectorf* OSQPVectorf_copy_new(OSQPVectorf *a){

  OSQPVectorf *b;
  b         = c_malloc(sizeof(OSQPVectorf));
  b->length = a->length;
  b->values = c_malloc(b->length * sizeof(c_float));

  OSQPVectorf_copy(a,b);
  return b;
}

OSQPVectori* OSQPVectori_copy_new(OSQPVectori *a){

  OSQPVectori *b;
  b         = c_malloc(sizeof(OSQPVectori));
  b->length = a->length;
  b->values = c_malloc(b->length * sizeof(c_int));

  OSQPVectori_copy(a,b);
  return b;
}

#endif // end EMBEDDED

c_int OSQPVectorf_length(OSQPVectorf *a){return a->length;}
c_int OSQPVectori_length(OSQPVectori *a){return a->length;}

/* Pointer to vector data (floats) */
c_float* OSQPVectorf_data(OSQPVectorf *a){return a->values;}
c_float* OSQPVectori_data(OSQPVectori *a){return a->values;}

void OSQPVectorf_copy(OSQPVectorf *a,OSQPVectorf *b){

  c_int i;

  assert(a->length == b->length);

  for (i = 0; i < b->length; i++) {
    b->values[i] = a->values[i];
  }
}

void OSQPVectori_copy(OSQPVectori *a,OSQPVectori *b){

  c_int i;

  assert(a->length == b->length);

  for (i = 0; i < b->length; i++) {
    b->values[i] = a->values[i];
  }
}

void OSQPVectorf_set_scalar(OSQPVectorf *a, c_float sc){
  c_int i;
  for (i = 0; i < (int)a->length; i++) {
    a->values[i] = sc;
  }
  return;
}

void OSQPVectori_set_scalar(OSQPVectori *a, c_int sc){
  c_int i;
  for (i = 0; i < a->length; i++) {
    a->values[i] = sc;
  }
}

void OSQPVectorf_add_scalar(OSQPVectorf *a, c_float sc){
  c_int i;
  for (i = 0; i < a->length; i++) {
    a->values[i] += sc;
  }
}

void OSQPVectori_add_scalar(OSQPVectori *a, c_int sc){
  c_int i;
  for (i = 0; i < a->length; i++) {
    a->values[i] += sc;
  }
}

void OSQPVectorf_mult_scalar(OSQPVectorf *a, c_float sc){
  c_int i;
  for (i = 0; i < a->length; i++) {
    a->values[i] *= sc;
  }
}

void OSQPVectorf_negate(OSQPVectorf *a){
  c_int i;
  for (i = 0; i < a->length; i++) {
    a->values[i] = -a->values[i];
  }
}

void OSQPVectorf_add_scaled(OSQPVectorf       *c,
                             const OSQPVectorf *a,
                             const OSQPVectorf *b,
                             c_float           sc){
  c_int i;

  assert(a->length == b->length);
  assert(a->length == c->length);

  for (i = 0; i < a->length; i++) {
    c->values[i] =  a->values[i] + sc * b->values[i];
  }
}

c_float OSQPVectorf_norm_inf(const OSQPVectorf *v){

  c_int   i;
  c_float normval = 0.0;
  c_float absval;

  for (i = 0; i < v->length; i++) {
    absval = c_absval(v->values[i]);
    if (absval > normval) normval = absval;
  }
  return normval;
}

c_float OSQPVectorf_norm_1(const OSQPVectorf *v){

  c_int   i;
  c_float normval = 0.0;

  for (i = 0; i < v->length; i++) {
    normval += c_absval(v->values[i]);
  }
  return normval;
}

c_float OSQPVectorf_scaled_norm_inf(const OSQPVectorf *S, const OSQPVectorf *v){

  c_int   i;
  c_float absval;
  c_float normval = 0.0;

  assert(S->length == v->length);

  for (i = 0; i < v->length; i++) {
    absval = c_absval(S->values[i] * v->values[i]);
    if (absval > normval) normval = absval;
  }
  return normval;
}

c_float OSQPVectorf_scaled_norm_1(const OSQPVectorf *S, const OSQPVectorf *v){

  c_int   i;
  c_float normval = 0.0;

  assert(S->length == v->length);

  for (i = 0; i < v->length; i++) {
    normval += c_absval(S->values[i] * v->values[i]);
  }
  return normval;
}

c_float OSQPVectorf_norm_inf_diff(const OSQPVectorf *a,
                                  const OSQPVectorf *b){
  c_int   i;
  c_float absval;
  c_float normDiff = 0.0;

  assert(a->length == b->length);

  for (i = 0; i < a->length; i++) {
    absval = c_absval(a->values[i] - b->values[i]);
    if (absval > normDiff) normDiff = absval;
  }
  return normDiff;
}

c_float OSQPVectorf_norm_1_diff(const OSQPVectorf *a,
                                const OSQPVectorf *b){

  c_int   i;
  c_float normDiff = 0.0;

  assert(a->length == b->length);

  for (i = 0; i < a->length; i++) {
    normDiff += c_absval(a->values[i] - b->values[i]);
  }
  return normDiff;
}

c_float OSQPVectorf_sum(const OSQPVectorf *a){

  c_int   i;
  c_float val = 0.0;

  for (i = 0; i < a->length; i++) {
    val += a->values[i];
  }

  return val;
}

c_float OSQPVectorf_dot_prod(const OSQPVectorf *a, const OSQPVectorf *b){

  c_int   i; // Index
  c_float dotprod = 0.0;

  assert(a->length == b->length);

  for (i = 0; i < a->length; i++) {
    dotprod += a->values[i] * b->values[i];
  }

  return dotprod;
}

void OSQPVectorf_ew_prod(const OSQPVectorf *a,
                         const OSQPVectorf *b,
                               OSQPVectorf *c){

    c_int i;

    assert(a->length == b->length);
    assert(a->length == c->length);

    for (i = 0; i < a->length; i++) {
      c->values[i] = a->values[i] * b->values[i];
    }
}

#if EMBEDDED != 1

c_float OSQPVectorf_mean(const OSQPVectorf *a){

  if(a->length){
    return OSQPVectorf_sum(a)/(a->length);
  }
  else{
    return 0;
  }
}

void OSQPVectorf_ew_reciprocal(const OSQPVectorf *a, OSQPVectorf *b){

  c_int i;

  assert(a->length == b->length);

  for (i = 0; i < a->length; i++) {
    b->values[i] = (c_float)1.0 / a->values[i];
  }
}

void OSQPVectorf_ew_sqrt(OSQPVectorf *a){

  c_int i;

  for (i = 0; i < a->length; i++) {
    a->values[i] = c_sqrt(a->values[i]);
  }
}

void OSQPVectorf_ew_max(OSQPVectorf *a, c_float max_val){

  c_int i;

  for (i = 0; i < a->length; i++) {
    a->values[i] = c_max(a->values[i], max_val);
  }
}

void OSQPVectorf_ew_min(OSQPVectorf *a, c_float min_val){

  c_int i;

  for (i = 0; i < a->length; i++) {
    a->values[i] = c_min(a->values[i], min_val);
  }
}

void OSQPVectorf_ew_max_vec(const OSQPVectorf *a,
                            const OSQPVectorf *b,
                            OSQPVectorf       *c){
  c_int i;

  assert(a->length == b->length);
  assert(a->length == c->length);

  for (i = 0; i < a->length; i++) {
    c->values[i] = c_max(a->values[i], b->values[i]);
  }
}

void OSQPVectorf_ew_min_vec(const OSQPVectorf *a,
                            const OSQPVectorf *b,
                            OSQPVectorf       *c){
  c_int i;

  assert(a->length == b->length);
  assert(a->length == c->length);

  for (i = 0; i < a->length; i++) {
    c->values[i] = c_min(a->values[i], b->values[i]);
  }
}

#endif // EMBEDDED != 1


/* MATRIX FUNCTIONS ----------------------------------------------------------*/

/* Scalar multiplication, with file scope CSR/CSC implementations */

void OSQPMatrix_mult_scalar(OSQPMatrix *A, c_float sc){
  if(A->csc != OSQP_NULL) CscMatrix_mult_scalar(A->csc,sc);
  if(A->csr != OSQP_NULL) CsrMatrix_mult_scalar(A->csr,sc);
}

void CscMatrix_mult_scalar(CscMatrix *A, c_float sc) {
  c_int i, nnzA;
  nnzA = A->p[A->n];
  for (i = 0; i < nnzA; i++) {
    A->x[i] *= sc;
  }
}

void CsrMatrix_mult_scalar(CsrMatrix *A, c_float sc) {
  SWAP(A->m,A->n,c_int);
  CscMatrix_mult_scalar((CscMatrix *)A,sc);
  SWAP(A->m,A->n,c_int);
}

/* Left diagonal matrix multiplication, with file scope CSR/CSC implementations */

void OSQPMatrix_premult_diag(OSQPMatrix *A, const OSQPVectorf *d){
  if(A->csc != OSQP_NULL) CscMatrix_premult_diag(A->csc,d);
  if(A->csr != OSQP_NULL) CsrMatrix_premult_diag(A->csr,d);
}

void CscMatrix_premult_diag(CscMatrix *A, const OSQPVectorf *d){
  c_int j, i;
  assert(A->m == d->length);
  for (j = 0; j < A->n; j++) {                // Cycle over columns
    for (i = A->p[j]; i < A->p[j + 1]; i++) { // Cycle every row in the column
      A->x[i] *= d->values[A->i[i]];          // Scale by corresponding element
                                              // of d for row i
    }
  }
}

void CsrMatrix_premult_diag(CsrMatrix *A, const OSQPVectorf *d){
  /* postmult the transpose */
  SWAP(A->m,A->n,c_int);
  CscMatrix_postmult_diag((CscMatrix *)A,d);
  SWAP(A->m,A->n,c_int);
}

/* Right diagonal matrix multiplication, with file scope CSR/CSC implementations */

void OSQPMatrix_postmult_diag(OSQPMatrix *A, const OSQPVectorf *d){
  if(A->csc != OSQP_NULL) CscMatrix_postmult_diag(A->csc,d);
  if(A->csr != OSQP_NULL) CsrMatrix_postmult_diag(A->csr,d);
}

void CscMatrix_postmult_diag(CscMatrix *A, const OSQPVectorf* d){
  c_int j, i;
  assert(A->n == d->length);
  for (j = 0; j < A->n; j++) {                // Cycle over columns j
    for (i = A->p[j]; i < A->p[j + 1]; i++) { // Cycle every row i in column j
      A->x[i] *= d->values[j];                        // Scale by corresponding element
                                              // of d for column j
    }
  }
}

void CsrMatrix_postmult_diag(CsrMatrix *A, const OSQPVectorf *d){
  /* premult the transpose */
  SWAP(A->m,A->n,c_int);
  CscMatrix_premult_diag((CscMatrix *)A,d);
  SWAP(A->m,A->n,c_int);
}

/*
  Matrix vector multiplication A*x, with file scope CSR/CSC implementations
  For this operations, always prefer the CSR format if it is available
*/

void OSQPMatrix_Ax(const OSQPMatrix  *A,
                   const OSQPVectorf *x,
                   OSQPVectorf       *y,
                   c_int             sign){

  /* at least one format must exist */
  assert(A->csr != OSQP_NULL || A->csc != OSQP_NULL);

  if(A->symmetry == NONE){
    /*Assume that CSR is faster, so use that
     form if it exists.  Otherwise CSC */
    if(A->csr != OSQP_NULL){
      CsrMatrix_Ax(A->csr,x,y,sign,0);
    }
    else{
      CscMatrix_Ax(A->csc,x,y,sign,0);
    }
    return;
  /*
     if the matrix is symmetric in TRIU or TRIL
     format, then take some extra care.
     procedure will be (A+ (A'-D) )x, where D is the
     diagonal and is excluded on the second multiply
  */
  }
  else if(A->symmetry == TRIU || A->symmetry == TRIL){

    c_int sign2 = sign;
    if(sign2 == 0) sign2 = 1;  //add on the second step

   /*
    if both formats exist, then use both to
    maximize the row-wise operations.
   */
    if(A->csr != OSQP_NULL && A->csc != OSQP_NULL){
       CsrMatrix_Ax(A->csr,x,y,sign,0);
       CscMatrix_Atx(A->csc,x,y,sign2,1);
    }
    else if(A->csr != OSQP_NULL){
       CsrMatrix_Ax(A->csr,x,y,sign,0);
       CsrMatrix_Atx(A->csr,x,y,sign2,1);
    }
    else{
       CscMatrix_Ax(A->csc,x,y,sign,0);
       CscMatrix_Atx(A->csc,x,y,sign2,1);
    }
  }
  else assert(0);
}

void CscMatrix_Ax(const CscMatrix   *A,
                  const OSQPVectorf *x,
                  OSQPVectorf       *y,
                  c_int             sign,
                  c_int             skip_diag){

  c_int i, j, k;

  /* if sign == 0, y = 0 */
  if (sign == 0) OSQPVectorf_set_scalar(y, (c_float)0.0);

  // if A is empty, no output
  if (A->p[A->n] == 0) return;

  if(skip_diag){
    // y +=  A*x (includes sign = 0 case)
    if(sign >= 0){
    for (j = 0; j < A->n; j++) {
      for (i = A->p[j]; i < A->p[j + 1]; i++) {
        k = A->i[i];
        y->values[k] += (k == j) ? 0 : A->x[i] * x->values[j];
    }}}
    // y -= A*x
    else{
    for (j = 0; j < A->n; j++) {
      for (i = A->p[j]; i < A->p[j + 1]; i++) {
        k = A->i[i];
        y->values[k] -= (k == j) ? 0 :  A->x[i] * x->values[j];
    }}}
  }
  else{
    // y +=  A*x (includes sign = 0 case)
    if(sign >= 0){
    for (j = 0; j < A->n; j++) {
      for (i = A->p[j]; i < A->p[j + 1]; i++) {
        y->values[A->i[i]] += A->x[i] * x->values[j];
    }}}
    // y -= A*x
    else{
      for (j = 0; j < A->n; j++) {
        for (i = A->p[j]; i < A->p[j + 1]; i++) {
          y->values[A->i[i]] -= A->x[i] * x->values[j];
    }}}
  }
}

void CsrMatrix_Ax(const CsrMatrix   *A,
                  const OSQPVectorf *x,
                  OSQPVectorf       *y,
                  c_int             sign,
                  c_int             skip_diag){

    c_int i, j, k;

    /* if sign == 0, y = 0 */
    if (sign == 0) OSQPVectorf_set_scalar(y, (c_float)0.0);

    // if A is empty, no output
    if (A->p[A->m] == 0) return;

    if(skip_diag){
      // y +=  A*x (includes sign = 0 case)
      if(sign >= 0){
      for (j = 0; j < A->m; j++) {
        for (k = A->p[j]; k < A->p[j + 1]; k++) {
          i = A->i[k];
          y->values[j] += (i == j) ? 0 : A->x[k] * x->values[i];
        }}}
      else{
      for (j = 0; j < A->m; j++) {
        for (k = A->p[j]; k < A->p[j + 1]; k++) {
          i = A->i[k];
          y->values[j] -= (i == j) ? 0 : A->x[k] * x->values[i];
        }}}
    }
    else{
      // y +=  A*x (includes sign = 0 case)
      if(sign >= 0){
      for (j = 0; j < A->m; j++) {
        for (k = A->p[j]; k < A->p[j + 1]; k++) {
          y->values[j] += A->x[k] * x->values[A->i[k]];
        }}}
      else{
      for (j = 0; j < A->m; j++) {
        for (k = A->p[j]; k < A->p[j + 1]; k++) {
          y->values[j] -= A->x[k] * x->values[A->i[k]];
        }}}
    }
  }

/*
  Matrix vector A^T*x multiplication, with file scope CSR/CSC implementations
  For this operation, always prefer the CSR format if it is available
*/

void OSQPMatrix_Atx(const OSQPMatrix  *A,
                   const OSQPVectorf *x,
                   OSQPVectorf       *y,
                   c_int             sign){

  /* at least one format must exist */
  assert(A->csr != OSQP_NULL || A->csc != OSQP_NULL);

  if(A->symmetry == NONE){
    /*Assume that CSC is faster, so use that
     form if it exists.  Otherwise CSR */
    if(A->csc != OSQP_NULL){
      CscMatrix_Atx(A->csc,x,y,sign,0);
    }
    else{
      CsrMatrix_Atx(A->csr,x,y,sign,0);
    }
  }
  else{
    OSQPMatrix_Ax(A,x,y,sign);
  }
}

void CscMatrix_Atx(const CscMatrix  *A,
                   const OSQPVectorf *x,
                   OSQPVectorf       *y,
                   c_int             sign,
                   c_int             skip_diag){
  // Cast to CSR to multiply by the transpose
  SWAP(A->m,A->n,c_int);
  CsrMatrix_Ax((const CsrMatrix*)A,x,y,sign,skip_diag);
  SWAP(A->m,A->n,c_int);
}

void CsrMatrix_Atx(const CsrMatrix  *A,
                   const OSQPVectorf *x,
                   OSQPVectorf       *y,
                   c_int             sign,
                   c_int             skip_diag){
  // Cast to CSR to multiply by the transpose
  SWAP(A->m,A->n,c_int);
  CscMatrix_Ax((const CscMatrix*)A,x,y,sign,skip_diag);
  SWAP(A->m,A->n,c_int);
}


#if EMBEDDED != 1

/* Column-wise infinity norms */

void OSQPMatrix_inf_norm_cols(const OSQPMatrix *M,
                              OSQPVectorf      *E){

  /* at least one format must exist */
  assert(M->csr != OSQP_NULL || M->csc != OSQP_NULL);

  // Initialize zero max elements
  OSQPVectorf_set_scalar(E,0.0);

  if(M->symmetry == NONE){
    /*Assume that CSC is faster, so use that
    form if it exists.  Otherwise CSR */
    if(M->csc != OSQP_NULL){
      CscMatrix_inf_norm_cols(M->csc,E);
    }
    else{
      CsrMatrix_inf_norm_cols(M->csr,E);
    }
  }
  else if(M->symmetry == TRIL || M->symmetry == TRIU){
    /* if both formats exist, then use both to
       maximize the row-wise operations.
     */
   if(M->csr != OSQP_NULL && M->csc != OSQP_NULL){
      CscMatrix_inf_norm_cols(M->csc,E);
      CsrMatrix_inf_norm_rows(M->csr,E);
   }
   else if(M->csc != OSQP_NULL){
      CscMatrix_inf_norm_cols(M->csc,E);
      CscMatrix_inf_norm_rows(M->csc,E);
   }
   else{
      CsrMatrix_inf_norm_cols(M->csr,E);
      CsrMatrix_inf_norm_rows(M->csr,E);
   }
  }
  else assert(0);

}

void CscMatrix_inf_norm_cols(const CscMatrix *M,
                             OSQPVectorf     *E){

   // Warning! E is *not* set to zero to start, so that
   // we can use this function as part of a two-step procedure
   // to compute the norm of a TRIU or TRIL matrix from
   // OSQPMatrix_inf_norm_<dim>

  c_int j, ptr;

  assert(M->n == E->length);

  /* Compute maximum across columns */
  for (j = 0; j < M->n; j++) {
   for (ptr = M->p[j]; ptr < M->p[j + 1]; ptr++) {
     E->values[j] = c_max(c_absval(M->x[ptr]), E->values[j]);
   }
  }
}

void CsrMatrix_inf_norm_cols(const CsrMatrix *M,
                             OSQPVectorf     *E){
   SWAP(M->m,M->n,c_int);
   CscMatrix_inf_norm_rows((const CscMatrix*)M,E);
   SWAP(M->m,M->n,c_int);
}

/* row wise infinity norms */

void OSQPMatrix_inf_norm_rows(const OSQPMatrix *M,
                              OSQPVectorf      *E){

  /* at least one format must exist */
  assert(M->csr != OSQP_NULL || M->csc != OSQP_NULL);

  if(M->symmetry == NONE){

    /*
     Initialize zero max elements
     done here so that symmetric norm
     case doesn't doubly initialise E
    */
    OSQPVectorf_set_scalar(E,0.0);

    /*Assume that CSR is faster, so use that
    form if it exists.  Otherwise CSC */
    if(M->csr != OSQP_NULL){
      CsrMatrix_inf_norm_rows(M->csr,E);
    }
    else{
      CscMatrix_inf_norm_rows(M->csc,E);
    }
  }
  else if(M->symmetry == TRIL || M->symmetry == TRIU){
      /* just use col norms instead*/
      OSQPMatrix_inf_norm_cols(M,E);
  }
  else assert(0);
}

void CscMatrix_inf_norm_rows(const CscMatrix *M,
                             OSQPVectorf     *E){

  // Warning! E is *not* set to zero to start, so that
  // we can use this function as part of a two-step procedure
  // to compute the norm of a TRIU or TRIL matrix from
  // OSQPMatrix_inf_norm_<dim>

  c_int i, j, ptr;

  assert(M->m == E->length);

  // Compute maximum across rows
  for (j = 0; j < M->m; j++) {
    for (ptr = M->p[j]; ptr < M->p[j + 1]; ptr++) {
      i    = M->i[ptr];
      E->values[i] = c_max(c_absval(M->x[ptr]), E->values[i]);
    }
  }
}

void CsrMatrix_inf_norm_rows(const CsrMatrix *M,
                            OSQPVectorf      *E){
  SWAP(M->m,M->n,c_int);
  CscMatrix_inf_norm_cols((const CscMatrix*)M,E);
  SWAP(M->m,M->n,c_int);
}

#endif /* if EMBEDDED != 1 */

c_float OSQPMatrix_quad_form(const OSQPMatrix  *P,
                             const OSQPVectorf *x){

  c_float val;

  assert(P->symmetry == TRIU || P->symmetry == TRIL);
  assert(P->csc != OSQP_NULL || P-> csr != OSQP_NULL);

  /*
  It's not obviously better to use CSR or CSC here,
  so just use CSC by default if it exists
  */
  if(P->csc) val = CscMatrix_quad_form(P->csc,x);
  else       val = CscMatrix_quad_form(P->csr,x);

  return val;
}

c_float CscMatrix_quad_form(const CscMatrix  *P,
                            const OSQPVectorf *x){

  c_float quad_form = 0.;
  c_int   i, j, ptr;                                // Pointers to iterate over
                                                    // matrix: (i,j) a element
                                                    // pointer
  assert(P->m == P->n);
  assert(x->length == P->m);

  //NB : Assumes that the matrix is in upper or lower triangular form.

  for (j = 0; j < P->n; j++) {                      // Iterate over columns
    for (ptr = P->p[j]; ptr < P->p[j + 1]; ptr++) { // Iterate over rows
      i = P->i[ptr];                                // Row index

      if (i == j) {                                 // Diagonal element
        quad_form += (c_float).5 * P->x[ptr] * x->values[i] * x->values[i];
      }
      else{                                         // Off-diagonal element
        quad_form += P->x[ptr] * x->values[i] * x->values[j];
      }
    }
  }
  return quad_form;
}

c_float CsrMatrix_quad_form(const CsrMatrix  *P,
                            const OSQPVectorf *x){
  return CscMatrix_quad_form((const CscMatrix *)P,x);
}
