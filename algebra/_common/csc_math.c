
#include "glob_opts.h"
#include "osqp.h"

/* internal utilities for zero-ing, setting and scaling without libraries */

void vec_set_scalar(c_float* v, c_float val, c_int n){
  c_int i;
  for(i = 0; i< n; i++) v[i] = val;
}

void vec_mult_scalar(c_float* v, c_float val, c_int n){
  c_int i;
  for(i = 0; i< n; i++) v[i] *= val;
}

void vec_negate(c_float* v, c_int n){
  c_int i;
  for(i = 0; i< n; i++) v[i] = -v[i];
}


/* CSC matrix operations implementation ------*/

/* update some or all matrix values */

void csc_update_values(csc           *M,
                       const c_float *Mx_new,
                       const c_int   *Mx_new_idx,
                       c_int          M_new_n){

  c_int i;

  // Update subset of elements
  if (Mx_new_idx) { // Change only Mx_new_idx
    for (i = 0; i < M_new_n; i++) {
      M->x[Mx_new_idx[i]] = Mx_new[i];
    }
  }
  else{ // Change whole M.  Assumes M_new_n == nnz(M)
    for (i = 0; i < M_new_n; i++) {
      M->x[i] = Mx_new[i];
    }
  }
}


/* matrix times scalar */

void csc_scale(csc* A, c_float sc){
  c_int i, nnzA;
  nnzA = A->p[A->n];
  for (i = 0; i < nnzA; i++) {
    A->x[i] *= sc;
  }
}

/* A = L*A */

void csc_lmult_diag(csc* A, const c_float *d){

  c_int j, i;
  c_int*   Ap = A->p;
  c_int*   Ai = A->i;
  c_float* Ax = A->x;
  c_int     n = A->n;

  for (j = 0; j < n; j++) {               // Cycle over columns
    for (i = Ap[j]; i < Ap[j + 1]; i++) { // Cycle every row in the column
      Ax[i] *= d[Ai[i]];                  // Scale by corresponding element
                                          // of d for row i
    }
  }
}

/* A = A*R */

void csc_rmult_diag(csc* A, const c_float* d){

  c_int j, i;
  c_int*   Ap = A->p;
  c_float* Ax = A->x;
  c_int     n = A->n;

  for (j = 0; j < n; j++) {                // Cycle over columns j
    for (i = Ap[j]; i < Ap[j + 1]; i++) {  // Cycle every row i in column j
      Ax[i] *= d[j];                       // Scale by corresponding element
                                           // of d for column j
    }
  }
}

//y = alpha*A*x + beta*y, where A is symmetric and only triu is stored
void csc_Axpy_sym_triu(const csc   *A,
              const c_float *x,
              c_float *y,
              c_float alpha,
              c_float beta) {

    c_int i, j;
    c_int*   Ap = A->p;
    c_int*   Ai = A->i;
    c_float* Ax = A->x;
    c_int    An = A->n;
    c_int    Am = A->m;

    // first do the b*y part
    if (beta == 0)        vec_set_scalar(y, 0.0, Am);
    else if (beta ==  1)  ; //do nothing
    else if (beta == -1)  vec_negate(y, Am);
    else vec_mult_scalar(y,beta, Am);


    // if A is empty or zero
    if (Ap[An] == 0 || alpha == 0.0) return;

    if (alpha == -1) {
        // y -= A*x
        for (j = 0; j < An; j++) {
            for (i = Ap[j]; i < Ap[j + 1]; i++) {
                y[Ai[i]] -= Ax[i] * x[j];
                if(Ai[i] != j){
                    y[j]     -= Ax[i] * x[Ai[i]];
                }
    }}}

    else if(alpha == 1){
        // y +=  A*x
        for (j = 0; j < An; j++) {
            for (i = Ap[j]; i < Ap[j + 1]; i++) {
                y[Ai[i]] += Ax[i] * x[j];
                if(Ai[i] != j){
                    y[j]     += Ax[i] * x[Ai[i]];
                }
    }}}

    else{
        // y +=  alpha*A*x
        for (j = 0; j < An; j++) {
            for (i = Ap[j]; i < Ap[j + 1]; i++) {
                y[Ai[i]] += alpha*Ax[i] * x[j];
                if(Ai[i] != j){
                    y[j]     += alpha*Ax[i] * x[Ai[i]];
                }
    }}}
}

//y = alpha*A*x + beta*y
void csc_Axpy(const csc   *A,
                    const c_float *x,
                    c_float *y,
                    c_float alpha,
                    c_float beta) {

  c_int i, j;
  c_int*   Ap = A->p;
  c_int*   Ai = A->i;
  c_float* Ax = A->x;
  c_int    An = A->n;
  c_int    Am = A->m;

  // first do the b*y part
  if (beta == 0)        vec_set_scalar(y, 0.0, Am);
  else if (beta ==  1)  ; //do nothing
  else if (beta == -1)  vec_negate(y, Am);
  else vec_mult_scalar(y,beta, Am);


  // if A is empty or zero
  if (Ap[An] == 0 || alpha == 0.0) return;

  if (alpha == -1) {
    // y -= A*x
    for (j = 0; j < An; j++) {
      for (i = Ap[j]; i < Ap[j + 1]; i++) {
        y[Ai[i]] -= Ax[i] * x[j];
    }}}

  else if(alpha == +1){
    // y +=  A*x
    for (j = 0; j < An; j++) {
      for (i = Ap[j]; i < Ap[j + 1]; i++) {
        y[Ai[i]] += Ax[i] * x[j];
    }}}

  else{
    // y +=  alpha*A*x
    for (j = 0; j < An; j++) {
      for (i = Ap[j]; i < Ap[j + 1]; i++) {
        y[Ai[i]] += alpha*Ax[i] * x[j];
    }}}
}

//y = alpha*A'*x + beta*y

void csc_Atxpy(const csc *A, const c_float *x, c_float *y,
                     c_float alpha, c_float beta) {
  c_int j, k;
  c_int*   Ap = A->p;
  c_int*   Ai = A->i;
  c_float* Ax = A->x;
  c_int    An = A->n;

  // first do the b*y part
  if (beta == 0)        vec_set_scalar(y, 0.0, An);
  else if (beta ==  1)  ; //do nothing
  else if (beta == -1)  vec_negate(y, An);
  else vec_mult_scalar(y,beta, An);

  // if A is empty or alpha = 0
  if (Ap[An] == 0 || alpha == 0.0) {
    return;
  }

    if(alpha == -1){
      for (j = 0; j < A->n; j++) {
        for (k = Ap[j]; k < Ap[j + 1]; k++) {
          y[j] -= Ax[k] * x[Ai[k]];
    }}}

    else if(alpha == +1){
      for (j = 0; j < A->n; j++) {
        for (k = Ap[j]; k < Ap[j + 1]; k++) {
          y[j] += Ax[k] * x[Ai[k]];
    }}}

    else{
      for (j = 0; j < A->n; j++) {
        for (k = Ap[j]; k < Ap[j + 1]; k++) {
          y[j] += alpha*Ax[k] * x[Ai[k]];
    }}}
}

// 1/2 x'*P*x

c_float csc_quad_form(const csc *P, const c_float *x) {

  //NB:implementation assumes upper triangular part only

  c_float quad_form = 0.;
  c_int   i, j, ptr;
  c_int*   Pp = P->p;
  c_int*   Pi = P->i;
  c_float* Px = P->x;
  c_int    Pn = P->n;


  for (j = 0; j < Pn; j++) {                    // Iterate over columns
    for (ptr = Pp[j]; ptr < Pp[j + 1]; ptr++) { // Iterate over rows
      i = Pi[ptr];                            // Row index

      if (i == j) {                                 // Diagonal element
        quad_form += (c_float).5 * Px[ptr] * x[i] * x[i];
      }
      else if (i < j) {                             // Off-diagonal element
        quad_form += Px[ptr] * x[i] * x[j];
      }
      else {                                        // Element in lower triangle
#ifdef PRINTING
        c_eprint("quad_form matrix is not upper triangular");
#endif /* ifdef PRINTING */
        return -1.;
      }
    }
  }
  return quad_form;
}

/* columnwise infinity norm */

void csc_col_norm_inf(const csc *M, c_float *E) {

  c_int j, ptr;
  c_int*   Mp = M->p;
  c_float* Mx = M->x;
  c_int    Mn = M->n;

  // Initialize zero max elements
  vec_set_scalar(E, 0.0, Mn);

  // Compute maximum across columns
  for (j = 0; j < Mn; j++) {
    for (ptr = Mp[j]; ptr < Mp[j + 1]; ptr++) {
      E[j] = c_max(c_absval(Mx[ptr]), E[j]);
    }
  }
}

/* rowwise infinity norm */

void csc_row_norm_inf(const csc *M, c_float *E) {

  c_int i, j, ptr;
  c_int*   Mp = M->p;
  c_int*   Mi = M->i;
  c_float* Mx = M->x;
  c_int    Mn = M->n;
  c_int    Mm = M->m;

  // Initialize zero max elements
  vec_set_scalar(E, 0.0, Mm);

  // Compute maximum across rows
  for (j = 0; j < Mn; j++) {
    for (ptr = Mp[j]; ptr < Mp[j + 1]; ptr++) {
      i    = Mi[ptr];
      E[i] = c_max(c_absval(Mx[ptr]), E[i]);
    }
  }
}

/* rowwise infinity norm, only upper triangle specified */

void csc_row_norm_inf_sym_triu(const csc *M, c_float *E) {

  c_int   i, j, ptr;
  c_int*   Mp = M->p;
  c_int*   Mi = M->i;
  c_float* Mx = M->x;
  c_int    Mn = M->n;
  c_int    Mm = M->m;
  c_float abs_x;

  // Initialize zero max elements
  vec_set_scalar(E, 0.0, Mm);

  // Compute maximum across columns
  // Note that element (i, j) contributes to
  // -> Column j (as expected in any matrices)
  // -> Column i (which is equal to row i for symmetric matrices)
  for (j = 0; j < Mn; j++) {
    for (ptr = Mp[j]; ptr < Mp[j + 1]; ptr++) {
      i     = Mi[ptr];
      abs_x = c_absval(Mx[ptr]);
      E[j]  = c_max(abs_x, E[j]);

      if (i != j) {
        E[i] = c_max(abs_x, E[i]);
      }
    }
  }
}
