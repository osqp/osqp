
#include "glob_opts.h"
#include "osqp.h"

/* internal utilities for zero-ing, setting and scaling without libraries */

void vec_set_scalar(OSQPFloat* v, OSQPFloat val, OSQPInt n){
  OSQPInt i;
  for(i = 0; i< n; i++) v[i] = val;
}

void vec_mult_scalar(OSQPFloat* v, OSQPFloat val, OSQPInt n){
  OSQPInt i;
  for(i = 0; i< n; i++) v[i] *= val;
}

void vec_negate(OSQPFloat* v, OSQPInt n){
  OSQPInt i;
  for(i = 0; i< n; i++) v[i] = -v[i];
}


/* CSC matrix operations implementation ------*/

/* update some or all matrix values */

void csc_update_values(OSQPCscMatrix*   M,
                       const OSQPFloat* Mx_new,
                       const OSQPInt*   Mx_new_idx,
                             OSQPInt    M_new_n) {

  OSQPInt i;

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

void csc_scale(OSQPCscMatrix* A, OSQPFloat sc){
  OSQPInt i, nnzA;
  nnzA = A->p[A->n];
  for (i = 0; i < nnzA; i++) {
    A->x[i] *= sc;
  }
}

/* A = L*A */

void csc_lmult_diag(OSQPCscMatrix* A, const OSQPFloat* d){

  OSQPInt    j, i;
  OSQPInt    n  = A->n;
  OSQPInt*   Ap = A->p;
  OSQPInt*   Ai = A->i;
  OSQPFloat* Ax = A->x;

  for (j = 0; j < n; j++) {               // Cycle over columns
    for (i = Ap[j]; i < Ap[j + 1]; i++) { // Cycle every row in the column
      Ax[i] *= d[Ai[i]];                  // Scale by corresponding element
                                          // of d for row i
    }
  }
}

/* A = A*R */

void csc_rmult_diag(OSQPCscMatrix* A, const OSQPFloat* d){

  OSQPInt    j, i;
  OSQPInt    n  = A->n;
  OSQPInt*   Ap = A->p;
  OSQPFloat* Ax = A->x;

  for (j = 0; j < n; j++) {                // Cycle over columns j
    for (i = Ap[j]; i < Ap[j + 1]; i++) {  // Cycle every row i in column j
      Ax[i] *= d[j];                       // Scale by corresponding element
                                           // of d for column j
    }
  }
}

// d = diag(At*diag(D)*A)
void csc_AtDA_extract_diag(const OSQPCscMatrix* A,
                           const OSQPFloat*     D,
                                 OSQPFloat*     d) {
  OSQPInt    j, i;
  OSQPInt    n  = A->n;
  OSQPInt*   Ap = A->p;
  OSQPInt*   Ai = A->i;
  OSQPFloat* Ax = A->x;

  // Each entry of output vector is for a column, so cycle over columns
  for (j = 0; j < n; j++) {
    d[j] = 0;
    // Iterate over each entry in the column
    for (i = Ap[j]; i < Ap[j + 1]; i++) {
      d[j] += Ax[i]*Ax[i]*D[Ai[i]];
    }
  }
}

//y = alpha*A*x + beta*y, where A is symmetric and only triu is stored
void csc_Axpy_sym_triu(const OSQPCscMatrix* A,
                       const OSQPFloat*     x,
                             OSQPFloat*     y,
                             OSQPFloat      alpha,
                             OSQPFloat      beta) {

    OSQPInt    i, j;
    OSQPInt*   Ap = A->p;
    OSQPInt*   Ai = A->i;
    OSQPInt    An = A->n;
    OSQPInt    Am = A->m;
    OSQPFloat* Ax = A->x;

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
void csc_Axpy(const OSQPCscMatrix* A,
              const OSQPFloat*     x,
                    OSQPFloat*     y,
                    OSQPFloat      alpha,
                    OSQPFloat      beta) {

  OSQPInt    i, j;
  OSQPInt*   Ap = A->p;
  OSQPInt*   Ai = A->i;
  OSQPInt    An = A->n;
  OSQPInt    Am = A->m;
  OSQPFloat* Ax = A->x;

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

void csc_Atxpy(const OSQPCscMatrix* A,
               const OSQPFloat*     x,
                     OSQPFloat*     y,
                     OSQPFloat      alpha,
                     OSQPFloat      beta) {
  OSQPInt    j, k;
  OSQPInt    An = A->n;
  OSQPInt*   Ap = A->p;
  OSQPInt*   Ai = A->i;
  OSQPFloat* Ax = A->x;

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

// OSQPFloat csc_quad_form(const csc *P, const OSQPFloat *x) {

//   //NB:implementation assumes upper triangular part only

//   OSQPFloat quad_form = 0.;
//   OSQPInt   i, j, ptr;
//   OSQPInt*   Pp = P->p;
//   OSQPInt*   Pi = P->i;
//   OSQPFloat* Px = P->x;
//   OSQPInt    Pn = P->n;


//   for (j = 0; j < Pn; j++) {                    // Iterate over columns
//     for (ptr = Pp[j]; ptr < Pp[j + 1]; ptr++) { // Iterate over rows
//       i = Pi[ptr];                            // Row index

//       if (i == j) {                                 // Diagonal element
//         quad_form += (OSQPFloat).5 * Px[ptr] * x[i] * x[i];
//       }
//       else if (i < j) {                             // Off-diagonal element
//         quad_form += Px[ptr] * x[i] * x[j];
//       }
//       else {                                        // Element in lower triangle
//         c_eprint("quad_form matrix is not upper triangular");
//         return -1.;
//       }
//     }
//   }
//   return quad_form;
// }

/* columnwise infinity norm */

void csc_col_norm_inf(const OSQPCscMatrix* M, OSQPFloat* E) {

  OSQPInt    j, ptr;
  OSQPInt*   Mp = M->p;
  OSQPInt    Mn = M->n;
  OSQPFloat* Mx = M->x;

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

void csc_row_norm_inf(const OSQPCscMatrix* M, OSQPFloat* E) {

  OSQPInt    i, j, ptr;
  OSQPInt*   Mp = M->p;
  OSQPInt*   Mi = M->i;
  OSQPInt    Mn = M->n;
  OSQPInt    Mm = M->m;
  OSQPFloat* Mx = M->x;

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

void csc_row_norm_inf_sym_triu(const OSQPCscMatrix* M, OSQPFloat* E) {

  OSQPInt   i, j, ptr;
  OSQPInt*   Mp = M->p;
  OSQPInt*   Mi = M->i;
  OSQPInt    Mn = M->n;
  OSQPInt    Mm = M->m;
  OSQPFloat* Mx = M->x;
  OSQPFloat  abs_x;

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
