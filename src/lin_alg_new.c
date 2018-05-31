#include "lin_alg.h"


/* VECTOR FUNCTIONS ----------------------------------------------------------*/


#ifndef EMBEDDED

OsqpVectorf* OsqpVectorf_copy_new(OsqpVectorf *a){

  OsqpVectorf *b;
  b         = c_malloc(sizeof(OsqpVectorf));
  b->length = a->length;
  b->values = c_malloc(b->length * sizeof(c_float));

  OsqpVectorf_copy(a,b);

  return b;
}

OsqpVectori* OsqpVectori_copy_new(OsqpVectori *a){

  OsqpVectori *b;
  b         = c_malloc(sizeof(OsqpVectori));
  b->length = a->length;
  b->values = c_malloc(b->length * sizeof(c_int));

  OsqpVectori_copy(a,b);
}


#endif // end EMBEDDED

void OsqpVectorf_copy(OsqpVectorf *a,OsqpVectorf *b){

  c_int i;

  assert(a->length == b->length);

  for (i = 0; i < b->length; i++) {
    b->values[i] = a->values[i];
  }
  return b;
}

void OsqpVectori_copy(OsqpVectori *a,OsqpVectori *b){

  c_int i;

  assert(a->length == b->length);

  for (i = 0; i < b->length; i++) {
    b->values[i] = a->values[i];
  }
  return b;
}

void OsqpVectorf_set_scalar(OsqpVectorf *a, c_float sc){
  c_int i;
  for (i = 0; i < a->length; i++) {
    a->values[i] = sc;
  }
}

void OsqpVectori_set_scalar(OsqpVectori *a, c_int sc){
  c_int i;
  for (i = 0; i < a->length; i++) {
    a->values[i] = sc;
  }
}

void OsqpVectorf_add_scalar(OsqpVectorf *a, c_float sc){
  c_int i;
  for (i = 0; a->length < n; i++) {
    a->values[i] += sc;
  }
}

void OsqpVectori_add_scalar(OsqpVectori *a, c_int sc){
  c_int i;
  for (i = 0; a->length < n; i++) {
    a->values[i] += sc;
  }
}


void OsqpVectorf_mult_scalar(OsqpVectorf *a, c_float sc){
  c_int i;
  for (i = 0; i < a->length; i++) {
    a->values[i] *= sc;
  }
}


void OsqpVectorf_add_scaled(OsqpVectorf       *c,
                             const OsqpVectorf *a,
                             const OsqpVectorf *b,
                             c_float           sc){
  c_int i;

  assert(a->length == b->length);
  assert(a->length == c->length);

  for (i = 0; i < n; i++) {
    c->values[i] =  a->values[i] + sc * b->values[i];
  }
}

c_float OsqpVectorf_norm_inf(const OsqpVectorf *v){

  c_int   i;
  c_float normval = 0.0;
  c_float absval;

  for (i = 0; i < v->length; i++) {
    absval = c_absval(v->values[i]);
    if (absval > normval) normval = absval;
  }
  return normval;
}

c_float OsqpVectorf_norm_1(const OsqpVectorf *v){

  c_int   i;
  c_float absval;
  c_float normval = 0.0;

  for (i = 0; i < v->length; i++) {
    normval += c_absval(v->values[i]);
  }
  return normval;
}

c_float OsqpVectorf_scaled_norm_inf(const OsqpVectorf *S, const OsqpVectorf *v){

  c_int   i;
  c_float absval;
  c_float normval = 0.0;

  assert(S->length == v->length);

  for (i = 0; i < v->length; i++) {
    absval = c_absval(S[i] * v[i]);
    if (absval > normval) normval = absval;
  }
  return normval;
}

c_float OsqpVectorf_scaled_norm_1(const OsqpVectorf *S, const OsqpVectorf *v){

  c_int   i;
  c_float normval = 0.0;

  assert(S->length == v->length);

  for (i = 0; i < v->length; i++) {
    normval += c_absval(S[i] * v[i]);
  }
  return normval;
}


c_float OsqpVectorf_norm_inf_diff(const OsqpVectorf *a,
                                  const OsqpVectorf *b){
  c_int   i;
  c_float absval;
  c_float normDiff = 0.0;

  assert(a->length == b->length);

  for (i = 0; i < a->length; i++) {
    absval = c_absval(a[i] - b[i]);
    if (absval > normDiff) normDiff = absval;
  }
  return normDiff;
}

c_float OsqpVectorf_norm_1_diff(const OsqpVectorf *a,
                                const OsqpVectorf *b){

  c_int   i;
  c_float normDiff = 0.0;

  assert(a->length == b->length);

  for (i = 0; i < a->length; i++) {
    absval = c_absval(a[i] - b[i]);
    normDiff += absval;
  }
  return normDiff;
}


c_float OsqpVectorf_sum(const OsqpVectorf *a){

  c_int   i;
  c_float val = 0.0;

  for (i = 0; i < n; i++) {
    val += a->values[i];
  }

  return val;
}

c_float OsqpVectorf_mean(const OsqpVectorf *a){

  if(a->length){
    return OsqpVectorf_sum(a)/(a->length);
  }
  else{
    return 0;
  }
}

c_float OsqpVectorf_dot_prod(const OsqpVectorf *a, const OsqpVectorf *b){

  c_int   i; // Index
  c_float dotprod = 0.0;

  assert(a->length == b->length);

  for (i = 0; i < a->length; i++) {
    dotprod += a->values[i] * b->values[i];
  }

  return dotprod;
}

void OsqpVectorf_ew_prod(const OsqpVectorf *a,
                         const OsqpVectorf *b,
                               OsqpVectorf *c){

    c_int i;

    assert(a->length == b->length);
    assert(a->length == c->length);

    for (i = 0; i < a->length; i++) {
      c->values[i] = a->values[i] * b->values[i];
    }
}



#if EMBEDDED != 1

void OsqpVectorf_ew_reciprocal(const OsqpVectorf *a, OsqpVectorf *b){

  c_int i;

  assert(a->length == b->length);

  for (i = 0; i < a->length; i++) {
    b->values[i] = (c_float)1.0 / a->values[i];
  }
}


void OsqpVectorf_ew_sqrt(OsqpVectorf *a){

  c_int i;

  for (i = 0; i < a->length; i++) {
    a->values[i] = c_sqrt(a->values[i]);
  }
}

void OsqpVectorf_ew_max(OsqpVectorf *a, c_float max_val){

  c_int i;

  for (i = 0; i < a->length; i++) {
    a->values[i] = c_max(a->values[i], max_val);
  }
}

void OsqpVectorf_ew_min(OsqpVectorf *a, c_float min_val){

  c_int i;

  for (i = 0; i < a->length; i++) {
    a->values[i] = c_min(a->values[i], max_val);
  }
}

void OsqpVectorf_ew_max_vec(const OsqpVectorf *a,
                            const OsqpVectorf *b,
                            OsqpVectorf       *c){
  c_int i;

  assert(a->length == b->length);
  assert(a->length == c->length);

  for (i = 0; i < a->length; i++) {
    c->values[i] = c_max(a->values[i], b->values[i]);
  }
}

void OsqpVectorf_ew_min_vec(const OsqpVectorf *a,
                            const OsqpVectorf *b,
                            OsqpVectorf       *c){
  c_int i;

  assert(a->length == b->length);
  assert(a->length == c->length);

  for (i = 0; i < a->length; i++) {
    c->values[i] = c_min(a->values[i], b->values[i]);
  }
}

#endif // EMBEDDED != 1


/* MATRIX FUNCTIONS ----------------------------------------------------------*/

/* multiply scalar to matrix */
void mat_mult_scalar(csc *A, c_float sc) {
  c_int i, nnzA;

  nnzA = A->p[A->n];

  for (i = 0; i < nnzA; i++) {
    A->x[i] *= sc;
  }
}

void mat_premult_diag(csc *A, const c_float *d) {
  c_int j, i;

  for (j = 0; j < A->n; j++) {                // Cycle over columns
    for (i = A->p[j]; i < A->p[j + 1]; i++) { // Cycle every row in the column
      A->x[i] *= d[A->i[i]];                  // Scale by corresponding element
                                              // of d for row i
    }
  }
}

void mat_postmult_diag(csc *A, const c_float *d) {
  c_int j, i;

  for (j = 0; j < A->n; j++) {                // Cycle over columns j
    for (i = A->p[j]; i < A->p[j + 1]; i++) { // Cycle every row i in column j
      A->x[i] *= d[j];                        // Scale by corresponding element
                                              // of d for column j
    }
  }
}

void mat_vec(const csc *A, const c_float *x, c_float *y, c_int plus_eq) {
  c_int i, j;

  if (!plus_eq) {
    // y = 0
    for (i = 0; i < A->m; i++) {
      y[i] = 0;
    }
  }

  // if A is empty
  if (A->p[A->n] == 0) {
    return;
  }

  if (plus_eq == -1) {
    // y -=  A*x
    for (j = 0; j < A->n; j++) {
      for (i = A->p[j]; i < A->p[j + 1]; i++) {
        y[A->i[i]] -= A->x[i] * x[j];
      }
    }
  } else {
    // y +=  A*x
    for (j = 0; j < A->n; j++) {
      for (i = A->p[j]; i < A->p[j + 1]; i++) {
        y[A->i[i]] += A->x[i] * x[j];
      }
    }
  }
}

void mat_tpose_vec(const csc *A, const c_float *x, c_float *y,
                   c_int plus_eq, c_int skip_diag) {
  c_int i, j, k;

  if (!plus_eq) {
    // y = 0
    for (i = 0; i < A->n; i++) {
      y[i] = 0;
    }
  }

  // if A is empty
  if (A->p[A->n] == 0) {
    return;
  }

  if (plus_eq == -1) {
    // y -=  A*x
    if (skip_diag) {
      for (j = 0; j < A->n; j++) {
        for (k = A->p[j]; k < A->p[j + 1]; k++) {
          i     = A->i[k];
          y[j] -= i == j ? 0 : A->x[k] * x[i];
        }
      }
    } else {
      for (j = 0; j < A->n; j++) {
        for (k = A->p[j]; k < A->p[j + 1]; k++) {
          y[j] -= A->x[k] * x[A->i[k]];
        }
      }
    }
  } else {
    // y +=  A*x
    if (skip_diag) {
      for (j = 0; j < A->n; j++) {
        for (k = A->p[j]; k < A->p[j + 1]; k++) {
          i     = A->i[k];
          y[j] += i == j ? 0 : A->x[k] * x[i];
        }
      }
    } else {
      for (j = 0; j < A->n; j++) {
        for (k = A->p[j]; k < A->p[j + 1]; k++) {
          y[j] += A->x[k] * x[A->i[k]];
        }
      }
    }
  }
}

#if EMBEDDED != 1
void mat_inf_norm_cols(const csc *M, c_float *E) {
  c_int j, ptr;

  // Initialize zero max elements
  for (j = 0; j < M->n; j++) {
    E[j] = 0.;
  }

  // Compute maximum across columns
  for (j = 0; j < M->n; j++) {
    for (ptr = M->p[j]; ptr < M->p[j + 1]; ptr++) {
      E[j] = c_max(c_absval(M->x[ptr]), E[j]);
    }
  }
}

void mat_inf_norm_rows(const csc *M, c_float *E) {
  c_int i, j, ptr;

  // Initialize zero max elements
  for (j = 0; j < M->m; j++) {
    E[j] = 0.;
  }

  // Compute maximum across rows
  for (j = 0; j < M->n; j++) {
    for (ptr = M->p[j]; ptr < M->p[j + 1]; ptr++) {
      i    = M->i[ptr];
      E[i] = c_max(c_absval(M->x[ptr]), E[i]);
    }
  }
}

void mat_inf_norm_cols_sym_triu(const csc *M, c_float *E) {
  c_int   i, j, ptr;
  c_float abs_x;

  // Initialize zero max elements
  for (j = 0; j < M->n; j++) {
    E[j] = 0.;
  }

  // Compute maximum across columns
  // Note that element (i, j) contributes to
  // -> Column j (as expected in any matrices)
  // -> Column i (which is equal to row i for symmetric matrices)
  for (j = 0; j < M->n; j++) {
    for (ptr = M->p[j]; ptr < M->p[j + 1]; ptr++) {
      i     = M->i[ptr];
      abs_x = c_absval(M->x[ptr]);
      E[j]  = c_max(abs_x, E[j]);

      if (i != j) {
        E[i] = c_max(abs_x, E[i]);
      }
    }
  }
}

#endif /* if EMBEDDED != 1 */


c_float quad_form(const csc *P, const c_float *x) {
  c_float quad_form = 0.;
  c_int   i, j, ptr;                                // Pointers to iterate over
                                                    // matrix: (i,j) a element
                                                    // pointer

  for (j = 0; j < P->n; j++) {                      // Iterate over columns
    for (ptr = P->p[j]; ptr < P->p[j + 1]; ptr++) { // Iterate over rows
      i = P->i[ptr];                                // Row index

      if (i == j) {                                 // Diagonal element
        quad_form += (c_float).5 * P->x[ptr] * x[i] * x[i];
      }
      else if (i < j) {                             // Off-diagonal element
        quad_form += P->x[ptr] * x[i] * x[j];
      }
      else {                                        // Element in lower diagonal
                                                    // part
#ifdef PRINTING
        c_eprint("quad_form matrix is not upper triangular");
#endif /* ifdef PRINTING */
        return OSQP_NULL;
      }
    }
  }
  return quad_form;
}
