#include "csc.h"


CscMatrix* CscMatrix_matrix(c_int m, c_int n, c_int nzmax, c_float *x, c_int *i, c_int *p)
{
  CscMatrix *M = (CscMatrix *)c_malloc(sizeof(CscMatrix));

  if (!M) return OSQP_NULL;

  M->m     = m;
  M->n     = n;
  M->nnz   = -1;
  M->nzmax = nzmax;
  M->x     = x;
  M->i     = i;
  M->p     = p;
  return M;
}

CscMatrix* CscMatrix_spalloc(c_int m, c_int n, c_int nzmax, c_int values, c_int triplet) {
  CscMatrix *A = c_calloc(1, sizeof(CscMatrix)); /* allocate the CscMatrix struct */

  if (!A) return OSQP_NULL;            /* out of memory */

  A->m     = m;                        /* define dimensions and nzmax */
  A->n     = n;
  A->nzmax = nzmax = c_max(nzmax, 1);
  A->nnz   = triplet ? 0 : -1;         /* allocate triplet or comp.col */
  A->p     = c_calloc(triplet ? nzmax : n + 1, sizeof(c_int));
  A->i     = c_calloc(nzmax,  sizeof(c_int));
  A->x     = values ? c_calloc(nzmax,  sizeof(c_float)) : OSQP_NULL;
  if (!A->p || !A->i || (values && !A->x)){
    CscMatrix_spfree(A);
    return OSQP_NULL;
  } else return A;
}

void CscMatrix_spfree(CscMatrix *A) {
  if (A){
    if (A->p) c_free(A->p);
    if (A->i) c_free(A->i);
    if (A->x) c_free(A->x);
    c_free(A);
  }
}

CscMatrix* triplet_to_CscMatrix(const CscMatrix *T, c_int *TtoC) {
  c_int m, n, nnz, p, k, *Cp, *Ci, *w, *Ti, *Tj;
  c_float *Cx, *Tx;
  CscMatrix     *C;

  m  = T->m;
  n  = T->n;
  Ti = T->i;
  Tj = T->p;
  Tx = T->x;
  nnz = T->nnz;
  C  = CscMatrix_spalloc(m, n, nnz, Tx != OSQP_NULL, 0);     /* allocate result */
  w  = c_calloc(n, sizeof(c_int));                  /* get workspace */

  if (!C || !w) return CscMatrix_done(C, w, OSQP_NULL, 0);  /* out of memory */

  Cp = C->p;
  Ci = C->i;
  Cx = C->x;

  for (k = 0; k < nnz; k++) w[Tj[k]]++;  /* column counts */
  CscMatrix_cumsum(Cp, w, n);                 /* column pointers */

  for (k = 0; k < nnz; k++) {
    Ci[p = w[Tj[k]]++] = Ti[k];         /* A(i,j) is the pth entry in C */

    if (Cx) {
      Cx[p] = Tx[k];

      if (TtoC != OSQP_NULL) TtoC[k] = p;  // Assign vector of indices
    }
  }
  return CscMatrix_done(C, w, OSQP_NULL, 1);     /* success; free w and return C */
}

CscMatrix* triplet_to_csr(const CscMatrix *T, c_int *TtoC) {
  c_int m, n, nnz, p, k, *Cp, *Cj, *w, *Ti, *Tj;
  c_float *Cx, *Tx;
  CscMatrix     *C;

  m  = T->m;
  n  = T->n;
  Ti = T->i;
  Tj = T->p;
  Tx = T->x;
  nnz = T->nnz;
  C  = CscMatrix_spalloc(m, n, nnz, Tx != OSQP_NULL, 0);     /* allocate result */
  w  = c_calloc(m, sizeof(c_int));                  /* get workspace */

  if (!C || !w) return CscMatrix_done(C, w, OSQP_NULL, 0);  /* out of memory */

  Cp = C->p;
  Cj = C->i;
  Cx = C->x;

  for (k = 0; k < nnz; k++) w[Ti[k]]++;  /* row counts */
  CscMatrix_cumsum(Cp, w, m);                 /* row pointers */

  for (k = 0; k < nnz; k++) {
    Cj[p = w[Ti[k]]++] = Tj[k];         /* A(i,j) is the pth entry in C */

    if (Cx) {
      Cx[p] = Tx[k];

      if (TtoC != OSQP_NULL) TtoC[k] = p;  // Assign vector of indices
    }
  }
  return CscMatrix_done(C, w, OSQP_NULL, 1);     /* success; free w and return C */
}

c_int CscMatrix_cumsum(c_int *p, c_int *c, c_int n) {
  c_int i, nnz = 0;

  if (!p || !c) return -1;  /* check inputs */

  for (i = 0; i < n; i++)
  {
    p[i] = nnz;
    nnz  += c[i];
    c[i] = p[i];
  }
  p[n] = nnz;
  return nnz; /* return sum (c [0..n-1]) */
}

c_int* CscMatrix_pinv(c_int const *p, c_int n) {
  c_int k, *pinv;

  if (!p) return OSQP_NULL;                /* p = OSQP_NULL denotes identity */

  pinv = c_calloc(n, sizeof(c_int));     /* allocate result */

  if (!pinv) return OSQP_NULL;             /* out of memory */

  for (k = 0; k < n; k++) pinv[p[k]] = k;  /* invert the permutation */
  return pinv;                             /* return result */
}

CscMatrix* CscMatrix_symperm(const CscMatrix *A, const c_int *pinv, c_int *AtoC, c_int values) {
  c_int i, j, p, q, i2, j2, n, *Ap, *Ai, *Cp, *Ci, *w;
  c_float *Cx, *Ax;
  CscMatrix     *C;

  n  = A->n;
  Ap = A->p;
  Ai = A->i;
  Ax = A->x;
  C  = CscMatrix_spalloc(n, n, Ap[n], values && (Ax != OSQP_NULL),
                   0);                                /* alloc result*/
  w = c_calloc(n, sizeof(c_int));                   /* get workspace */

  if (!C || !w) return CscMatrix_done(C, w, OSQP_NULL, 0);  /* out of memory */

  Cp = C->p;
  Ci = C->i;
  Cx = C->x;

  for (j = 0; j < n; j++)    /* count entries in each column of C */
  {
    j2 = pinv ? pinv[j] : j; /* column j of A is column j2 of C */

    for (p = Ap[j]; p < Ap[j + 1]; p++) {
      i = Ai[p];

      if (i > j) continue;     /* skip lower triangular part of A */
      i2 = pinv ? pinv[i] : i; /* row i of A is row i2 of C */
      w[c_max(i2, j2)]++;      /* column count of C */
    }
  }
  CscMatrix_cumsum(Cp, w, n);        /* compute column pointers of C */

  for (j = 0; j < n; j++) {
    j2 = pinv ? pinv[j] : j;   /* column j of A is column j2 of C */

    for (p = Ap[j]; p < Ap[j + 1]; p++) {
      i = Ai[p];

      if (i > j) continue;                             /* skip lower triangular
                                                          part of A*/
      i2                         = pinv ? pinv[i] : i; /* row i of A is row i2
                                                          of C */
      Ci[q = w[c_max(i2, j2)]++] = c_min(i2, j2);

      if (Cx) Cx[q] = Ax[p];

      if (AtoC) { // If vector AtoC passed, store values of the mappings
        AtoC[p] = q;
      }
    }
  }
  return CscMatrix_done(C, w, OSQP_NULL, 1); /* success; free workspace, return C */
}

CscMatrix* copy_CscMatrix_mat(const CscMatrix *A) {
  CscMatrix *B = CscMatrix_spalloc(A->m, A->n, A->p[A->n], 1, 0);

  if (!B) return OSQP_NULL;

  prea_int_vec_copy(A->p, B->p, A->n + 1);
  prea_int_vec_copy(A->i, B->i, A->p[A->n]);
  prea_vec_copy(A->x, B->x, A->p[A->n]);

  return B;
}

void prea_copy_CscMatrix_mat(const CscMatrix *A, CscMatrix *B) {
  prea_int_vec_copy(A->p, B->p, A->n + 1);
  prea_int_vec_copy(A->i, B->i, A->p[A->n]);
  prea_vec_copy(A->x, B->x, A->p[A->n]);

  B->nzmax = A->nzmax;
}

CscMatrix* CscMatrix_done(CscMatrix *C, void *w, void *x, c_int ok) {
  c_free(w);                   /* free workspace */
  c_free(x);
  if (ok) return C;
  else {
    CscMatrix_spfree(C);
    return OSQP_NULL;
  }
}

CscMatrix* CscMatrix_to_triu(CscMatrix *M) {
  CscMatrix  *M_trip;    // Matrix in triplet format
  CscMatrix  *M_triu;    // Resulting upper triangular matrix
  c_int nnzorigM;  // Number of nonzeros from original matrix M
  c_int nnzmaxM;   // Estimated maximum number of elements of upper triangular M
  c_int n;         // Dimension of M
  c_int ptr, i, j; // Counters for (i,j) and index in M
  c_int z_M = 0;   // Counter for elements in M_trip


  // Check if matrix is square
  if (M->m != M->n) {
#ifdef PRINTING
    c_eprint("Matrix M not square");
#endif /* ifdef PRINTING */
    return OSQP_NULL;
  }
  n = M->n;

  // Get number of nonzeros full M
  nnzorigM = M->p[n];

  // Estimate nnzmaxM
  // Number of nonzero elements in original M + diagonal part.
  // -> Full matrix M as input: estimate is half the number of total elements +
  // diagonal = .5 * (nnzorigM + n)
  // -> Upper triangular matrix M as input: estimate is the number of total
  // elements + diagonal = nnzorigM + n
  // The maximum between the two is nnzorigM + n
  nnzmaxM = nnzorigM + n;

  // OLD
  // nnzmaxM = n*(n+1)/2;  // Full upper triangular matrix (This version
  // allocates too much memory!)
  // nnzmaxM = .5 * (nnzorigM + n);  // half of the total elements + diagonal

  // Allocate M_trip
  M_trip = CscMatrix_spalloc(n, n, nnzmaxM, 1, 1); // Triplet format

  if (!M_trip) {
#ifdef PRINTING
    c_eprint("Upper triangular matrix extraction failed (out of memory)");
#endif /* ifdef PRINTING */
    return OSQP_NULL;
  }

  // Fill M_trip with only elements in M which are in the upper triangular
  for (j = 0; j < n; j++) { // Cycle over columns
    for (ptr = M->p[j]; ptr < M->p[j + 1]; ptr++) {
      // Get row index
      i = M->i[ptr];

      // Assign element only if in the upper triangular
      if (i <= j) {
        // c_print("\nM(%i, %i) = %.4f", M->i[ptr], j, M->x[ptr]);

        M_trip->i[z_M] = i;
        M_trip->p[z_M] = j;
        M_trip->x[z_M] = M->x[ptr];

        // Increase counter for the number of elements
        z_M++;
      }
    }
  }

  // Set number of nonzeros
  M_trip->nnz = z_M;

  // Convert triplet matrix to CscMatrix format
  M_triu = triplet_to_CscMatrix(M_trip, OSQP_NULL);

  // Assign number of nonzeros of full matrix to triu M
  M_triu->nzmax = nnzmaxM;

  // Cleanup and return result
  CscMatrix_spfree(M_trip);

  // Return matrix in triplet form
  return M_triu;
}



/* matrix times scalar */

void CscMatrix_scale(CscMatrix* A, c_float sc){
  c_int i, nnzA;
  nnzA = A->p[A->n];
  for (i = 0; i < nnzA; i++) {
    A->x[i] *= sc;
  }
}

void CscMatrix_lmult_diag(CscMatrix* A, const OSQPVectorf *L){

  c_int j, i;
  c_float* d  = OSQPVectorf_data(L);
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

void CscMatrix_rmult_diag(CscMatrix* A, const OSQPVectorf *R){

  c_int j, i;
  c_float* d  = OSQPVectorf_data(R);
  c_int*   Ap = A->p;
  c_int*   Ai = A->i;
  c_float* Ax = A->x;
  c_int     n = A->n;

  for (j = 0; j < n; j++) {                // Cycle over columns j
    for (i = Ap[j]; i < Ap[j + 1]; i++) {  // Cycle every row i in column j
      Ax[i] *= d[j];                       // Scale by corresponding element
                                           // of d for column j
    }
  }
}

//y = alpha*A*x + beta*y
void CscMatrix_Axpy(const CscMatrix   *A,
                    const OSQPVectorf *x,
                    OSQPVectorf *y,
                    c_float alpha,
                    c_float beta) {

  c_int i, j;
  c_int*   Ap = A->p;
  c_int*   Ai = A->i;
  c_float* Ax = A->x;
  c_int    An = A->n;
  c_float* xv = OSQPVectorf_data(x);
  c_float* yv = OSQPVectorf_data(y);

  // first do the b*y part
  if (beta == 0)        OSQPVectorf_set_scalar(y,0.0);
  else if (beta ==  1)  ; //do nothing
  else if (beta == -1)  OSQPVectorf_negate(y);
  else OSQPVectorf_mult_scalar(y,beta);


  // if A is empty or zero
  if (Ap[An] == 0 || alpha == 0.0) return;

  if (alpha == -1) {
    // y -= A*x
    for (j = 0; j < An; j++) {
      for (i = Ap[j]; i < Ap[j + 1]; i++) {
        yv[Ai[i]] -= Ax[i] * xv[j];
    }}}

  else if(alpha == 1){
    // y +=  A*x
    for (j = 0; j < An; j++) {
      for (i = Ap[j]; i < Ap[j + 1]; i++) {
        yv[Ai[i]] += Ax[i] * xv[j];
    }}}

  else{
    // y +=  alpha*A*x
    for (j = 0; j < An; j++) {
      for (i = Ap[j]; i < Ap[j + 1]; i++) {
        yv[Ai[i]] += alpha*Ax[i] * xv[j];
    }}}
}

void CscMatrix_Atxpy(const CscMatrix *A, const OSQPVectorf *x, OSQPVectorf *y,
                     c_float alpha, c_float beta, c_int skip_diag) {
  c_int i, j, k;
  c_int*   Ap = A->p;
  c_int*   Ai = A->i;
  c_float* Ax = A->x;
  c_int    An = A->n;
  c_float* xv = OSQPVectorf_data(x);
  c_float* yv = OSQPVectorf_data(y);

  // first do the b*y part
  if (beta == 0)        OSQPVectorf_set_scalar(y,0.0);
  else if (beta ==  1)  ; //do nothing
  else if (beta == -1)  OSQPVectorf_negate(y);
  else OSQPVectorf_mult_scalar(y,beta);

  // if A is empty or alpha = 0
  if (Ap[An] == 0 || alpha == 0.0) {
    return;
  }

  if(skip_diag){

      if(alpha == 1){
        for (j = 0; j < A->n; j++) {
          for (k = Ap[j]; k < Ap[j + 1]; k++) {
            yv[j] -= i == j ? 0 : Ax[k] * xv[Ai[k]];
      }}}

      else if(alpha == -1){
        for (j = 0; j < A->n; j++) {
          for (k = Ap[j]; k < Ap[j + 1]; k++) {
            yv[j] += i == j ? 0 : Ax[k] * xv[Ai[k]];
      }}}

      else{
        for (j = 0; j < A->n; j++) {
          for (k = Ap[j]; k < Ap[j + 1]; k++) {
            yv[j] += i == j ? 0 : alpha * Ax[k] * xv[Ai[k]];
      }}}
  }

  else {  //not skipping the diagonal

    if(alpha == 1){
      for (j = 0; j < A->n; j++) {
        for (k = Ap[j]; k < Ap[j + 1]; k++) {
          yv[j] -= Ax[k] * xv[Ai[k]];
    }}}

    else if(alpha == -1){
      for (j = 0; j < A->n; j++) {
        for (k = Ap[j]; k < Ap[j + 1]; k++) {
          yv[j] += Ax[k] * xv[Ai[k]];
    }}}

    else{
      for (j = 0; j < A->n; j++) {
        for (k = Ap[j]; k < Ap[j + 1]; k++) {
          yv[j] += alpha*Ax[k] * xv[Ai[k]];
    }}}
  }
}

c_float CscMatrix_quad_form(const CscMatrix *P, const OSQPVectorf *x) {

  //NB:implementation assumes upper triangular part only

  c_float quad_form = 0.;
  c_int   i, j, ptr;
  c_float* xv = OSQPVectorf_data(x);
  c_int*   Pp = P->p;
  c_int*   Pi = P->i;
  c_float* Px = P->x;
  c_int    Pn = P->n;


  for (j = 0; j < Pn; j++) {                    // Iterate over columns
    for (ptr = Pp[j]; ptr < Pp[j + 1]; ptr++) { // Iterate over rows
      i = Pi[ptr];                            // Row index

      if (i == j) {                                 // Diagonal element
        quad_form += (c_float).5 * Px[ptr] * xv[i] * xv[i];
      }
      else if (i < j) {                             // Off-diagonal element
        quad_form += Px[ptr] * xv[i] * xv[j];
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


#if EMBEDDED != 1

void CscMatrix_col_norm_inf(const CscMatrix *M, OSQPVectorf *E) {

  c_int j, ptr;
  c_float* Ev = OSQPVectorf_data(E);
  c_int*   Mp = M->p;
  c_int*   Mi = M->i;
  c_float* Mx = M->x;
  c_int    Mn = M->n;

  // Initialize zero max elements
  OSQPVectorf_set_scalar(E,0.0);

  // Compute maximum across columns
  for (j = 0; j < Mn; j++) {
    for (ptr = Mp[j]; ptr < Mp[j + 1]; ptr++) {
      Ev[j] = c_max(c_absval(Mx[ptr]), Ev[j]);
    }
  }
}

void CscMatrix_row_norm_inf(const CscMatrix *M, OSQPVectorf *E) {

  c_int i, j, ptr;
  c_float* Ev = OSQPVectorf_data(E);
  c_int*   Mp = M->p;
  c_int*   Mi = M->i;
  c_float* Mx = M->x;
  c_int    Mn = M->n;

  // Initialize zero max elements
  OSQPVectorf_set_scalar(E,0.0);

  // Compute maximum across rows
  for (j = 0; j < Mn; j++) {
    for (ptr = Mp[j]; ptr < Mp[j + 1]; ptr++) {
      i    = Mi[ptr];
      Ev[i] = c_max(c_absval(Mx[ptr]), Ev[i]);
    }
  }
}

void CscMatrix_row_norm_inf_sym_triu(const CscMatrix *M, OSQPVectorf *E) {

  c_int   i, j, ptr;
  c_float* Ev = OSQPVectorf_data(E);
  c_int*   Mp = M->p;
  c_int*   Mi = M->i;
  c_float* Mx = M->x;
  c_int    Mn = M->n;
  c_float abs_x;

  // Initialize zero max elements
  OSQPVectorf_set_scalar(E,0.0);

  // Compute maximum across columns
  // Note that element (i, j) contributes to
  // -> Column j (as expected in any matrices)
  // -> Column i (which is equal to row i for symmetric matrices)
  for (j = 0; j < Mn; j++) {
    for (ptr = Mp[j]; ptr < Mp[j + 1]; ptr++) {
      i     = Mi[ptr];
      abs_x = c_absval(Mx[ptr]);
      Ev[j]  = c_max(abs_x, Ev[j]);

      if (i != j) {
        Ev[i] = c_max(abs_x, Ev[i]);
      }
    }
  }
}

#endif /* if EMBEDDED != 1 */
