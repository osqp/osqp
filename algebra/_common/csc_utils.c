#include "printing.h"
#include "csc_utils.h"

//========== Logical, testing and debug ===========

OSQPInt csc_is_eq(OSQPCscMatrix* A,
                  OSQPCscMatrix* B,
                  OSQPFloat      tol) {
  OSQPInt j, i;

  // If number of columns does not coincide, they are not equal.
  if (A->n != B->n) {
      return 0;
  }

  for (j = 0; j < A->n; j++) { // Cycle over columns j
    // if column pointer of next colimn does not coincide, they are not equal
    // NB: first column always has A->p[0] = B->p[0] = 0 by construction.
    if (A->p[j+1] != B->p[j+1]) {
        return 0;
    }

    for (i = A->p[j]; i < A->p[j + 1]; i++) { // Cycle rows i in column j
      if ((A->i[i] != B->i[i]) ||             // Different row indices
          (c_absval(A->x[i] - B->x[i]) > tol)) {
        return 0;
      }
    }
  }
  return 1;
}


//========= Internal utility functions  ===========

#ifndef OSQP_EMBEDDED_MODE

static void* csc_malloc(OSQPInt n, OSQPInt size) {
  return c_malloc(n * size);
}

static void* csc_calloc(OSQPInt n, OSQPInt size) {
  return c_calloc(n, size);
}

#endif /* OSQP_EMBEDDED_MODE */

static void prea_int_vec_copy(const OSQPInt* a, OSQPInt* b, OSQPInt n) {
  OSQPInt i;
  for (i = 0; i < n; i++) b[i] = a[i];
}

static void prea_vec_copy(const OSQPFloat* a, OSQPFloat* b, OSQPInt n) {
  OSQPInt i;
  for (i = 0; i < n; i++)  b[i] = a[i];
}

static void float_vec_set_scalar(OSQPFloat* a, OSQPFloat sc, OSQPInt n) {
  OSQPInt i;
  for (i = 0; i < n; i++) a[i] = sc;
}

static void int_vec_set_scalar(OSQPInt* a, OSQPInt sc, OSQPInt n) {
  OSQPInt i;
  for (i = 0; i < n; i++) a[i] = sc;
}

OSQPInt csc_cumsum(OSQPInt* p, OSQPInt* c, OSQPInt n) {
  OSQPInt i, nz = 0;

  if (!p || !c) return -1;  /* check inputs */

  for (i = 0; i < n; i++)
  {
    p[i] = nz;
    nz  += c[i];
    c[i] = p[i];
  }
  p[n] = nz;
  return nz; /* return sum (c [0..n-1]) */
}

//==================================================

// OSQPCscMatrix* csc_matrix(OSQPInt m, OSQPInt n, OSQPInt nzmax, OSQPFloat *x, OSQPInt *i, OSQPInt *p)
// {
//   OSQPCscMatrix *M = (OSQPCscMatrix *)c_malloc(sizeof(OSQPCscMatrix));

//   if (!M) return OSQP_NULL;

//   M->m     = m;
//   M->n     = n;
//   M->nz    = -1;
//   M->nzmax = nzmax;
//   M->x     = x;
//   M->i     = i;
//   M->p     = p;
//   return M;
// }

#ifndef OSQP_EMBEDDED_MODE

OSQPCscMatrix* csc_spalloc(OSQPInt m,
                           OSQPInt n,
                           OSQPInt nzmax,
                           OSQPInt values,
                           OSQPInt triplet) {
  OSQPCscMatrix* A = c_calloc(1, sizeof(OSQPCscMatrix)); /* allocate the OSQPCscMatrix struct */

  if (!A) return OSQP_NULL;            /* out of memory */

  A->m     = m;                        /* define dimensions and nzmax */
  A->n     = n;
  A->nzmax = nzmax = c_max(nzmax, 0);
  A->nz    = triplet ? 0 : -1;         /* allocate triplet or comp.col */
  A->p     = csc_malloc(triplet ? nzmax : n + 1, sizeof(OSQPInt));
  A->i     = values ? csc_malloc(nzmax,  sizeof(OSQPInt)) : OSQP_NULL;
  A->x     = values ? csc_malloc(nzmax,  sizeof(OSQPFloat)) : OSQP_NULL;
  if (!A->p || (values && !A->i ) || (values && !A->x)){
    csc_spfree(A);
    return OSQP_NULL;
  } else return A;
}

void csc_spfree(OSQPCscMatrix* A) {
  if (A){
    if (A->p) c_free(A->p);
    if (A->i) c_free(A->i);
    if (A->x) c_free(A->x);
    c_free(A);
  }
}

OSQPCscMatrix* csc_submatrix_byrows(const OSQPCscMatrix* A,
                                          OSQPInt*       rows){

  OSQPInt        j;
  OSQPCscMatrix* R;
  OSQPInt        nzR = 0;
  OSQPInt        An = A->n;
  OSQPInt        Am = A->m;
  OSQPInt*       Ap = A->p;
  OSQPInt*       Ai = A->i;
  OSQPFloat*     Ax = A->x;
  OSQPInt*       Rp;
  OSQPInt*       Ri;
  OSQPFloat*     Rx;
  OSQPInt        Rm = 0;
  OSQPInt        ptr;
  OSQPInt*       rridx; //mapping from row indices to reduced row indices

  rridx = (OSQPInt*)c_malloc(Am * sizeof(OSQPInt));
  if(!rridx) return OSQP_NULL;

  //count the number of rows in the reduced
  //matrix, and build an index from the input
  //matrix rows to the reduced rows
  Rm    = 0;
  for(j = 0; j < Am; j++){
     if(rows[j]) rridx[j] = Rm++;
  }

  // Count number of nonzeros in Ared
  for (j = 0; j < Ap[An]; j++) {
    if(rows[A->i[j]]) nzR++;
  }

  // Form R = A(rows,:), where nrows = sum(rows != 0)
  R = csc_spalloc(Rm, An, nzR, 1, 0);
  if (!R) return OSQP_NULL;

  // no active constraints
  if (Rm == 0) {
    int_vec_set_scalar(R->p, 0, An + 1);
  }

  else{
    nzR = 0; // reset counter
    Rp = R->p;
    Ri = R->i;
    Rx = R->x;

    for (j = 0; j < An; j++) { // Cycle over columns of A
      Rp[j] = nzR;
      for (ptr = Ap[j]; ptr < Ap[j + 1]; ptr++) {
        // Cycle over elements in j-th column
        if (rows[A->i[ptr]]) {
          Ri[nzR] = rridx[Ai[ptr]];
          Rx[nzR] = Ax[ptr];
          nzR++;
    }}}
    // Update the last element in R->p
    Rp[An] = nzR;
  }

  c_free(rridx); //free internal work index

  return R;
}

OSQPCscMatrix* triplet_to_csc(const OSQPCscMatrix* T, OSQPInt* TtoC) {

  OSQPInt    m, n, nz, p, k;
  OSQPInt*   Cp;
  OSQPInt*   Ci;
  OSQPInt*   w;
  OSQPInt*   Ti;
  OSQPInt*   Tj;
  OSQPFloat* Cx;
  OSQPFloat* Tx;
  OSQPCscMatrix* C;

  m  = T->m;
  n  = T->n;
  Ti = T->i;
  Tj = T->p;
  Tx = T->x;
  nz = T->nz;
  C  = csc_spalloc(m, n, nz, Tx != OSQP_NULL, 0);     /* allocate result */
  w  = csc_calloc(n, sizeof(OSQPInt));                  /* get workspace */

  if (!C || !w) return csc_done(C, w, OSQP_NULL, 0);  /* out of memory */

  Cp = C->p;
  Ci = C->i;
  Cx = C->x;

  for (k = 0; k < nz; k++) w[Tj[k]]++;  /* column counts */
  csc_cumsum(Cp, w, n);                 /* column pointers */

  for (k = 0; k < nz; k++) {
    Ci[p = w[Tj[k]]++] = Ti[k];         /* A(i,j) is the pth entry in C */

    if (Cx) {
      Cx[p] = Tx[k];

      if (TtoC != OSQP_NULL) TtoC[k] = p;  // Assign vector of indices
    }
  }
  return csc_done(C, w, OSQP_NULL, 1);     /* success; free w and return C */
}

OSQPCscMatrix* triplet_to_csr(const OSQPCscMatrix* T, OSQPInt* TtoC) {
  OSQPInt    m, n, nz, p, k;
  OSQPInt*   Cp;
  OSQPInt*   Cj;
  OSQPInt*   w;
  OSQPInt*   Ti;
  OSQPInt*   Tj;
  OSQPFloat* Cx;
  OSQPFloat* Tx;
  OSQPCscMatrix* C;

  m  = T->m;
  n  = T->n;
  Ti = T->i;
  Tj = T->p;
  Tx = T->x;
  nz = T->nz;
  C  = csc_spalloc(m, n, nz, Tx != OSQP_NULL, 0);     /* allocate result */
  w  = csc_calloc(m, sizeof(OSQPInt));                  /* get workspace */

  if (!C || !w) return csc_done(C, w, OSQP_NULL, 0);  /* out of memory */

  Cp = C->p;
  Cj = C->i;
  Cx = C->x;

  for (k = 0; k < nz; k++) w[Ti[k]]++;  /* row counts */
  csc_cumsum(Cp, w, m);                 /* row pointers */

  for (k = 0; k < nz; k++) {
    Cj[p = w[Ti[k]]++] = Tj[k];         /* A(i,j) is the pth entry in C */

    if (Cx) {
      Cx[p] = Tx[k];

      if (TtoC != OSQP_NULL) TtoC[k] = p;  // Assign vector of indices
    }
  }
  return csc_done(C, w, OSQP_NULL, 1);     /* success; free w and return C */
}

#endif /* OSQP_EMBEDDED_MODE */

void csc_extract_diag(const OSQPCscMatrix* A,
                            OSQPFloat*     d) {
  OSQPInt    i, ptr;
  OSQPInt    n  = A->n;
  OSQPInt*   Ap = A->p;
  OSQPInt*   Ai = A->i;
  OSQPFloat* Ax = A->x;

  /* Initialize output vector to 0 */
  float_vec_set_scalar(d, 0.0, n);

  /* Loop over columns to find when the row index equals column index */
  for(i = 0; i < n; i++) {
    for (ptr = Ap[i]; ptr < Ap[i + 1]; ptr++) {
      if (Ai[ptr] == i)
        d[i] = Ax[ptr];
    }
  }
}

#ifndef OSQP_EMBEDDED_MODE

OSQPInt* csc_pinv(const OSQPInt* p, OSQPInt n) {
  OSQPInt  k;
  OSQPInt* pinv;

  if (!p) return OSQP_NULL;                /* p = OSQP_NULL denotes identity */

  pinv = csc_malloc(n, sizeof(OSQPInt));     /* allocate result */

  if (!pinv) return OSQP_NULL;             /* out of memory */

  for (k = 0; k < n; k++) pinv[p[k]] = k;  /* invert the permutation */
  return pinv;                             /* return result */
}

OSQPCscMatrix* csc_symperm(const OSQPCscMatrix* A,
                           const OSQPInt*       pinv,
                                 OSQPInt*       AtoC,
                                 OSQPInt        values) {
  OSQPInt    i, j, p, q, i2, j2, n;
  OSQPInt*   Ap;
  OSQPInt*   Ai;
  OSQPInt*   Cp;
  OSQPInt*   Ci;
  OSQPInt*   w;
  OSQPFloat* Cx;
  OSQPFloat* Ax;
  OSQPCscMatrix* C;

  n  = A->n;
  Ap = A->p;
  Ai = A->i;
  Ax = A->x;
  C  = csc_spalloc(n, n, Ap[n], values && (Ax != OSQP_NULL),
                   0);                                /* alloc result*/
  w = csc_calloc(n, sizeof(OSQPInt));                   /* get workspace */

  if (!C || !w) return csc_done(C, w, OSQP_NULL, 0);  /* out of memory */

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
  csc_cumsum(Cp, w, n);        /* compute column pointers of C */

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
  return csc_done(C, w, OSQP_NULL, 1); /* success; free workspace, return C */
}

OSQPCscMatrix* csc_copy(const OSQPCscMatrix* A) {
  OSQPCscMatrix* B = csc_spalloc(A->m, A->n, A->p[A->n], (A->x != OSQP_NULL), 0);

  if (!B) return OSQP_NULL;

  prea_int_vec_copy(A->p, B->p, A->n + 1);
  prea_int_vec_copy(A->i, B->i, A->p[A->n]);
  prea_vec_copy(A->x, B->x, A->p[A->n]);

  return B;
}

// void csc_copy_prea(const OSQPCscMatrix *A, OSQPCscMatrix *B) {

//   prea_int_vec_copy(A->p, B->p, A->n + 1);
//   prea_int_vec_copy(A->i, B->i, A->p[A->n]);
//   prea_vec_copy(A->x, B->x, A->p[A->n]);

//   B->nzmax = A->nzmax;
// }

OSQPFloat* csc_to_dns(OSQPCscMatrix* M)
{
  OSQPInt i, j = 0; // Predefine row index and column index
  OSQPInt idx;

  // Initialize matrix of zeros
  OSQPFloat* A = (OSQPFloat *)c_calloc(M->m * M->n, sizeof(OSQPFloat));
  if (!A) return OSQP_NULL;

  // Allocate elements
  for (idx = 0; idx < M->p[M->n]; idx++)
  {
    // Get row index i (starting from 1)
    i = M->i[idx];

    // Get column index j (increase if necessary) (starting from 1)
    while (M->p[j + 1] <= idx) {
      j++;
    }

    // Assign values to A
    A[j * (M->m) + i] = M->x[idx];
  }
  return A;
}

OSQPCscMatrix* csc_done(OSQPCscMatrix* C,
                        void*          w,
                        void*          x,
                        OSQPInt        ok) {
  c_free(w);                   /* free workspace */
  c_free(x);
  if (ok) return C;
  else {
    csc_spfree(C);
    return OSQP_NULL;
  }
}

#endif /* OSQP_EMBEDDED_MODE */

// OSQPCscMatrix* csc_to_triu(OSQPCscMatrix *M) {
//   OSQPCscMatrix  *M_trip;    // Matrix in triplet format
//   OSQPCscMatrix  *M_triu;    // Resulting upper triangular matrix
//   OSQPInt nnzorigM;  // Number of nonzeros from original matrix M
//   OSQPInt nnzmaxM;   // Estimated maximum number of elements of upper triangular M
//   OSQPInt n;         // Dimension of M
//   OSQPInt ptr, i, j; // Counters for (i,j) and index in M
//   OSQPInt z_M = 0;   // Counter for elements in M_trip


//   // Check if matrix is square
//   if (M->m != M->n) {
//     c_eprint("Matrix M not square");
//     return OSQP_NULL;
//   }
//   n = M->n;

//   // Get number of nonzeros full M
//   nnzorigM = M->p[n];

//   // Estimate nnzmaxM
//   // Number of nonzero elements in original M + diagonal part.
//   // -> Full matrix M as input: estimate is half the number of total elements +
//   // diagonal = .5 * (nnzorigM + n)
//   // -> Upper triangular matrix M as input: estimate is the number of total
//   // elements + diagonal = nnzorigM + n
//   // The maximum between the two is nnzorigM + n
//   nnzmaxM = nnzorigM + n;

//   // OLD
//   // nnzmaxM = n*(n+1)/2;  // Full upper triangular matrix (This version
//   // allocates too much memory!)
//   // nnzmaxM = .5 * (nnzorigM + n);  // half of the total elements + diagonal

//   // Allocate M_trip
//   M_trip = csc_spalloc(n, n, nnzmaxM, 1, 1); // Triplet format

//   if (!M_trip) {
//     c_eprint("Upper triangular matrix extraction failed (out of memory)");
//     return OSQP_NULL;
//   }

//   // Fill M_trip with only elements in M which are in the upper triangular
//   for (j = 0; j < n; j++) { // Cycle over columns
//     for (ptr = M->p[j]; ptr < M->p[j + 1]; ptr++) {
//       // Get row index
//       i = M->i[ptr];

//       // Assign element only if in the upper triangular
//       if (i <= j) {
//         // c_print("\nM(%i, %i) = %.4f", M->i[ptr], j, M->x[ptr]);

//         M_trip->i[z_M] = i;
//         M_trip->p[z_M] = j;
//         M_trip->x[z_M] = M->x[ptr];

//         // Increase counter for the number of elements
//         z_M++;
//       }
//     }
//   }

//   // Set number of nonzeros
//   M_trip->nz = z_M;

//   // Convert triplet matrix to csc format
//   M_triu = triplet_to_csc(M_trip, OSQP_NULL);

//   // Assign number of nonzeros of full matrix to triu M
//   M_triu->nzmax = nnzmaxM;

//   // Cleanup and return result
//   csc_spfree(M_trip);

//   // Return matrix in triplet form
//   return M_triu;
// }

#ifndef OSQP_EMBEDDED_MODE

OSQPCscMatrix* triu_to_csc(OSQPCscMatrix* M) {
    OSQPCscMatrix* M_trip;    // Matrix in triplet format
    OSQPCscMatrix* M_symm;    // Resulting symmetric sparse matrix
    OSQPInt        n;         // Dimension of M
    OSQPInt        ptr, i, j; // Counters for (i,j) and index in M
    OSQPInt        z_M = 0;   // Counter for elements in M_trip

    if (M->m != M->n) {
        c_eprint("Matrix M not square");
        return OSQP_NULL;
    }
    n = M->n;

    // Estimate nzmax = twice the original (ignoring the double counted diagonal)
    M_trip = csc_spalloc(n, n, 2 * M->p[n], 1, 1);  // Triplet format
    if (!M_trip) {
        c_eprint("Matrix extraction failed (out of memory)");
        return OSQP_NULL;
    }

    for (j = 0; j < n; j++) {                           // Cycle over columns
        for (ptr = M->p[j]; ptr < M->p[j+1]; ptr++) {   // Index into i/x
            i = M->i[ptr];                              // Row index
            M_trip->i[z_M] = i;
            M_trip->p[z_M] = j;
            M_trip->x[z_M] = M->x[ptr];
            z_M++;

            // If we're above the diagonal, create another triplet entry with i,j reversed,
            // taking advantage of the fact that triplet entries can be in arbitrary order.
            if (i < j) {
                M_trip->i[z_M] = j;
                M_trip->p[z_M] = i;
                M_trip->x[z_M] = M->x[ptr];
                z_M++;
            }
        }
    }
    M_trip->nz = z_M;

    // Convert triplet matrix to csc format
    M_symm = triplet_to_csc(M_trip, OSQP_NULL);
    M_symm->nzmax = z_M;

    csc_spfree(M_trip);
    return M_symm;
}

OSQPCscMatrix* vstack(OSQPCscMatrix* A, OSQPCscMatrix* B) {
    OSQPCscMatrix* M_trip;    // Matrix in triplet format
    OSQPCscMatrix* M;         // Resulting csc matrix
    OSQPInt        m1, m2;    // No. of rows in A, B respectively
    OSQPInt        n;         // No. of columns in A/B
    OSQPInt        ptr, i, j; // Counters for (i,j) and index in M
    OSQPInt        z_M = 0;   // Counter for elements in M_trip

    if (A->n != B->n) {
        c_eprint("Matrix A and B do not have the same number of columns");
        return OSQP_NULL;
    }
    m1 = A->m;
    m2 = B->m;
    n = A->n;

    // Estimate nzmax = twice the original (ignoring the double counted diagonal)
    M_trip = csc_spalloc(m1 + m2, n, A->nzmax + B->nzmax, 1, 1);  // Triplet format
    if (!M_trip) {
        c_eprint("Matrix allocation failed (out of memory)");
        return OSQP_NULL;
    }

    for (j = 0; j < n; j++) {                           // Cycle over columns
        for (ptr = A->p[j]; ptr < A->p[j+1]; ptr++) {   // Index into i/x
            i = A->i[ptr];                              // Row index
            M_trip->i[z_M] = i;
            M_trip->p[z_M] = j;
            M_trip->x[z_M] = A->x[ptr];
            z_M++;
        }
    }
    for (j = 0; j < n; j++) {                           // Cycle over columns
        for (ptr = B->p[j]; ptr < B->p[j+1]; ptr++) {   // Index into i/x
            i = B->i[ptr] + m1;                         // Row index
            M_trip->i[z_M] = i;
            M_trip->p[z_M] = j;
            M_trip->x[z_M] = B->x[ptr];
            z_M++;
        }
    }
    M_trip->nz = z_M;

    // Convert triplet matrix to csc format
    M = triplet_to_csc(M_trip, OSQP_NULL);
    M->nzmax = z_M;

    csc_spfree(M_trip);
    return M;
}

#endif /* OSQP_EMBEDDED_MODE */
