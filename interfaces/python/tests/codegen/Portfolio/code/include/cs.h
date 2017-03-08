/* NB: this is a subset of the routines in the CSPARSE package by
   Tim Davis et. al., for the full package please visit
   http://www.cise.ufl.edu/research/sparse/CSparse/ */

#ifndef CS_H
#define CS_H

#ifdef __cplusplus
extern "C" {
#endif

#include "types.h"    // CSC matrix type
#include "lin_alg.h"  // Vector copy operations

/*****************************************************************************
 * Create and free CSC Matrices                                              *
 *****************************************************************************/

/* Create Compressed-Column-Sparse matrix from existing arrays
   (no MALLOC to create inner arrays x, i, p)
 */
csc* csc_matrix(c_int m, c_int n, c_int nzmax, c_float* x, c_int* i, c_int* p);


/* Create uninitialized CSC matrix atricture
   (uses MALLOC to create inner arrays x, i, p)
   Arguments
   ---------
   m,n: dimensions
   nzmax: max number of nonzero elements
   values: 1/0 allocate values
   triplet: 1/0 allocate matrix for CSC or Triplet format
 */
csc *csc_spalloc(c_int m, c_int n, c_int nzmax, c_int values, c_int triplet);


/* Free sparse matrix
   (uses FREE to free inner arrays x, i, p)
 */
csc *csc_spfree(csc *A);


/* free workspace and return a sparse matrix result */
csc * csc_done(csc *C, void *w, void *x, c_int ok);

/*****************************************************************************
 * Copy Matrices                                                             *
 *****************************************************************************/
 /**
  *  Copy sparse CSC matrix A to output.
  *  output is allocated by this function (uses MALLOC)
  */
 csc * copy_csc_mat(const csc* A);


 /**
  *  Copy sparse CSC matrix A to B (B is preallocated, NO MALOC)
  */
 void prea_copy_csc_mat(const csc* A, csc* B);


/*****************************************************************************
 * Matrices Conversion                                                       *
 *****************************************************************************/


 /**
  * C = compressed-column CSC from matrix T in triplet form
  *
  * TtoC stores the vector of indeces from T to C
  *  -> C[TtoC[i]] = T[i]
  *
  * @param  T    matrix in triplet format
  * @param  TtoC vector of indeces from triplet to CSC format
  * @return      matrix in CSC format
  */
 csc *triplet_to_csc(const csc *T, c_int * TtoC);

/* Convert sparse to dense */
c_float * csc_to_dns(csc * M);


/**
 * Convert square CSC matrix into upper triangular one
 *
 * @param  M         Matrix to be converted
 * @return           Upper triangular matrix in CSC format
 */
csc * csc_to_triu(csc * M);



/*****************************************************************************
 * Extra operations                                                          *
 *****************************************************************************/

/* p [0..n] = cumulative sum of c [0..n-1], and then copy p [0..n-1] into c */
c_int csc_cumsum(c_int *p, c_int *c, c_int n);

/* Compute inverse of permuation matrix stored in the vector p.
 * The computed inverse is also stored in a vector.
 */
c_int *csc_pinv(c_int const *p, c_int n);

/**
 * C = A(p,p)= PAP' where A and C are symmetric the upper part stored;
 *  N.B. pinv not p!
 * @param  A      Original matrix (upper-triangular)
 * @param  pinv   Inverse of permutation vector
 * @param  AtoC   Mapping from indeces of A-x to C->x
 * @param  values Are values of A allocated?
 * @return        New matrix (allocated)
 */
csc *csc_symperm(const csc *A, const c_int *pinv, c_int * AtoC, c_int values);



#ifdef __cplusplus
}
#endif

#endif
