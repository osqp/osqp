#ifndef CSC_UTILS_H
#define CSC_UTILS_H

#include "osqp_configure.h"
#include "osqp_api_types.h"
#include "glob_opts.h"

#ifdef __cplusplus
extern "C" {
#endif

// ========== Logical, testing and debug ===========

OSQPInt csc_is_eq(OSQPCscMatrix* A, OSQPCscMatrix* B, OSQPFloat tol);

/*****************************************************************************
* Create and free CSC Matrices                                              *
*****************************************************************************/

#ifndef OSQP_EMBEDDED_MODE

/**
 * Create uninitialized CSC matrix structure
    (uses MALLOC to create inner arrays x, i, p)
 * @param  m       First dimension
 * @param  n       Second dimension
 * @param  nzmax   Maximum number of nonzero elements
 * @param  values  Allocate values (0/1)
 * @param  triplet Allocate CSC or triplet format matrix (1/0)
 * @return         Matrix pointer
 */
OSQPCscMatrix* csc_spalloc(OSQPInt m,
                           OSQPInt n,
                           OSQPInt nzmax,
                           OSQPInt values,
                           OSQPInt triplet);


/**
 * Free sparse matrix
    (uses FREE to free inner arrays x, i, p)
 * @param  A Matrix in CSC format
 */
void csc_spfree(OSQPCscMatrix* A);

/**
 * Create a new matrix from a subset of the rows of A
    (uses MALLOC to create the new matrix)
    * @param  A      CSC matrix
    * @param  rows   vector indicating which rows to select (all nonzeros are selected)
                     this should be the same length as A->m
    * @return    Returns A(rows,:) if successful, null otherwise
 */
OSQPCscMatrix* csc_submatrix_byrows(const OSQPCscMatrix* A,
                                    OSQPInt*             rows);


/**
 * free workspace and return a sparse matrix result
 * @param  C  CSC matrix
 * @param  w  Workspace vector
 * @param  x  Workspace vector
 * @param  ok flag
 * @return    Return result C if OK, otherwise free it
 */
OSQPCscMatrix* csc_done(OSQPCscMatrix* C,
                        void*          w,
                        void*          x,
                        OSQPInt        ok);
#endif /* OSQP_EMBEDDED_MODE */

/*****************************************************************************
* Copy Matrices                                                             *
*****************************************************************************/

#ifndef OSQP_EMBEDDED_MODE

/**
 *  Copy sparse CSC matrix A to output.
 *  output is allocated by this function (uses MALLOC)
 */
OSQPCscMatrix* csc_copy(const OSQPCscMatrix* A);

// /**
//  *  Copy sparse CSC matrix A to B (B is preallocated, NO MALLOC)
//  */
// void csc_copy_prea(const OSQPCscMatrix* A,
//                          OSQPCscMatrix* B);

/* Convert sparse CSC to dense (uses MALLOC)*/
OSQPFloat* csc_to_dns(OSQPCscMatrix* M);

#endif /* OSQP_EMBEDDED_MODE */

/*****************************************************************************
* Matrices Conversion                                                       *
*****************************************************************************/

#ifndef OSQP_EMBEDDED_MODE

/**
 * C = compressed-column CSC from matrix T in triplet form
 *
 * TtoC stores the vector of indices from T to C
 *  -> C[TtoC[i]] = T[i]
 *
 * @param  T    matrix in triplet format
 * @param  TtoC vector of indices from triplet to CSC format
 * @return      matrix in CSC format
 */
OSQPCscMatrix* triplet_to_csc(const OSQPCscMatrix* T,
                                    OSQPInt*       TtoC);


/**
 * C = compressed-row CSR from matrix T in triplet form
 *
 * TtoC stores the vector of indices from T to C
 *  -> C[TtoC[i]] = T[i]
 *
 * @param  T    matrix in triplet format
 * @param  TtoC vector of indices from triplet to CSR format
 * @return      matrix in CSR format
 */
OSQPCscMatrix* triplet_to_csr(const OSQPCscMatrix* T,
                                    OSQPInt*       TtoC);


// /**
//  * Convert square CSC matrix into upper triangular one
//  *
//  * @param  M         Matrix to be converted
//  * @return           Upper triangular matrix in CSC format
//  */
// OSQPCscMatrix* csc_to_triu(OSQPCscMatrix* M);

/**
 * Convert upper triangular square CSC matrix into symmetric by copying
 * data above diagonal.
 *
 * @param  M         Matrix to be converted
 * @return           Symmetric matrix in CSC format
 */
OSQPCscMatrix* triu_to_csc(OSQPCscMatrix* M);


/**
 * Vertically stack two csc matrices
 *
 * @param  A         First CSC matrix
 * @param  B         Second CSC matrix
 * @return           CSC matrix resulting from vstacking A and B
 */
OSQPCscMatrix* vstack(OSQPCscMatrix* A,
                      OSQPCscMatrix* B);

#endif /* OSQP_EMBEDDED_MODE */

/*****************************************************************************
* Extra operations                                                          *
*****************************************************************************/

// /**
//  * p [0..n] = cumulative sum of c [0..n-1], and then copy p [0..n-1] into c
//  *
//  * @param  p Create cumulative sum into p
//  * @param  c Vector of which we compute cumulative sum
//  * @param  n Number of elements
//  * @return   Exitflag
//  */
// OSQPInt csc_cumsum(OSQPInt *p,
//                  OSQPInt *c,
//                  OSQPInt  n);

/**
 * Extract main diagonal from matrix A into vector d.
 */
void csc_extract_diag(const OSQPCscMatrix* A,
                            OSQPFloat*     d);

#ifndef OSQP_EMBEDDED_MODE

/**
 * Compute inverse of permutation matrix stored in the vector p.
 * The computed inverse is also stored in a vector.
 * Allocates return vector (uses MALLOC)
 */
OSQPInt* csc_pinv(const OSQPInt* p,
                        OSQPInt  n);

/**
 * C = A(p,p)= PAP' where A and C are symmetric the upper part stored;
 *  NB: pinv not p!
 * @param  A      Original matrix (upper-triangular)
 * @param  pinv   Inverse of permutation vector
 * @param  AtoC   Mapping from indices of A-x to C->x
 * @param  values Are values of A allocated?
 * @return        New matrix (allocated)
 */
OSQPCscMatrix* csc_symperm(const OSQPCscMatrix* A,
                           const OSQPInt*       pinv,
                                 OSQPInt*       AtoC,
                                 OSQPInt        values);

#endif /* OSQP_EMBEDDED_MODE */

#ifdef __cplusplus
}
#endif

#endif /* ifndef CSC_UTILS_H */
