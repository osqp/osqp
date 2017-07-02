#ifndef KKT_H
#define KKT_H

#ifdef __cplusplus
extern "C" {
#endif

#include "types.h"

#ifndef EMBEDDED

#include "cs.h"

/**
 * Form square symmetric KKT matrix of the form
 *
 * [P + scalar1 I,         A';
 *  A             -scalar2 I]
 *
 * N.B. Only the upper triangular part is stuffed!
 *
 *
 *  If Pdiag_idx is not OSQP_NULL, it saves the index of the diagonal
 * elements of P there and the number of diagonal elements in Pdiag_n.
 *
 * Similarly, if rhotoKKT is not null,
 * it saves where the values of scalar2 go in the final KKT matrix
 *
 * N.B. Pdiag_idx needs to be freed!
 *
 * @param  P          cost matrix (already just upper triangular part)
 * @param  A          linear constraint matrix
 * @param  scalar1    regularization parameter scalar1
 * @param  scalar2    regularization parameter scalar2
 * @param  PtoKKT     (modified) index mapping from elements of P to KKT matrix
 * @param  AtoKKT     (modified) index mapping from elements of A to KKT matrix
 * @param  Pdiag_idx  (modified) Address of the index of diagonal elements in P
 * @param  Pdiag_n    (modified) Address to the number of diagonal elements in P
 * @param  scalar2toKKT    (modified) index mapping from scalar2 to elements of KKT
 * @return            return status flag
 */
csc * form_KKT(const csc * P, const  csc * A, c_float scalar1, c_float scalar2,
               c_int * PtoKKT, c_int * AtoKKT, c_int **Pdiag_idx, c_int *Pdiag_n, c_int * scalar2toKKT);
#endif // ifndef EMBEDDED


#if EMBEDDED != 1
/**
* Update KKT matrix using the elements of P
*
* @param KKT       KKT matrix in CSC form (upper-triangular)
* @param P         P matrix in CSC form (upper-triangular)
* @param PtoKKT    Vector of pointers from P->x to KKT->x
* @param scalar1   Scalar added to the diagonal elements
* @param Pdiag_idx Index of diagonal elements in P->x
* @param Pdiag_n   Number of diagonal elements of P
*/
void update_KKT_P(csc * KKT, const csc * P, const c_int * PtoKKT, const c_float scalar1, const c_int * Pdiag_idx, const c_int Pdiag_n);



/**
 * Update KKT matrix using the elements of A
 *
 * @param KKT       KKT matrix in CSC form (upper-triangular)
 * @param A         A matrix in CSC form (upper-triangular)
 * @param AtoKKT    Vector of pointers from A->x to KKT->x
 */
void update_KKT_A(csc * KKT, const csc * A, const c_int * AtoKKT);



/**
 * Update KKT matrix with new scalar2
 *
 * @param KKT           KKT matrix
 * @param scalar2       scalar2
 * @param scalar2toKKT  index where scalar2 enters in the KKT matrix
 * @param m             number of constraints
 */
void update_KKT_scalar2(csc * KKT, const c_float scalar2, const c_int * scalar2toKKT, const c_int m);

#endif // EMBEDDED != 1





#ifdef __cplusplus
}
#endif

#endif
