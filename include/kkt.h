#ifndef KKT_H
#define KKT_H

#ifdef __cplusplus
extern "C" {
#endif

#include "types.h"

/**
 * Form square symmetric KKT matrix of the form
 *
 * [P + scalar1 I,         A';
 *  A             -scalar2 I]
 *
 * N.B. Only the upper triangular part is stuffed!
 *
 * @param  P          cost matrix (already just upper triangular part)
 * @param  A          linear constraint matrix
 * @param  scalar1    regularization parameter scalar1
 * @param  scalar2    regularization parameter scalar2
 * @param  PtoKKT     (modified) index mapping from elements of P to KKT matrix
 * @param  AtoKKT     (modified) index mapping from elements of A to KKT matrix
 * @return            return status flag
 */
csc * form_KKT(const csc * P, const  csc * A, c_float scalar1, c_float scalar2,
               c_int * PtoKKT, c_int * AtoKKT);



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
void update_KKT_P(csc * KKT, csc * P, c_int * PtoKKT, c_float scalar1, c_int * Pdiag_idx, c_int Pdiag_n);



/**
 * Update KKT matrix using the elements of A
 *
 * @param KKT       KKT matrix in CSC form (upper-triangular)
 * @param A         A matrix in CSC form (upper-triangular)
 * @param AtoKKT    Vector of pointers from A->x to KKT->x
 */
void update_KKT_A(csc * KKT, csc * A, c_int * AtoKKT);



#ifdef __cplusplus
}
#endif

#endif
