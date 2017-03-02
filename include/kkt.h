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


#ifdef __cplusplus
}
#endif

#endif
