#ifndef KKT_H
#define KKT_H


#include "osqp.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifndef OSQP_EMBEDDED_MODE

#include "csc_utils.h"

/**
 * Form square symmetric KKT matrix of the form
 *
 * [P + param1 I,            A';
 *  A             -diag(param2)]
 *
 * NB: Only the upper triangular part is filled
 *
 *
 * If rhotoKKT is not null it stores indices of param2 in the final KKT matrix
 *
 * @param  P          data for P in csc format (triu form)
 * @param  A          data for A in csc format
 * @param  format     CSC (0) or CSR (1)
 * @param  param1     regularization parameter
 * @param  param2     regularization parameter (vector)
 * @param  param2_sc  regularization parameter (scalar, used if param2 is NULL)
 * @param  PtoKKT     (modified) index mapping from elements of P to KKT matrix
 * @param  AtoKKT     (modified) index mapping from elements of A to KKT matrix
 * @param  param2toKKT(modified) index mapping from param2 to elements of
 *KKT
 * @return            return status flag
 */
 OSQPCscMatrix* form_KKT(OSQPCscMatrix* P,
                         OSQPCscMatrix* A,
                         OSQPInt        format,
                         OSQPFloat      param1,
                         OSQPFloat*     param2,
                         OSQPFloat      param2_sc,
                         OSQPInt*       PtoKKT,
                         OSQPInt*       AtoKKT,
                         OSQPInt*       param2toKKT);
# endif // ifndef OSQP_EMBEDDED_MODE


# if OSQP_EMBEDDED_MODE != 1

/**
 * Update KKT matrix using the elements of P
 *
 * @param KKT       KKT matrix in CSC form (upper-triangular)
 * @param P         P matrix in csc format (triu form)
 * @param P_new_idx indices of P to be updated
 * @param P_new_n   number of elements of P to be updated
 * @param PtoKKT    Vector of pointers from P->x to KKT->x
 * @param param1    Parameter added to the diagonal elements of P
 * @param format    0 for CSC, 1 for CSR
 */
 void update_KKT_P(OSQPCscMatrix* KKT,
                   OSQPCscMatrix* P,
                   const OSQPInt* Px_new_idx,
                   OSQPInt        P_new_n,
                   OSQPInt*       PtoKKT,
                   OSQPFloat      param1,
                   OSQPInt        format);


/**
 * Update KKT matrix using the elements of A
 *
 * @param KKT       KKT matrix in CSC form (upper-triangular)
 * @param  A        A matrix in csc format
 * @param A_new_idx indices of A to be updated
 * @param A_new_n   number of elements of A to be updated
 * @param AtoKKT    Vector of pointers from A->x to KKT->x
 */
 void update_KKT_A(OSQPCscMatrix* KKT,
                   OSQPCscMatrix* A,
                   const OSQPInt* Ax_new_idx,
                   OSQPInt        A_new_n,
                   OSQPInt*       AtoKKT);


/**
 * Update KKT matrix with new param2
 *
 * @param KKT           KKT matrix
 * @param param2        Parameter of the KKT matrix (vector)
 * @param param2_sc     Parameter of the KKT matrix (scalar, used if param2 == NULL)
 * @param param2toKKT   index where param2 enters in the KKT matrix
 * @param m             number of constraints
 */
void update_KKT_param2(OSQPCscMatrix* KKT,
                       OSQPFloat*     param2,
                       OSQPFloat      param2_sc,
                       OSQPInt*       param2toKKT,
                       OSQPInt        m);

# endif // OSQP_EMBEDDED_MODE != 1

#ifdef __cplusplus
}
#endif

#endif /* ifndef KKT_H */
