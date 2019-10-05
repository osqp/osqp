#ifndef KKT_H
#define KKT_H

# ifdef __cplusplus
extern "C" {
# endif // ifdef __cplusplus

# include "osqp.h"

# ifndef EMBEDDED

#  include "csc_utils.h"

/**
 * Form square symmetric KKT matrix of the form
 *
 * [P + param1 I,            A';
 *  A             -diag(param2)]
 *
 * NB: Only the upper triangular part is filled
 *
 *
 *  If Pdiag_idx is not OSQP_NULL, it saves the index of the diagonal
 * elements of P there and the number of diagonal elements in Pdiag_n.
 *
 * Similarly, if rhotoKKT is not null,
 * it saves where the values of param2 go in the final KKT matrix
 *
 * NB: Pdiag_idx needs to be freed!
 *
 * @param  P_x        value data for P in csc format (triu form)
 * @param  P_i        row data for P in csc format (triu form)
 * @param  P_p        column data for P in csc format (triu form)
 * @param  A_x        value data for A in csc format
 * @param  A_i        row data for A in csc format
 * @param  A_p        column data for A in csc format
 * @param  m          number of constraints (A is m x n)
 * @param  n          number of primal variables (P is n x n)
 * @param  format     CSC (0) or CSR (1)
 * @param  param1     regularization parameter
 * @param  param2     regularization parameter (vector)
 * @param  PtoKKT     (modified) index mapping from elements of P to KKT matrix
 * @param  AtoKKT     (modified) index mapping from elements of A to KKT matrix
 * @param  Pdiag_idx  (modified) Address of the index of diagonal elements in P
 * @param  Pdiag_n    (modified) Address to the number of diagonal elements in P
 * @param  param2toKKT    (modified) index mapping from param2 to elements of
 *KKT
 * @return            return status flag
 */
 csc* form_KKT(c_float*    P_x,
               c_int*      P_i,
               c_int*      P_p,
               c_float*    A_x,
               c_int*      A_i,
               c_int*      A_p,
               c_int       m,
               c_int       n,
               c_int       format,
               c_float     param1,
               c_float    *param2,
               c_int      *PtoKKT,
               c_int      *AtoKKT,
               c_int     **Pdiag_idx,
               c_int      *Pdiag_n,
               c_int      *param2toKKT);
# endif // ifndef EMBEDDED


# if EMBEDDED != 1

/**
 * Update KKT matrix using the elements of P
 *
 * @param KKT       KKT matrix in CSC form (upper-triangular)
 * @param  P_x      value data for P in csc format (triu form)
 * @param  P_p      column data for P in csc format (triu form)
 * @param  n        number of primal variables (P is n x n)
 * @param PtoKKT    Vector of pointers from P->x to KKT->x
 * @param param1    Parameter added to the diagonal elements of P
 * @param Pdiag_idx Index of diagonal elements in P->x
 * @param Pdiag_n   Number of diagonal elements of P
 */
 void update_KKT_P(csc         *KKT,
                   c_float*    P_x,
                   c_int*      P_p,
                   c_int       n,
                   c_int  *PtoKKT,
                   c_float param1,
                   c_int  *Pdiag_idx,
                   c_int   Pdiag_n);


/**
 * Update KKT matrix using the elements of A
 *
 * @param KKT       KKT matrix in CSC form (upper-triangular)
 * @param  A_x      value data for A in csc format
 * @param  A_p      column data for A in csc format
 * @param  n        number of primal variables (P is n x n)
 * @param AtoKKT    Vector of pointers from A->x to KKT->x
 */
 void update_KKT_A(csc *KKT,
                   c_float*    A_x,
                   c_int*      A_p,
                   c_int       n,
                   c_int*      AtoKKT);


/**
 * Update KKT matrix with new param2
 *
 * @param KKT           KKT matrix
 * @param param2        Parameter of the KKT matrix (vector)
 * @param param2toKKT   index where param2 enters in the KKT matrix
 * @param m             number of constraints
 */
void update_KKT_param2(csc           *KKT,
                       c_float *param2,
                       c_int   *param2toKKT,
                       c_int    m);

# endif // EMBEDDED != 1


# ifdef __cplusplus
}
# endif // ifdef __cplusplus

#endif // ifndef KKT_H
