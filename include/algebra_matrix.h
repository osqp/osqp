#ifndef ALGEBRA_MATRIX_H
# define ALGEBRA_MATRIX_H


# ifdef __cplusplus
extern "C" {
# endif // ifdef __cplusplus

#include "glob_opts.h"

/*
 * OSQPMatrix types.  Not defined here since it
 *   is implementation specific
 */

/* float valued matrices*/
typedef struct OSQPMatrix_ OSQPMatrix;


/* MATRIX FUNCTIONS ----------------------------------------------------------*/

//DEBUG: OSQP Wrappers.   Provide documentation of functions

//A = L*A, with diagonal entries of L specified
void OSQPMatrix_lmult_diag(OSQPMatrix *A, const OSQPVectorf *L);

//A = A*R, with diagonal entries of R specified
void OSQPMatrix_rmult_diag(OSQPMatrix *A, const OSQPVectorf *R);

//y = A*x + beta*y
void OSQPMatrix_Axpy(const OSQPMatrix *A,
                      const OSQPVectorf *x,
                      OSQPVectorf *y,
                      c_float alpha,
                      c_float beta);

//y = A^T*x + beta*y
void OSQPMatrix_Atxpy(const OSQPMatrix *A,
                      const OSQPVectorf *x,
                      OSQPVectorf *y,
                      c_float alpha,
                      c_float beta);

c_float OSQPMatrix_quad_form(const OSQPMatrix *P, const OSQPVectorf *x);

#if EMBEDDED != 1

void OSQPMatrix_col_norm_inf(const OSQPMatrix *M, OSQPVectorf *E);

void OSQPMatrix_row_norm_inf(const OSQPMatrix *M, OSQPVectorf *E);

#endif /* if EMBEDDED != 1 */

#ifndef EMBEDDED

OSQPMatrix* OSQPMatrix_symperm(const OSQPMatrix *A, const OSQPVectori *pinv, OSQPVectori *AtoC, c_int values);

#endif // ndef EMBEDDED



# ifdef __cplusplus
}
# endif // ifdef __cplusplus

#endif // ifndef ALGEBRA_MATRIX_H
