#ifndef ALGEBRA_MATRIX_H
#define ALGEBRA_MATRIX_H


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


/*  Non-embeddable functions (using malloc) ----------------------------------*/

#ifndef EMBEDDED

//Make a copy from a csc matrix.  Returns OSQP_NULL on failure
OSQPMatrix* OSQPMatrix_new_from_csc(const csc* A, c_int is_triu);

#endif //EMBEDDED


/*  direct data access functions ---------------------------------------------*/

/*  These functions allow getting/setting data
*   in the OSQPMatrix type.   Data is passed in/out using bare
*   pointers instead of OSQPVectors since these functions interface
*   with user defined linear solvers and the user API
*/

void OSQPMatrix_update_values(OSQPMatrix    *M,
                            const c_float   *Mx_new,
                            const c_int     *Mx_new_idx,
                            c_int           M_new_n);

c_int    OSQPMatrix_get_m(const OSQPMatrix *M);
c_int    OSQPMatrix_get_n(const OSQPMatrix *M);
c_float* OSQPMatrix_get_x(const OSQPMatrix *M);
c_int*   OSQPMatrix_get_i(const OSQPMatrix *M);
c_int*   OSQPMatrix_get_p(const OSQPMatrix *M);
c_int    OSQPMatrix_get_nnz(const OSQPMatrix *M);


/* math functions ----------------------------------------------------------*/

//DEBUG: OSQP Wrappers.   Provide documentation of functions

//A = sc*A
void OSQPMatrix_mult_scalar(OSQPMatrix *A, c_float sc);

//A = L*A, with diagonal entries of L specified
void OSQPMatrix_lmult_diag(OSQPMatrix *A, const OSQPVectorf *L);

//A = A*R, with diagonal entries of R specified
void OSQPMatrix_rmult_diag(OSQPMatrix *A, const OSQPVectorf *R);

//y = alpha*A*x + beta*y
void OSQPMatrix_Axpy( const OSQPMatrix *A,
                      const OSQPVectorf *x,
                      OSQPVectorf *y,
                      c_float alpha,
                      c_float beta);

//y = alpha*A^T*x + beta*y
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

void OSQPMatrix_free(OSQPMatrix *M);

OSQPMatrix* OSQPMatrix_submatrix_byrows(const OSQPMatrix* A, const OSQPVectori* rows, c_int nrows);

#endif // ndef EMBEDDED



# ifdef __cplusplus
}
# endif // ifdef __cplusplus

#endif // ifndef ALGEBRA_MATRIX_H
