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


#ifndef EMBEDDED

/*  logical functions ------------------------------------------------------*/

c_int OSQPMatrix_is_eq(const OSQPMatrix *A,
                       const OSQPMatrix *B,
                       c_float           tol);

/*  Non-embeddable functions (using malloc) ----------------------------------*/


//Make a copy from a csc matrix.  Returns OSQP_NULL on failure
OSQPMatrix* OSQPMatrix_new_from_csc(const csc *A,
                                    c_int      is_triu);

#endif //EMBEDDED


/*  direct data access functions ---------------------------------------------*/

/*  These functions allow getting data in csc format from
*   the OSQPMatrix type.   Data is passed in/out using bare
*   pointers instead of OSQPVectors since these functions interface
*   with user defined linear solvers and the user API
*/

void OSQPMatrix_update_values(OSQPMatrix    *M,
                              const c_float *Mx_new,
                              const c_int   *Mx_new_idx,
                              c_int          M_new_n);

/* returns the row dimension */
c_int    OSQPMatrix_get_m(const OSQPMatrix *M);

/* returns the columns dimension */
c_int    OSQPMatrix_get_n(const OSQPMatrix *M);

/* returns a pointer to the array of data values */
c_float* OSQPMatrix_get_x(const OSQPMatrix *M);

/* returns a pointer to the array of row indices */
c_int*   OSQPMatrix_get_i(const OSQPMatrix *M);

/* returns a pointer to the array of col indices (csc format).  Should be n+1 long */
c_int*   OSQPMatrix_get_p(const OSQPMatrix *M);

/* returns the number of nonzeros (length of x and i arrays) */
c_int    OSQPMatrix_get_nz(const OSQPMatrix *M);


/* math functions ----------------------------------------------------------*/

//A = sc*A
void OSQPMatrix_mult_scalar(OSQPMatrix *A,
                            c_float     sc);

//A = L*A, with diagonal entries of L specified
void OSQPMatrix_lmult_diag(OSQPMatrix        *A,
                           const OSQPVectorf *L);

//A = A*R, with diagonal entries of R specified
void OSQPMatrix_rmult_diag(OSQPMatrix        *A,
                           const OSQPVectorf *R);

//y = alpha*A*x + beta*y
void OSQPMatrix_Axpy(const OSQPMatrix  *A,
                     const OSQPVectorf *x,
                     OSQPVectorf       *y,
                     c_float            alpha,
                     c_float            beta);

//y = alpha*A^T*x + beta*y
void OSQPMatrix_Atxpy(const OSQPMatrix  *A,
                      const OSQPVectorf *x,
                      OSQPVectorf       *y,
                      c_float            alpha,
                      c_float            beta);

c_float OSQPMatrix_quad_form(const OSQPMatrix  *P,
                             const OSQPVectorf *x);

#if EMBEDDED != 1

void OSQPMatrix_col_norm_inf(const OSQPMatrix *M,
                             OSQPVectorf      *E);

void OSQPMatrix_row_norm_inf(const OSQPMatrix *M,
                             OSQPVectorf      *E);

#endif /* if EMBEDDED != 1 */

#ifndef EMBEDDED

void OSQPMatrix_free(OSQPMatrix *M);

OSQPMatrix* OSQPMatrix_submatrix_byrows(const OSQPMatrix  *A,
                                        const OSQPVectori *rows);

#endif // ndef EMBEDDED



# ifdef __cplusplus
}
# endif // ifdef __cplusplus

#endif // ifndef ALGEBRA_MATRIX_H
