#ifndef ALGEBRA_MATRIX_H
#define ALGEBRA_MATRIX_H

#include "osqp_api_types.h"

#include "glob_opts.h"
#include "algebra_vector.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 *  OSQPMatrix types.  Not defined here since it
 *  is implementation-specific
 */

/* float valued matrices*/
typedef struct OSQPMatrix_ OSQPMatrix;


#ifndef OSQP_EMBEDDED_MODE

/*  logical functions ------------------------------------------------------*/

OSQPInt OSQPMatrix_is_eq(const OSQPMatrix* A,
                         const OSQPMatrix* B,
                               OSQPFloat   tol);

/*  Non-embeddable functions (using malloc) ----------------------------------*/


//Make a copy from a csc matrix.  Returns OSQP_NULL on failure
OSQPMatrix* OSQPMatrix_new_from_csc(const OSQPCscMatrix* A,
                                          OSQPInt        is_triu);

/* Return a copy of the matrix in CSC format */
OSQPCscMatrix* OSQPMatrix_get_csc(const OSQPMatrix* M);

/* Return a copy of a matrix as output (Uses MALLOC) */
OSQPMatrix* OSQPMatrix_copy_new(const OSQPMatrix* A);

// Convert an upper triangular matrix into a fully populated matrix
OSQPMatrix* OSQPMatrix_triu_to_symm(const OSQPMatrix* A);

// Vertically stack two matrices
OSQPMatrix* OSQPMatrix_vstack(const OSQPMatrix* A, const OSQPMatrix* B);

#endif //OSQP_EMBEDDED_MODE


/*  direct data access functions ---------------------------------------------*/

/*  These functions allow getting data in csc format from
*   the OSQPMatrix type.  Data is passed in/out using bare
*   pointers instead of OSQPVectors since these functions interface
*   with user defined linear solvers and the user API
*/

void OSQPMatrix_update_values(OSQPMatrix*      M,
                              const OSQPFloat* Mx_new,
                              const OSQPInt*   Mx_new_idx,
                              OSQPInt          M_new_n);

/* returns the row dimension */
OSQPInt    OSQPMatrix_get_m(const OSQPMatrix* M);

/* returns the columns dimension */
OSQPInt    OSQPMatrix_get_n(const OSQPMatrix* M);

/* returns a pointer to the array of data values */
OSQPFloat* OSQPMatrix_get_x(const OSQPMatrix* M);

/* returns a pointer to the array of row indices */
OSQPInt*   OSQPMatrix_get_i(const OSQPMatrix* M);

/* returns a pointer to the array of col indices (csc format).  Should be n+1 long */
OSQPInt*   OSQPMatrix_get_p(const OSQPMatrix* M);

/* returns the number of nonzeros (length of x and i arrays) */
OSQPInt    OSQPMatrix_get_nz(const OSQPMatrix* M);

/* math functions ----------------------------------------------------------*/

//A = sc*A
void OSQPMatrix_mult_scalar(OSQPMatrix* A,
                            OSQPFloat     sc);

//A = L*A, with diagonal entries of L specified
void OSQPMatrix_lmult_diag(OSQPMatrix*        A,
                           const OSQPVectorf* L);

//A = A*R, with diagonal entries of R specified
void OSQPMatrix_rmult_diag(OSQPMatrix*        A,
                           const OSQPVectorf* R);

// d = diag(At*diag(D)*A)
void OSQPMatrix_AtDA_extract_diag(const OSQPMatrix*  A,
                                  const OSQPVectorf* D,
                                        OSQPVectorf* d);

// Extract the main diagonal of A into d
void OSQPMatrix_extract_diag(const OSQPMatrix*  A,
                                   OSQPVectorf* d);

//y = alpha*A*x + beta*y
void OSQPMatrix_Axpy(const OSQPMatrix*  A,
                     const OSQPVectorf* x,
                     OSQPVectorf*       y,
                     OSQPFloat          alpha,
                     OSQPFloat          beta);

//y = alpha*A^T*x + beta*y
void OSQPMatrix_Atxpy(const OSQPMatrix*  A,
                      const OSQPVectorf* x,
                      OSQPVectorf*       y,
                      OSQPFloat          alpha,
                      OSQPFloat          beta);

// OSQPFloat OSQPMatrix_quad_form(const OSQPMatrix  *P,
//                              const OSQPVectorf *x);

#if OSQP_EMBEDDED_MODE != 1

void OSQPMatrix_col_norm_inf(const OSQPMatrix*  M,
                                   OSQPVectorf* E);

void OSQPMatrix_row_norm_inf(const OSQPMatrix*  M,
                                   OSQPVectorf* E);

#endif /* if OSQP_EMBEDDED_MODE != 1 */

#ifndef OSQP_EMBEDDED_MODE

void OSQPMatrix_free(OSQPMatrix* M);

OSQPMatrix* OSQPMatrix_submatrix_byrows(const OSQPMatrix*  A,
                                        const OSQPVectori* rows);

#endif /* ifndef OSQP_EMBEDDED_MODE */

#ifdef __cplusplus
}
#endif

#endif /* ifndef ALGEBRA_MATRIX_H */
