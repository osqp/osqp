#ifndef ALGEBRA_MATRIX_H
# define ALGEBRA_MATRIX_H


# ifdef __cplusplus
extern "C" {
# endif // ifdef __cplusplus

# include "types.h"


/* MATRIX FUNCTIONS ----------------------------------------------------------*/

/* multiply scalar to matrix */
void mat_mult_scalar(csc    *A,
                     c_float sc);

/* Premultiply matrix A by diagonal matrix with diagonal d,
   i.e. scale the rows of A by d
 */
void mat_premult_diag(csc           *A,
                      const c_float *d);

/* Premultiply matrix A by diagonal matrix with diagonal d,
   i.e. scale the columns of A by d
 */
void mat_postmult_diag(csc           *A,
                       const c_float *d);


/* Matrix-vector multiplication
 *    y  =  A*x  (if plus_eq == 0)
 *    y +=  A*x  (if plus_eq == 1)
 *    y -=  A*x  (if plus_eq == -1)
 */
void mat_vec(const csc     *A,
             const c_float *x,
             c_float       *y,
             c_int          plus_eq);


/* Matrix-transpose-vector multiplication
 *    y  =  A'*x  (if plus_eq == 0)
 *    y +=  A'*x  (if plus_eq == 1)
 *    y -=  A'*x  (if plus_eq == -1)
 * If skip_diag == 1, then diagonal elements of A are assumed to be zero.
 */
void mat_tpose_vec(const csc     *A,
                   const c_float *x,
                   c_float       *y,
                   c_int          plus_eq,
                   c_int          skip_diag);


# if EMBEDDED != 1

/**
 * Infinity norm of each matrix column
 * @param M	Input matrix
 * @param E     Vector of infinity norms
 *
 */
void mat_inf_norm_cols(const csc *M,
                       c_float   *E);

/**
 * Infinity norm of each matrix row
 * @param M	Input matrix
 * @param E     Vector of infinity norms
 *
 */
void mat_inf_norm_rows(const csc *M,
                       c_float   *E);

/**
 * Infinity norm of each matrix column
 * Matrix M is symmetric upper-triangular
 *
 * @param M	Input matrix (symmetric, upper-triangular)
 * @param E     Vector of infinity norms
 *
 */
void mat_inf_norm_cols_sym_triu(const csc *M,
                                c_float   *E);

# endif // EMBEDDED != 1

/**
 * Compute quadratic form f(x) = 1/2 x' P x
 * @param  P quadratic matrix in CSC form (only upper triangular)
 * @param  x argument float vector
 * @return   quadratic form value
 */
c_float quad_form(const csc     *P,
                  const c_float *x);


# ifdef __cplusplus
}
# endif // ifdef __cplusplus

#endif // ifndef ALGEBRA_MATRIX_H
