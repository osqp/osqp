#ifndef LIN_ALG_H
# define LIN_ALG_H


# ifdef __cplusplus
extern "C" {
# endif // ifdef __cplusplus

# include "types.h"


/* VECTOR FUNCTIONS ----------------------------------------------------------*/

# ifndef EMBEDDED

/* Return a copy of a float vector a as output (Uses MALLOC)*/
OsqpVectorf* OsqpVectorf_copy_new(OsqpVectorf *a);

/* Return a copy of an int vector a as output (Uses MALLOC)*/
OsqpVectori* OsqpVectori_copy_new(OsqpVectori *a);

# endif // ifndef EMBEDDED

/* copy a float vector a into another vector b (pre-allocated) */
void OsqpVectorf_copy(OsqpVectorf *a,OsqpVectorf *b);

/* copy an int vector a into another vector b (pre-allocated) */
void OsqpVectori_copy(OsqpVectori *a,OsqpVectori *b);

/* set float vector to scalar */
void OsqpVectorf_set_scalar(OsqpVectorf *a, c_float sc);

/* set int vector to scalar */
void OsqpVectori_set_scalar(OsqpVectori *a, c_int sc);

/* add scalar to float vector */
void OsqpVectorf_add_scalar(OsqpVectorf *a, c_float sc);

/* add scalar to int vector */
void OsqpVectori_add_scalar(OsqpVectori *a, c_int sc);

/* multiply float vector by float */
void OsqpVectorf_mult_scalar(OsqpVectorf *a, c_float sc);


/* c = a + sc*b */
void OsqpVectorf_add_scaled(OsqpVectorf       *c,
                             const OsqpVectorf *a,
                             const OsqpVectorf *b,
                             c_float           sc);


/* ||v||_inf */
c_float OsqpVectorf_norm_inf(const OsqpVectorf *v);

/* ||v||_1 */
c_float OsqpVectorf_norm_1(const OsqpVectorf *v);

/* ||Sv||_inf */
c_float OsqpVectorf_scaled_norm_inf(const OsqpVectorf *S, const OsqpVectorf *v);

/* ||Sv||_1 */
c_float OsqpVectorf_scaled_norm_1(const OsqpVectorf *S, const OsqpVectorf *v);

/* ||a - b||_inf */
c_float OsqpVectorf_norm_inf_diff(const OsqpVectorf *a,
                                  const OsqpVectorf *b);


/* sum of vector elements */
c_float OsqpVectorf_sum(const OsqpVectorf *a);

/* mean of vector elements */
c_float OsqpVectorf_mean(const OsqpVectorf *a);



/* Inner product a'b */
c_float OsqpVectorf_dot_prod(const OsqpVectorf *a,
                             const OsqpVectorf *b);

/* elementwse product a.*b stored in c*/
void OsqpVectorf_ew_prod(const OsqpVectorf *a,
                         const OsqpVectorf *b,
                         OsqpVectorf       *c);


# if EMBEDDED != 1

/* Vector elementwise reciprocal b = 1./a (needed for scaling)*/
void OsqpVectorf_ew_reciprocal(const OsqpVectorf *a, OsqpVectorf *b);

/* elementwise sqrt of the vector elements */
void OsqpVectorf_ew_sqrt(OsqpVectorf *a);

/* elementwise max between each vector component and max_val */
void OsqpVectorf_ew_max(OsqpVectorf *a, c_float max_val);

/* elementwise max between each vector component and max_val */
void OsqpVectorf_ew_min(OsqpVectorf *a, c_float min_val);

/* Elementwise maximum between vectors c = max(a, b) */
void OsqpVectorf_ew_max_vec(const OsqpVectorf *a,
                            const OsqpVectorf *b,
                            OsqpVectorf       *c);

/* Elementwise minimum between vectors c = min(a, b) */
void OsqpVectorf_ew_min_vec(const OsqpVectorf *a,
                            const OsqpVectorf *b,
                            OsqpVectorf       *c);

# endif // if EMBEDDED != 1


/* MATRIX FUNCTIONS ----------------------------------------------------------*/

/* multiply matrix by scalar */
void OSQPMatrix_mult_scalar(OSQPMatrix *A, c_float sc);

/* Premultiply (i.e. left) matrix A by diagonal matrix with diagonal d,
   i.e. scale the rows of A by d
 */
void OSQPMatrix_premult_diag(OSQPMatrix *A, const c_float *d);

/* Postmultiply (i.e. right) matrix A by diagonal matrix with diagonal d,
   i.e. scale the columns of A by d
 */
void OSQPMatrix_postmult_diag(OSQPMatrix *A, const c_float *d);


/* Matrix-vector multiplication
 *    y  =  A*x  (if plus_eq == 0)
 *    y +=  A*x  (if plus_eq == 1)
 *    y -=  A*x  (if plus_eq == -1)
 */
void OSQPMatrix_Ax(const OSQPMatrix  *A,
                   const OSQPVectorf *x,
                   OSQPVectorf       *y,
                   c_int             sign);


/* Matrix-transpose-vector multiplication
 *    y  =  A'*x  (if plus_eq == 0)
 *    y +=  A'*x  (if plus_eq == 1)
 *    y -=  A'*x  (if plus_eq == -1)
 * If skip_diag == 1, then diagonal elements of A are assumed to be zero.
 */
 void OSQPMatrix_Atx(const OSQPMatrix  *A,
                     const OSQPVectorf *x,
                     OSQPVectorf       *y,
                     c_int             sign
                     c_int             skip_diag);


# if EMBEDDED != 1

/**
 * Infinity norm of each matrix column
 * @param M	Input matrix
 * @param E     Vector of infinity norms
 *
 */
void OSQPMatrix_inf_norm_cols(const OSQPMatrix *M,
                              OSQPVectorf      *E);

/**
 * Infinity norm of each matrix row
 * @param M	Input matrix
 * @param E     Vector of infinity norms
 *
 */
void OSQPMatrix_inf_norm_rows(const OSQPMatrix *M,
                              OSQPVectorf      *E);

/**
 * Infinity norm of each matrix column
 * Matrix M is symmetric upper-triangular
 *
 * @param M	Input matrix (symmetric, upper-triangular)
 * @param E     Vector of infinity norms
 *
 */
void OSQPMatrix_inf_norm_cols_sym_triu(const OSQPMatrix *M,
                                       OSQPVectorf      *E);

# endif // EMBEDDED != 1

/**
 * Compute quadratic form f(x) = 1/2 x' P x
 * @param  P quadratic matrix in CSC form (only upper triangular)
 * @param  x argument float vector
 * @return   quadratic form value
 */
c_float OSQPMatrix_quad_form(const OSQPMatrix  *P,
                             const OSQPVectorf *x);


# ifdef __cplusplus
}
# endif // ifdef __cplusplus

#endif // ifndef LIN_ALG_H
