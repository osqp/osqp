#ifndef LIN_ALG_H
# define LIN_ALG_H


# ifdef __cplusplus
extern "C" {
# endif // ifdef __cplusplus

# include "types.h"


/* VECTOR FUNCTIONS ----------------------------------------------------------*/

# ifndef EMBEDDED

/* Return a copy of a float vector a as output (Uses MALLOC)*/
OSQPVectorf* OSQPVectorf_copy_new(OSQPVectorf *a);

/* Return a copy of an int vector a as output (Uses MALLOC)*/
OSQPVectori* OSQPVectori_copy_new(OSQPVectori *a);

# endif // ifndef EMBEDDED

/* copy a float vector a into another vector b (pre-allocated) */
void OSQPVectorf_copy(OSQPVectorf *a,OSQPVectorf *b);

/* copy an int vector a into another vector b (pre-allocated) */
void OSQPVectori_copy(OSQPVectori *a,OSQPVectori *b);

/* set float vector to scalar */
void OSQPVectorf_set_scalar(OSQPVectorf *a, c_float sc);

/* set int vector to scalar */
void OSQPVectori_set_scalar(OSQPVectori *a, c_int sc);

/* add scalar to float vector */
void OSQPVectorf_add_scalar(OSQPVectorf *a, c_float sc);

/* add scalar to int vector */
void OSQPVectori_add_scalar(OSQPVectori *a, c_int sc);

/* multiply float vector by float */
void OSQPVectorf_mult_scalar(OSQPVectorf *a, c_float sc);

/* change sign */
void OSQPVectorf_negate(OSQPVectorf *a);


/* c = a + sc*b */
void OSQPVectorf_add_scaled(OSQPVectorf       *c,
                             const OSQPVectorf *a,
                             const OSQPVectorf *b,
                             c_float           sc);


/* ||v||_inf */
c_float OSQPVectorf_norm_inf(const OSQPVectorf *v);

/* ||v||_1 */
c_float OSQPVectorf_norm_1(const OSQPVectorf *v);

/* ||Sv||_inf */
c_float OSQPVectorf_scaled_norm_inf(const OSQPVectorf *S, const OSQPVectorf *v);

/* ||Sv||_1 */
c_float OSQPVectorf_scaled_norm_1(const OSQPVectorf *S, const OSQPVectorf *v);

/* ||a - b||_inf */
c_float OSQPVectorf_norm_inf_diff(const OSQPVectorf *a,
                                  const OSQPVectorf *b);


/* sum of vector elements */
c_float OSQPVectorf_sum(const OSQPVectorf *a);

/* mean of vector elements */
c_float OSQPVectorf_mean(const OSQPVectorf *a);



/* Inner product a'b */
c_float OSQPVectorf_dot_prod(const OSQPVectorf *a,
                             const OSQPVectorf *b);

/* elementwise product a.*b stored in c*/
void OSQPVectorf_ew_prod(const OSQPVectorf *a,
                         const OSQPVectorf *b,
                         OSQPVectorf       *c);


# if EMBEDDED != 1

/* Vector elementwise reciprocal b = 1./a (needed for scaling)*/
void OSQPVectorf_ew_reciprocal(const OSQPVectorf *a, OSQPVectorf *b);

/* elementwise sqrt of the vector elements */
void OSQPVectorf_ew_sqrt(OSQPVectorf *a);

/* elementwise max between each vector component and max_val */
void OSQPVectorf_ew_max(OSQPVectorf *a, c_float max_val);

/* elementwise max between each vector component and max_val */
void OSQPVectorf_ew_min(OSQPVectorf *a, c_float min_val);

/* Elementwise maximum between vectors c = max(a, b) */
void OSQPVectorf_ew_max_vec(const OSQPVectorf *a,
                            const OSQPVectorf *b,
                            OSQPVectorf       *c);

/* Elementwise minimum between vectors c = min(a, b) */
void OSQPVectorf_ew_min_vec(const OSQPVectorf *a,
                            const OSQPVectorf *b,
                            OSQPVectorf       *c);

# endif // if EMBEDDED != 1


/* MATRIX FUNCTIONS ----------------------------------------------------------*/

/* multiply matrix by scalar */
void OSQPMatrix_mult_scalar(OSQPMatrix *A, c_float sc);

/* Premultiply (i.e. left) matrix A by diagonal matrix with diagonal d,
   i.e. scale the rows of A by d
 */
void OSQPMatrix_premult_diag(OSQPMatrix *A, const OSQPVectorf *d);

/* Postmultiply (i.e. right) matrix A by diagonal matrix with diagonal d,
   i.e. scale the columns of A by d
 */
void OSQPMatrix_postmult_diag(OSQPMatrix *A, const OSQPVectorf *d);


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
 */
 void OSQPMatrix_Atx(const OSQPMatrix  *A,
                     const OSQPVectorf *x,
                     OSQPVectorf       *y,
                     c_int             sign);


# if EMBEDDED != 1

/**
 * Infinity norm of each matrix column
 * @param M	Input matrix
 * @param E     Vector of infinity norms
 *
 */
void OSQPMatrix_inf_norm_cols(const OSQPMatrix *M,OSQPVectorf *E);

/**
 * Infinity norm of each matrix row
 * @param M	Input matrix
 * @param E     Vector of infinity norms
 *
 */
void OSQPMatrix_inf_norm_rows(const OSQPMatrix *M,OSQPVectorf *E);

# endif // EMBEDDED != 1

/**
 * Compute quadratic form f(x) = 1/2 x' P x
 * @param  P symmetric matrix
 * @param  x argument float vector
 * @return   quadratic form value
 */
c_float OSQPMatrix_quad_form(const OSQPMatrix  *P, const OSQPVectorf *x);

# ifdef __cplusplus
}
# endif // ifdef __cplusplus

#endif // ifndef LIN_ALG_H
