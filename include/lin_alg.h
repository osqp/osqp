#ifndef LIN_ALG_H
#define LIN_ALG_H


#ifdef __cplusplus
extern "C" {
#endif

#include "types.h"



/* VECTOR FUNCTIONS ----------------------------------------------------------*/

#ifndef EMBEDDED

/* copy vector a into output (Uses MALLOC)*/
c_float * vec_copy(c_float *a, c_int n);

#endif

/* copy vector a into preallocated vector b */
void prea_vec_copy(c_float *a, c_float * b, c_int n);

/* copy integer vector a into preallocated vector b */
void prea_int_vec_copy(c_int *a, c_int * b, c_int n);

/* set float vector to scalar */
void vec_set_scalar(c_float *a, c_float sc, c_int n);

/* set integer vector to scalar */
void int_vec_set_scalar(c_int *a, c_int sc, c_int n);

/* add scalar to vector*/
void vec_add_scalar(c_float *a, c_float sc, c_int n);

/* multiply scalar to vector*/
void vec_mult_scalar(c_float *a, c_float sc, c_int n);


/* a += sc*b */
void vec_add_scaled(c_float *a, const c_float *b, c_int n, c_float sc);

/* ||v||_inf */
c_float vec_norm_inf(const c_float *v, c_int l);

/* ||a - b||_inf */
c_float vec_norm_inf_diff(const c_float *a, const c_float *b, c_int l);


/* ||a - b||^2 */
c_float vec_norm2_sq_diff(const c_float *a, const c_float *b, c_int l);

/* ||v||_2^2 */
c_float vec_norm2_sq(const c_float *v, c_int l);


/* Vector elementwise reciprocal b = 1./a (needed for scaling)*/
void vec_ew_recipr(const c_float *a, c_float *b, c_int n);


/* Inner product a'b */
c_float vec_prod(const c_float *a, const c_float *b, c_int n);

/* elementwse product a.*b stored in b*/
void vec_ew_prod(const c_float *a, c_float *b, c_int n);

#if EMBEDDED != 1
/* elementwise sqrt of the vector elements */
void vec_ew_sqrt(c_float *a, c_int n);
#endif

/* elementwise max between each vector component and max_val */
void vec_ew_max(c_float *a, c_int n, c_float max_val);

/* elementwise min between each vector component and max_val */
void vec_ew_min(c_float *a, c_int n, c_float min_val);


/* MATRIX FUNCTIONS ----------------------------------------------------------*/
/* Vertically concatenate arrays and return Z = [A' B']'
(uses MALLOC to create inner arrays x, i, p within Z)
*/
// csc * vstack(csc *A, csc *B);


/* Premultiply matrix A by diagonal matrix with diagonal d,
i.e. scale the rows of A by d
*/
void mat_premult_diag(csc *A, const c_float *d);

/* Premultiply matrix A by diagonal matrix with diagonal d,
i.e. scale the columns of A by d
*/
void mat_postmult_diag(csc *A, const c_float *d);

#ifndef EMBEDDEED
/* Elementwise square matrix M */
void mat_ew_sq(csc * A);

/* Elementwise absolute value of matrix M */
void mat_ew_abs(csc * A);

/**
 * Trace of matrix M in cdc format
 * @param  M Input matrix
 * @return   Trace
 */
c_float mat_trace(csc * M);

/**
 * Frobenius norm squared of matrix M
 * @param  M Input matrix
 * @return   Frobenius norm squared
 */
c_float mat_fro_sq(csc * M);
#endif // ifndef embedded


/* Matrix-vector multiplication
 *    y  =  A*x  (if plus_eq == 0)
 *    y +=  A*x  (if plus_eq == 1)
 *    y -=  A*x  (if plus_eq == -1)
*/
void mat_vec(const csc *A, const c_float *x, c_float *y, c_int plus_eq);


/* Matrix-transpose-vector multiplication
 *    y  =  A'*x  (if plus_eq == 0)
 *    y +=  A'*x  (if plus_eq == 1)
 *    y -=  A'*x  (if plus_eq == -1)
 * If skip_diag == 1, then diagonal elements of A are assumed to be zero.
*/
void mat_tpose_vec(const csc *A, const c_float *x, c_float *y,
                   c_int plus_eq, c_int skip_diag);

/**
 * Compute quadratic form f(x) = 1/2 x' P x
 * @param  P quadratic matrix in CSC form (only upper triangular)
 * @param  x argument float vector
 * @return   quadratic form value
 */
c_float quad_form(const csc * P, const c_float * x);


#ifdef __cplusplus
}
#endif

#endif
