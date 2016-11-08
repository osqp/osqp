#ifndef LIN_ALG_H
#define LIN_ALG_H
#include <math.h>
#include "constants.h"
#include "glob_opts.h"
#include "cs.h"

/* VECTOR FUNCTIONS ----------------------------------------------------------*/

/* copy vector a into output (Uses MALLOC)*/
c_float * vec_copy(c_float *a, c_int n);

/* copy vector a into preallocated vector b */
void prea_vec_copy(c_float *a, c_float * b, c_int n);


/* ||a - b||_2
*/
c_float vec_norm2_diff(const c_float *a, const c_float *b, c_int l);

/* a += sc*b */
void vec_add_scaled(c_float *a, const c_float *b, c_int n, c_float sc);

/* ||v||_2^2 */
c_float vec_norm2_sq(const c_float *v, c_int l);

/* ||v||_2 */
c_float vec_norm2(const c_float *v, c_int l);

// /* ||v||_inf */
// c_float vec_normInf(const c_float *a, c_int l);


/* Vector elementwise reciprocal b = 1./a (needed for scaling)*/
void vec_ew_recipr(const c_float *a, c_float *b, c_int n);


/* Inner product a'b */
c_float vec_prod(const c_float *a, const c_float *b, c_int n);

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


/* Elementwise square matrix M */
void mat_ew_sq(csc * A);

/* Elementwise absolute value of matrix M */
void mat_ew_abs(csc * A);

/* Matrix-vector multiplication
 *    y  =  A*x  (if plus_eq == 0)
 *    y +=  A*x  (if plus_eq == 1)
 */
void mat_vec(const csc *A, const c_float *x, c_float *y, c_int plus_eq);


/**
 * Compute quadratic form f(x) = 1/2 x' P x
 * @param  P quadratic matrix in CSC form (only upper triangular)
 * @param  x argument float vector
 * @return   quadratic form value
 */
c_float quad_form(const csc * P, const c_float * x);

#endif
