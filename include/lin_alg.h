#ifndef LIN_ALG_H
#define LIN_ALG_H
#include <math.h>
#include "constants.h"
#include "glob_opts.h"
#include "cs.h"

/* VECTOR FUNCTIONS ----------------------------------------------------------*/
/* ||a - b||_2 (TODO: if it is only needed for tests remove it and put it in util.h)
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


/* copy vector b into a (TODO: if it is only needed for tests remove it and put it in util.h)
*/
void vec_copy(c_float *a, const c_float *b, c_int n);


/* Vector elementwise reciprocal b = 1./a (needed for scaling)*/
void vec_ew_recipr(const c_float *a, c_float *b, c_int n);


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
void mat_vec(csc *A, const c_float *x, c_float *y, c_int plus_eq);

#endif
