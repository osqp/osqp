#ifndef LIN_ALG_H
#define LIN_ALG_H
#include <math.h>
#include "constants.h"
#include "glob_opts.h"
#include "cs.h"

/* VECTOR FUNCTIONS ----------------------------------------------------------*/
/* ||a - b||_2 */
c_float vec_norm2_diff(const c_float *a, const c_float *b, c_int l);

/* a += sc*b */
void vec_add_scaled(c_float *a, const c_float *b, c_int n, const c_float sc);

/* ||v||_2^2 */
c_float vec_norm2_sq(const c_float *v, c_int l);

/* ||v||_2 */
c_float vec_norm2(const c_float *v, c_int l);

/* ||v||_inf */
c_float vec_normInf(const c_float *a, c_int l);

/* copy vector b into a */
void vec_copy(c_float *a, c_float *b, c_int n);


/* MATRIX FUNCTIONS ----------------------------------------------------------*/
/* Vertically concatenate arrays and return Z = [A' B']'
(uses MALLOC to create inner arrays x, i, p within Z)
*/
csc * vstack(csc *A, csc *B);




#endif
