#ifndef LIN_ALG_H
#define LIN_ALG_H
#include <math.h>
#include "constants.h"

/* ||a - b||_2 */
c_float vec_norm2_diff(const c_float *a, const c_float *b, c_int l);

/* a += sc*b */
void vec_add_scaled(c_float *a, const c_float *b, c_int n, const c_float sc);

/* ||v||_2^2 */
c_float vec_norm2_sq(const c_float *v, c_int len);

/* ||v||_2 */
c_float vec_norm2(const c_float *v, c_int len);

#endif
