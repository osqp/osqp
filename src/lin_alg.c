#include "lin_alg.h"
// #include <math.h>

/* ||a - b||_2 */
c_float vec_norm2_diff(const c_float *a, const c_float *b, c_int l) {
    c_float nmDiff = 0.0, tmp;
    c_int i;
    for (i = 0; i < l; ++i) {
        tmp = (a[i] - b[i]);
        nmDiff += tmp * tmp;
    }
    return c_sqrtf(nmDiff);
}

/* a += sc*b */
void vec_add_scaled(c_float *a, const c_float *b, c_int n, const c_float sc) {
    c_int i;
    for (i = 0; i < n; ++i) {
        a[i] += sc * b[i];
    }
}

/* ||v||_2^2 */
c_float vec_norm2_sq(const c_float *v, c_int len) {
    c_int i;
    c_float nmsq = 0.0;
    for (i = 0; i < len; ++i) {
        nmsq += v[i] * v[i];
    }
    return nmsq;
}

/* ||v||_2 */
c_float vec_norm2(const c_float *v, c_int len) {
    return c_sqrtf(vec_norm2_sq(v, len));
}

/* ||v||_inf */
c_float vec_normInf(const c_float *a, c_int l) {
    c_float tmp, max = 0.0;
    c_int i;
    for (i = 0; i < l; ++i) {
        tmp = c_abs(a[i]);
        if (tmp > max)
            max = tmp;
    }
    return max;
}

/* copy vector b into a */
void vec_copy(c_float *a, c_float *b, c_int n) {
    for (c_int i=0; i<n; i++) {
        a[i] = b[i];
    }
}
