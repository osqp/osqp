#include "osqp.h"

c_float vec_norm_inf(const c_float *v, c_int l) {
  c_int   i;
  c_float abs_v_i;
  c_float max = 0.0;

  for (i = 0; i < l; i++) {
    abs_v_i = c_absval(v[i]);

    if (abs_v_i > max) max = abs_v_i;
  }
  return max;
}

c_float vec_norm_inf_diff(const c_float *a, const c_float *b, c_int l) {
  c_float nmDiff = 0.0, tmp;
  c_int   i;

  for (i = 0; i < l; i++) {
    tmp = c_absval(a[i] - b[i]);

    if (tmp > nmDiff) nmDiff = tmp;
  }
  return nmDiff;
}
