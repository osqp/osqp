#include "osqp.h"

OSQPFloat vec_norm_inf(const OSQPFloat* v, OSQPInt l) {
  OSQPInt   i;
  OSQPFloat abs_v_i;
  OSQPFloat max = 0.0;

  for (i = 0; i < l; i++) {
    abs_v_i = c_absval(v[i]);

    if (abs_v_i > max) max = abs_v_i;
  }
  return max;
}

OSQPFloat vec_norm_inf_diff(const OSQPFloat* a, const OSQPFloat* b, OSQPInt l) {
  OSQPFloat nmDiff = 0.0, tmp;
  OSQPInt   i;

  for (i = 0; i < l; i++) {
    tmp = c_absval(a[i] - b[i]);

    if (tmp > nmDiff) nmDiff = tmp;
  }
  return nmDiff;
}
