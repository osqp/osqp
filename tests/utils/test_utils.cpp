#include "osqp.h"
#include "osqp_tester.h"

// Needed for the c_absval define
#include "glob_opts.h"

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

OSQPInt isLinsysSupported(enum osqp_linsys_solver_type solver) {
  OSQPInt caps = osqp_capabilities();

  if((caps & OSQP_CAPABILITY_DIRECT_SOLVER) && (solver == OSQP_DIRECT_SOLVER)) {
    return 1;
  }

  if((caps & OSQP_CAPABILITY_INDIRECT_SOLVER) && (solver == OSQP_INDIRECT_SOLVER)) {
    return 1;
  }

  return 0;
}
