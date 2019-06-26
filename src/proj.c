#include "proj.h"


void project(OSQPWorkspace *work, OSQPVectorf *z) {
  OSQPVectorf_ew_bound_vec(z,z,work->data->l,work->data->u);
}

void project_polar_reccone(OSQPVectorf      *yv,
                           OSQPVectorf      *lv,
                           OSQPVectorf      *uv,
                           c_float       infval){

  c_int i; // Index for loops
  c_int    m = OSQPVectorf_length(yv);
  c_float* y = OSQPVectorf_data(yv);
  c_float* l = OSQPVectorf_data(lv);
  c_float* u = OSQPVectorf_data(uv);

  for (i = 0; i < m; i++) {
    if (u[i]   > +infval) {       // Infinite upper bound
      if (l[i] < -infval) {       // Infinite lower bound
        // Both bounds infinite
        y[i] = 0.0;
      } else {
        // Only upper bound infinite
        y[i] = c_min(y[i], 0.0);
      }
    } else if (l[i] < -infval) {  // Infinite lower bound
      // Only lower bound infinite
      y[i] = c_max(y[i], 0.0);
    }
  }
}

c_int test_in_polar_reccone(OSQPVectorf    *yv,
                          OSQPVectorf      *lv,
                          OSQPVectorf      *uv,
                          c_float       infval,
                          c_float         tol){

  c_int i; // Index for loops

  c_int    m = OSQPVectorf_length(yv);
  c_float* y = OSQPVectorf_data(yv);
  c_float* l = OSQPVectorf_data(lv);
  c_float* u = OSQPVectorf_data(uv);

  for (i = 0; i < m; i++) {
    if (((u[i] < +infval) &&
         (y[i] > +tol)) ||
        ((l[i] > -infval) &&
         (y[i] < -tol))) {
      // At least one condition not satisfied -> not dual infeasible
      return 0;
    }
  }
  return 1;
}

void project_normalcone(OSQPWorkspace *work, OSQPVectorf *z, OSQPVectorf *y) {

  // NB: Use z_prev as temporary vector

  //z_prev = z + y;
  OSQPVectorf_plus(work->z_prev,z,y);

  // z = min(max(z_prev,l),u)
  OSQPVectorf_ew_bound_vec(z, work->z_prev, work->data->l, work->data->u);

  //y = z_prev - z;
  OSQPVectorf_minus(y,work->z_prev,z);
}
