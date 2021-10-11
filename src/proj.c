#include "proj.h"


void project(OSQPVectorf       *z,
             const OSQPVectorf *l,
             const OSQPVectorf *u) {

  OSQPVectorf_ew_bound_vec(z, z, l, u);
}

void project_polar_reccone(OSQPVectorf       *y,
                           const OSQPVectorf *l,
                           const OSQPVectorf *u,
                           c_float            infval) {

  OSQPVectorf_project_polar_reccone(y, l, u, infval);
}

c_int test_in_reccone(const OSQPVectorf *y,
                      const OSQPVectorf *l,
                      const OSQPVectorf *u,
                      c_float            infval,
                      c_float            tol) {

  return OSQPVectorf_in_reccone(y, l, u, infval, tol);
}

#ifndef EMBEDDED

void project_normalcone(OSQPVectorf       *z,
                        OSQPVectorf       *y,
                        const OSQPVectorf *l,
                        const OSQPVectorf *u) {

  // y <- z + y;  z <- proj_C(y);  y <- y - z
  OSQPVectorf_plus(y, z, y);
  OSQPVectorf_ew_bound_vec(z, y, l, u);
  OSQPVectorf_minus(y, y, z);
}

# endif /* ifndef EMBEDDED */
