#include "proj.h"


void project(OSQPVectorf*       z,
             const OSQPVectorf* l,
             const OSQPVectorf* u) {

  OSQPVectorf_ew_bound_vec(z, z, l, u);
}

void project_polar_reccone(OSQPVectorf*       y,
                           const OSQPVectorf* l,
                           const OSQPVectorf* u,
                           OSQPFloat          infval) {

  OSQPVectorf_project_polar_reccone(y, l, u, infval);
}

OSQPInt test_in_reccone(const OSQPVectorf* y,
                        const OSQPVectorf* l,
                        const OSQPVectorf* u,
                              OSQPFloat    infval,
                              OSQPFloat    tol) {

  return OSQPVectorf_in_reccone(y, l, u, infval, tol);
}

#ifndef OSQP_EMBEDDED_MODE

void project_normalcone(OSQPVectorf*       z,
                        OSQPVectorf*       y,
                        const OSQPVectorf* l,
                        const OSQPVectorf* u) {

  // y <- y + z;  z <- proj_C(y);  y <- y - z
  OSQPVectorf_plus(y, y, z);
  OSQPVectorf_ew_bound_vec(z, y, l, u);
  OSQPVectorf_minus(y, y, z);
}

# endif /* ifndef OSQP_EMBEDDED_MODE */
