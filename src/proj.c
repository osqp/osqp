#include "proj.h"


void project(OSQPWorkspace *work, OSQPVectorf *z) {

  OSQPVectorf_ew_bound_vec(z,z,work->data->l,work->data->u);

}

void project_polar_reccone(OSQPVectorf      *y,
                           OSQPVectorf      *l,
                           OSQPVectorf      *u,
                           c_float       infval){

  OSQPVectorf_project_polar_reccone(y,l,u,infval);

}


c_int test_in_polar_reccone(const OSQPVectorf *y,
                            const OSQPVectorf *l,
                            const OSQPVectorf *u,
                            c_float        infval,
                            c_float          tol){

  return OSQPVectorf_in_polar_reccone(y,l,u,infval,tol);
  
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
