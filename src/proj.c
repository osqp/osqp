#include "proj.h"
#include "lin_alg.h"


void project(OSQPWorkspace *work, OSQPVectorf *z) {
  OSQPVectorf_ew_bound_vec(work->data->l, work->data->u,z,z);
}


void project_normalcone(OSQPWorkspace *work, OSQPVectorf *z, OSQPVectorf *y) {

  // NB: Use z_prev as temporary vector

  //z_prev = z + y;
  OSQPVectorf_add_scaled(work->z_prev,z,y,1);

  // z = min(max(z_prev,l),u)
  OSQPVectorf_ew_bound_vec(work->data->l, work->data->u, work->z_prev,z);

  //y = z_prev - z;
  OSQPVectorf_add_scaled(y,work->z_prev,z,-1);
}
