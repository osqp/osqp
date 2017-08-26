#include "proj.h"



void project(OSQPWorkspace *work, c_float * z){
    c_int i;

    for (i = 0 ; i < work->data->m; i++){
        z[i] = c_min(c_max(z[i],
                     work->data->l[i]), // Between lower
                     work->data->u[i]); // and upper bounds
    }

}


void project_normalcone(OSQPWorkspace *work, c_float *z, c_float *y){
    c_int j;

    // NB: Use z_prev as temporary vector

    for (j = 0; j < work->data->m; j++) {
        work->z_prev[j] = z[j] + y[j];
        z[j] = c_min(c_max(work->z_prev[j], work->data->l[j]), work->data->u[j]);
        y[j] = work->z_prev[j] - z[j];
    }

}
