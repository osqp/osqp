#include "proj.h"


/**
 * Project z onto C = [l, u]
 * @param work Workspace
 */
void project_z(OSQPWorkspace *work){
    c_int i;

    for (i = 0 ; i < work->data->m; i++){
        work->z[i] = c_min(c_max(work->z[i],
                     work->data->l[i]), // Between lower
                     work->data->u[i]); // and upper bounds
    }

}
