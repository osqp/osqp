#include "polish.h"
#include "util.h"

/***********************************************************
 *   Polishing of the solution obtained from the ADMM    * *
 ***********************************************************/


void polish(Work *work) {
    c_int *ind_lx, *ind_ux, *ind_fx, n_lx=0, n_ux=0, n_fx=0;
    c_int *ind_lA, *ind_uA, *ind_fA, n_lA=0, n_uA=0, n_fA=0;
    c_int i;

    // Allocate memory for indices showing which constraints are active
    ind_lx = c_malloc(work->data->n * sizeof(c_int));
    ind_ux = c_malloc(work->data->n * sizeof(c_int));
    ind_fx = c_malloc(work->data->n * sizeof(c_int));
    ind_lA = c_malloc(work->data->m * sizeof(c_int));
    ind_uA = c_malloc(work->data->m * sizeof(c_int));
    ind_fA = c_malloc(work->data->m * sizeof(c_int));

    // Guess which bounds are lower-active, upper-active and free
    for (i = 0; i < work->data->n; i++){
        if ((work->z[i] < 0.5 * (work->data->lx[i] + work->data->ux[i])) &&
            (work->z[i] - work->data->lx[i] < work->settings->rho * work->u[i]) {
                ind_lx[n_lx++] = i;       // lower-active
        }
        else if ((work->z[i] > 0.5 * (work->data->lx[i] + work->data->ux[i])) &&
                 (work->data->ux[i] - work->z[i] < work->settings->rho * work->u[i]) {
                    ind_ux[n_ux++] = i;   // upper-active
        }
        else {
            ind_fx[u_fx++] = i;           // free
        }
    }

    // Guess which linear constraints are lower-active, upper-active and free
    for (i = 0; i < work->data->m; i++){
        if ((work->z[work->data->m + i] < 0.5 * (work->data->lA[i] + work->data->uA[i])) &&
            (work->z[work->data->m + i] - work->data->lA[i] <
             work->settings->rho * work->u[work->data->m + i]) {
                ind_lA[n_lA++] = i;       // lower-active
        }
        else if ((work->z[work->data->m + i] > 0.5 * (work->data->lA[i] + work->data->uA[i])) &&
                 (work->data->uA[i] - work->z[work->data->m + i] <
                  work->settings->rho * work->u[work->data->m + i]) {
                    ind_uA[n_uA++] = i;   // upper-active
        }
        else {
            ind_fA[u_fA++] = i;           // free
        }
    }

    // Cleanup
    c_free(ind_lx);
    c_free(ind_ux);
    c_free(ind_fx);
    c_free(ind_lA);
    c_free(ind_uA);
    c_free(ind_fA);
}


#endif
