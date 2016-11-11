#include "polish.h"
#include "util.h"

/***********************************************************
 *   Polishing of the solution obtained from the ADMM    * *
 ***********************************************************/


 /*  Print an int vector
  *  TODO: This function is only for debugging. To be removed.
 */
 void print_vec_int(c_int * x, c_int n, char *name) {
     c_print("%s = [", name);
     for(c_int i=0; i<n; i++) {
         c_print(" %d ", x[i]);
     }
     c_print("]\n");
 }



void solve_polish(Work *work) {
    c_int *ind_lx, *ind_ux, *ind_fx, n_lx=0, n_ux=0, n_fx=0;
    c_int *ind_lA, *ind_uA, *ind_fA, n_lA=0, n_uA=0, n_fA=0;
    c_int i;
    // c_float center;

    // Allocate memory for indices showing which constraints are active
    ind_lx = c_calloc(1, work->data->n * sizeof(c_int));
    ind_ux = c_calloc(1, work->data->n * sizeof(c_int));
    ind_fx = c_calloc(1, work->data->n * sizeof(c_int));
    ind_lA = c_calloc(1, work->data->m * sizeof(c_int));
    ind_uA = c_calloc(1, work->data->m * sizeof(c_int));
    ind_fA = c_calloc(1, work->data->m * sizeof(c_int));

    // Guess which bounds are lower-active, upper-active and free
    for (i = 0; i < work->data->n; i++){
        // center =  0.5 * (work->data->lx[i] + work->data->ux[i]);
        if ( // (work->z[i] <= center + POLISH_TOL) &&
            (work->z[i] - work->data->lx[i] < -work->settings->rho * work->u[i])) {
                ind_lx[n_lx++] = i;       // lower-active
        }
        else if ( // (work->z[i] > center - POLISH_TOL) &&
                 (work->data->ux[i] - work->z[i] < work->settings->rho * work->u[i])) {
                    ind_ux[n_ux++] = i;   // upper-active
        }
        else {
            ind_fx[n_fx++] = i;           // free
        }
    }

    // Guess which linear constraints are lower-active, upper-active and free
    for (i = 0; i < work->data->m; i++){
        // center = 0.5 * (work->data->lA[i] + work->data->uA[i]);
        if ( // (work->z[work->data->n + i] <= center + POLISH_TOL) &&
            (work->z[work->data->n + i] - work->data->lA[i] <
             -work->settings->rho * work->u[work->data->n + i])) {
                ind_lA[n_lA++] = i;       // lower-active
        }
        else if ( // (work->z[work->data->n + i] > center - POLISH_TOL) &&
                 (work->data->uA[i] - work->z[work->data->n + i] <
                  work->settings->rho * work->u[work->data->n + i])) {
                    ind_uA[n_uA++] = i;   // upper-active
        }
        else {
            ind_fA[n_fA++] = i;           // free
        }
    }

    // DEBUG
    print_vec_int(ind_lx, n_lx, "ind_lx");
    print_vec_int(ind_ux, n_ux, "ind_ux");
    print_vec_int(ind_fx, n_fx, "ind_fx");
    c_print("\n");
    print_vec_int(ind_lA, n_lA, "ind_lA");
    print_vec_int(ind_uA, n_uA, "ind_uA");
    print_vec_int(ind_fA, n_fA, "ind_fA");
    c_print("\n");
    print_vec(work->z + work->data->n, work->data->m, "Ax");
    print_vec(work->data->uA, work->data->m, "uA");

    // Update private variable pol. If direct method used, perform factorization

    // Call a function from private.c to solve the corresponding linear system
    // tmp = settings->rho;
    // settings->rho = EPS;
    // solve_lin_sys(settings, pol, rhs);

    // Check whether the dual vars stored in rhs have correct signs

    // If yes, update solution. Otherwise

    // Cleanup
    c_free(ind_lx);
    c_free(ind_ux);
    c_free(ind_fx);
    c_free(ind_lA);
    c_free(ind_uA);
    c_free(ind_fA);
}
