#include "scaling.h"

// Scale data stored in workspace
c_int scale_data(Work * work){

    c_int i, n_plus_m; // Iteration
    n_plus_m = work->data->n + work->data->m;

    // Allocate scaling structure
    work->scaling = c_malloc(sizeof(Scaling));
    work->scaling->D = c_malloc(work->data->n * sizeof(c_float));
    work->scaling->Dinv = c_malloc(work->data->n * sizeof(c_float));
    work->scaling->E = c_malloc(work->data->m * sizeof(c_float));
    work->scaling->Einv = c_malloc(work->data->m * sizeof(c_float));

    // Allocate scaling vectors
    c_float * s = c_malloc(n_plus_m*sizeof(c_float));
    c_float * s_prev = c_malloc(n_plus_m*sizeof(c_float));

    // Set s to ones(n+m)
    vec_set_scalar(s, 1., n_plus_m);
    vec_set_scalar(s_prev, 1., n_plus_m);

    // Form KKT matrix to be scaled
    csc * KKT = form_KKT(work->data->P, work->data->A, 0., 1);

    // DEBUG
    // #if PRINTLEVEL > 0
    // print_csc_matrix(KKT, "KKT");
    // #endif

    // Perform elementwise operations on KKT
    if (work->settings->scaling_norm == 1) {
        mat_ew_abs(KKT);
    }
    else if (work->settings->scaling_norm == 2) {
        mat_ew_sq(KKT);
    }

    for(i = 0; i < work->settings->max_scaling_iter; i++){

        // s_prev = s
        prea_vec_copy(s, s_prev, n_plus_m);

        // s = KKT * s (s=s_prev before this step)
        // N.B. We compute KKT * s + KKT' * s (no diagonal included) because
        //      only upper triangular part of KKT is stored
        mat_vec(KKT, s_prev, s, 0);
        mat_tpose_vec(KKT, s_prev, s, 1, 1);      // += KKT' * x (lower triang part)

        // s = s + scaling_eps
        vec_add_scalar(s, 1E-08, n_plus_m);

        // s = 1./s
        vec_ew_recipr(s, s, n_plus_m);

        // s = (n + m) * s
        vec_mult_scalar(s, n_plus_m, n_plus_m);

        if(vec_norm2_diff(s, s_prev, n_plus_m) < work->settings->scaling_eps)
            break;
    }

    #if PRINTLEVEL > 0
    if (i == work->settings->max_scaling_iter - 1)
        c_print("maximum scaling steps reached\n");
    #endif


    // Finally normalize by sqrt if 2-norm involved (see pdf)
    if (work->settings->scaling_norm == 2) vec_ew_sqrt(s, n_plus_m);

    // Store D, Dinv, E, Einv in workspace
    prea_vec_copy(s, work->scaling->D, work->data->n);
    vec_ew_recipr(work->scaling->D, work->scaling->Dinv, work->data->n);
    prea_vec_copy(s + work->data->n, work->scaling->E, work->data->m);
    vec_ew_recipr(work->scaling->E, work->scaling->Einv, work->data->m);

    // Scale data
    mat_premult_diag(work->data->P, work->scaling->D);
    mat_postmult_diag(work->data->P, work->scaling->D);
    vec_ew_prod(work->scaling->D, work->data->q, work->data->n);

    mat_premult_diag(work->data->A, work->scaling->E);
    mat_postmult_diag(work->data->A, work->scaling->D);
    vec_ew_prod(work->scaling->E, work->data->lA, work->data->m);
    vec_ew_prod(work->scaling->E, work->data->uA, work->data->m);

    // DEBUG: print scaled data
    // #if PRINTLEVEL > 0
    // print_csc_matrix(work->data->P, "P");
    // print_csc_matrix(work->data->A, "A");
    // print_vec(work->data->q, work->data->n, "q");
    // print_vec(work->data->lA, work->data->m, "lA");
    // print_vec(work->data->uA, work->data->m, "uA");
    // print_vec(work->scaling->D, work->data->m, "D");
    // print_vec(work->scaling->Dinv, work->data->m, "Dinv");
    // print_vec(work->scaling->E, work->data->m, "E");
    // print_vec(work->scaling->Einv, work->data->m, "Einv");
    // #endif


    // Free allocated variables
    c_free(s);
    c_free(s_prev);
    csc_spfree(KKT);

    return 0;
}

// // Scale solution
// c_int scale_solution(Work * work){
//     // primal
//     vec_ew_prod(work->scaling->Dinv, work->solution->x, work->data->n);
//
//     // dual
//     vec_ew_prod(work->scaling->Einv, work->solution->lambda, work->data->m);
//
//     return 0;
// }

// Unscale solution
c_int unscale_solution(Work * work){
    // primal
    vec_ew_prod(work->scaling->D, work->solution->x, work->data->n);

    // dual
    vec_ew_prod(work->scaling->E, work->solution->lambda, work->data->m);

    return 0;
}
