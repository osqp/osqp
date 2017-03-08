#include "scaling.h"

#if EMBEDDED != 1
// Scale data stored in workspace
c_int scale_data(OSQPWorkspace * work){
    // Temporary pointers to P->x and A->x to keep them
    c_float * P_x_temp;
    c_float * A_x_temp;

    c_int i; // Iteration


    // Store P->x and A->x in P_x and A_x and assign pointers to them
    prea_vec_copy(work->data->P->x, work->P_x, work->data->P->p[work->data->n]);
    prea_vec_copy(work->data->A->x, work->A_x, work->data->A->p[work->data->n]);
    P_x_temp = work->data->P->x; work->data->P->x = work->P_x;
    A_x_temp = work->data->A->x; work->data->A->x = work->A_x;

    // Perform elementwise operations on A and P
    if (work->settings->scaling_norm == 1) {
        mat_ew_abs(work->data->P);
        mat_ew_abs(work->data->A);
    }
    else if (work->settings->scaling_norm == 2) {
        mat_ew_sq(work->data->P);
        mat_ew_sq(work->data->A);
    }

    for(i = 0; i < work->settings->scaling_iter; i++){

        // Move current vectors to temporary ones
        prea_vec_copy(work->scaling->D, work->D_temp, work->data->n);
        prea_vec_copy(work->scaling->E, work->E_temp, work->data->m);

        // Compute (in case scaling norm is 2 we call Psq, Asq)
        //
        //      [d = Psq * d + Asq' * e]
        //      [e = Asq * d           ]

        // d = Psq * d
        //  N.B. We compute d = Psq * d + Psq' * d (no diagonal included) because
        //      only upper triangular part of P is stored
        mat_vec(work->data->P, work->D_temp, work->scaling->D, 0);
        mat_tpose_vec(work->data->P, work->D_temp, work->scaling->D, 1, 1);

        // d += Asq' * e
        mat_tpose_vec(work->data->A, work->E_temp, work->scaling->D, 1, 0);

        // e = Asq * d
        mat_vec(work->data->A, work->D_temp, work->scaling->E, 0);


        // d = d + SCALING_REG
        // e = e + SCALING_REG
        vec_add_scalar(work->scaling->D, SCALING_REG, work->data->n);
        vec_add_scalar(work->scaling->E, SCALING_REG, work->data->m);

        // d = 1./d
        // e = 1./e
        vec_ew_recipr(work->scaling->D, work->scaling->D, work->data->n);
        vec_ew_recipr(work->scaling->E, work->scaling->E, work->data->m);

        // d = (n + m) * d
        // e = (n + m) * e
        vec_mult_scalar(work->scaling->D, work->data->n + work->data->m,
                        work->data->n);
        vec_mult_scalar(work->scaling->E, work->data->n + work->data->m,
                        work->data->m);

        // Bound vectors between maximum and minimum allowed scaling
        vec_ew_max(work->scaling->D, work->data->n, MIN_SCALING);
        vec_ew_min(work->scaling->D, work->data->n, MAX_SCALING);
        vec_ew_max(work->scaling->E, work->data->m, MIN_SCALING);
        vec_ew_min(work->scaling->E, work->data->m, MAX_SCALING);

    }


    // Finally normalize by sqrt if 2-norm involved (see pdf)
    if (work->settings->scaling_norm == 2) {
        vec_ew_sqrt(work->scaling->D, work->data->n);
        vec_ew_sqrt(work->scaling->E, work->data->m);
    }

    // Store Dinv, Einv in workspace
    vec_ew_recipr(work->scaling->D, work->scaling->Dinv, work->data->n);
    vec_ew_recipr(work->scaling->E, work->scaling->Einv, work->data->m);

    // DEBUG
    // c_print("n = %i\n", work->data->n);
    // print_vec(s, n_plus_m, "s");
    // print_vec(work->scaling->D, work->data->n, "D");
    // print_vec(work->scaling->Dinv, work->data->n, "Dinv");
    // print_vec(work->scaling->E, work->data->m, "E");
    // print_vec(work->scaling->Einv, work->data->m, "Einv");

    // Restore values of P->x and A->x from stored ones
    work->data->P->x = P_x_temp;
    work->data->A->x = A_x_temp;

    // Scale data
    mat_premult_diag(work->data->P, work->scaling->D);
    mat_postmult_diag(work->data->P, work->scaling->D);
    vec_ew_prod(work->scaling->D, work->data->q, work->data->n);

    mat_premult_diag(work->data->A, work->scaling->E);
    mat_postmult_diag(work->data->A, work->scaling->D);
    vec_ew_prod(work->scaling->E, work->data->l, work->data->m);
    vec_ew_prod(work->scaling->E, work->data->u, work->data->m);


    return 0;
}
#endif

/**
 * Unscale problem matrices
 * @param  work Workspace
 * @return      exitflag
 */
c_int unscale_data(OSQPWorkspace * work){

    mat_premult_diag(work->data->P, work->scaling->Dinv);
    mat_postmult_diag(work->data->P, work->scaling->Dinv);
    vec_ew_prod(work->scaling->Dinv, work->data->q, work->data->n);

    mat_premult_diag(work->data->A, work->scaling->Einv);
    mat_postmult_diag(work->data->A, work->scaling->Dinv);
    vec_ew_prod(work->scaling->Einv, work->data->l, work->data->m);
    vec_ew_prod(work->scaling->Einv, work->data->u, work->data->m);

    return 0;
}



// // Scale solution
// c_int scale_solution(OSQPWorkspace * work){
//
//     // primal
//     vec_ew_prod(work->scaling->Dinv, work->solution->x, work->data->n);
//
//     // dual
//     vec_ew_prod(work->scaling->Einv, work->solution->y, work->data->m);
//
//     return 0;
// }


// Unscale solution
c_int unscale_solution(OSQPWorkspace * work){
    // primal
    vec_ew_prod(work->scaling->D, work->solution->x, work->data->n);

    // dual
    vec_ew_prod(work->scaling->E, work->solution->y, work->data->m);

    return 0;
}
