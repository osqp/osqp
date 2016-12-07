#include "auxil.h"
#include "util.h"
#include "proj.h"

/***********************************************************
 * Auxiliary functions needed to compute ADMM iterations * *
 ***********************************************************/

/**
 * Cold start workspace variables
 * @param work Workspace
 */
void cold_start(Work *work) {
    memset(work->x, 0, work->data->n * sizeof(c_float));
    memset(work->z, 0, work->data->m * sizeof(c_float));
    memset(work->y, 0, work->data->m * sizeof(c_float));
}


/**
 * Update RHS during first tep of ADMM iteration. Store it into (x,z).
 * @param  work Workspace
 */
static void compute_rhs(Work *work){
    c_int i; // Index
    for (i=0; i < work->data->n; i++){
        // Cycle over part related to x variables
        work->xz_tilde[i] = work->settings->rho * work->x[i] - work->data->q[i];
    }
    for (i = 0; i < work->data->m; i++){
        // Cycle over dual variable in the first step (nu)
        work->xz_tilde[i + work->data->n] = work->z_prev[i] - 1./work->settings->rho * work->y[i];
    }

}


/**
 * Update z_tilde variable after solving linear system (first ADMM step)
 *
 * @param work Workspace
 */
static void update_z_tilde(Work *work){
    c_int i; // Index
    for (i = 0; i < work->data->m; i++){
        work->xz_tilde[i + work->data->n] = work->z_prev[i] + 1./work->settings->rho * (work->xz_tilde[i + work->data->n] - work->y[i]);
    }
}


/**
 * Update x_tilde and z_tilde variable (first ADMM step)
 * @param work [description]
 */
void update_xz_tilde(Work * work){
    // Compute right-hand side
    compute_rhs(work);

    // Solve linear system
    solve_lin_sys(work->settings, work->priv, work->xz_tilde);

    // Update z_tilde variable after solving linear system
    update_z_tilde(work);
}


/**
* Update x (second ADMM step)
* @param work Workspace
*/
void update_x(Work * work){
    c_int i;

    // update x
    for (i = 0; i < work->data->n; i++){
        work->x[i] = work->settings->alpha * work->xz_tilde[i] +
                     (1.0 - work->settings->alpha) * work->x_prev[i];
    }
}


/**
* Update z (third ADMM step)
* @param work Workspace
*/
void update_z(Work *work){
    c_int i;

    // update z
    for (i = 0; i < work->data->m; i++){
        work->z[i] = work->settings->alpha * work->xz_tilde[i + work->data->n] +
                     (1.0 - work->settings->alpha) * work->z_prev[i] +
                     1./work->settings->rho * work->y[i];
    }

    // project z
    project_z(work);

}



/**
 * Update y variable (third ADMM step)
 * @param work Workspace
 */
void update_y(Work *work){
    c_int i; // Index
    for (i = 0; i < work->data->m; i++){

        #ifndef SKIP_INFEASIBILITY

        work->delta_y[i] = work->settings->rho *
                           (work->settings->alpha * work->xz_tilde[i + work->data->n] +
                           (1.0 - work->settings->alpha) * work->z_prev[i] - work->z[i]);
        work->y[i] += work->delta_y[i];

        #else

        work->y[i] += work->settings->rho *
                      (work->settings->alpha * work->xz_tilde[i + work->data->n] +
                      (1.0 - work->settings->alpha) * work->z_prev[i] - work->z[i]);

        #endif
    }
}

/**
 * Compute objective function from data at value x
 * @param  data Data structure
 * @param  x       Value x
 * @param  polish  Compute obj_val from polish data or not (bool)
 * @return         Objective function value
 */
c_float compute_obj_val(Work *work, c_int polish) {
    if (polish) {
        return quad_form(work->data->P, work->pol->x) +
               vec_prod(work->data->q, work->pol->x, work->data->n);
    } else {
        return quad_form(work->data->P, work->x) +
               vec_prod(work->data->q, work->x, work->data->n);
    }
}


/**
 * Return norm of primal residual
 * @param  work   Workspace
 * @param  polish Called from polish function (1) or from elsewhere (0)
 * @return        Norm of primal residual
 */
c_float compute_pri_res(Work * work, c_int polish){
    c_int j;
    c_float tmp, prim_resid_sq=0;
    if (polish) {
        // Called from polish() function
        for (j = 0; j < work->data->m; j++) {
            if (work->pol->Ax[j] < work->data->l[j]) {
                tmp = work->data->l[j] - work->pol->Ax[j];
                prim_resid_sq += tmp*tmp;
            } else if (work->pol->Ax[j] > work->data->u[j]) {
                tmp = work->pol->Ax[j] - work->data->u[j];
                prim_resid_sq += tmp*tmp;
            }
        }
        return c_sqrt(prim_resid_sq);
    } else {
        // Called from ADMM algorithm
        mat_vec(work->data->A, work->x, work->pri_res_vec, 0);
        return vec_norm2_diff(work->pri_res_vec, work->z, work->data->m);
    //
    //
    //
    //     for (j = 0; j < work->data->m; j++) {
    //         if (work->x[work->data->n + j] < work->data->l[j]) {
    //             tmp = work->data->l[j] - work->x[work->data->n + j];
    //             prim_resid_sq += tmp*tmp;
    //         } else if (work->x[work->data->n + j] > work->data->u[j]) {
    //             tmp = work->x[work->data->n + j] - work->data->u[j];
    //             prim_resid_sq += tmp*tmp;
    //         }
    //     }
    // }
    // return c_sqrt(prim_resid_sq);
    }
}



/**
 * Return norm of dual residual
 * TODO: Use more tailored residual (not general one)
 * @param  work   Workspace
 * @param  polish Called from polish() function (1) or from elsewhere (0)
 * @return        Norm of dual residual
 */
c_float compute_dua_res(Work * work, c_int polish){

    if (!polish){
        // dual_res = q
        prea_vec_copy(work->data->q, work->dua_res_vec, work->data->n);

        // += A' * y
        mat_tpose_vec(work->data->A, work->y, work->dua_res_vec, 1, 0);

        // += P * x (upper triangular part)
        mat_vec(work->data->P, work->x, work->dua_res_vec, 1);

        // += P' * x (lower triangular part with no diagonal)
        mat_tpose_vec(work->data->P, work->x, work->dua_res_vec, 1, 1);

        // Return norm
        return vec_norm2(work->dua_res_vec, work->data->n);

    } else {
        // Called from polish() function
        // dr = q + Ared'*y_red + P*x
        // NB: Only upper triangular part of P is stored.
        prea_vec_copy(work->data->q, work->dua_res_vec,
                      work->data->n);                    // dr = q
        mat_tpose_vec(work->pol->Ared, work->pol->y_red,
                      work->dua_res_vec, 1, 0);      // += Ared'*y_red
        mat_vec(work->data->P, work->pol->x,
                work->dua_res_vec, 1);               // += Px (upper triang part)
        mat_tpose_vec(work->data->P, work->pol->x,
                      work->dua_res_vec, 1, 1);      // += Px (lower triang part)
        return vec_norm2(work->dua_res_vec, work->data->n);
    }
}

    // if (polish) {
    //     // Called from polish() function
    //     // dr = q + Ared'*y_red + P*x
    //     // NB: Only upper triangular part of P is stored.
    //     prea_vec_copy(work->data->q, work->dua_res_ws_n,
    //                   work->data->n);                    // dr = q
    //     mat_tpose_vec(work->pol->Ared, work->pol->y_red,
    //                   work->dua_res_ws_n, 1, 0);      // += Ared'*y_red
    //     mat_vec(work->data->P, work->pol->x,
    //             work->dua_res_ws_n, 1);               // += Px (upper triang part)
    //     mat_tpose_vec(work->data->P, work->pol->x,
    //                   work->dua_res_ws_n, 1, 1);      // += Px (lower triang part)
    //     return vec_norm2(work->dua_res_ws_n, work->data->n);
    // } else {
    //     // TODO: Update computation of the dual residual
    //     // Called from ADMM algorithm
    //     // -dr = rho * [I  A']( z^{k+1} + (alpha-2)*z^k + (1-alpha)*x^{k+1} )
    //     // NB: I compute negative dual residual for the convenience
    //     prea_vec_copy(work->z, work->dua_res_ws_n, work->data->n);  // dr = z_x
    //     vec_add_scaled(work->dua_res_ws_n, work->z_prev,
    //                    work->data->n, work->settings->alpha-2.0);   // += (alpha-2)*z_prev_x
    //     vec_add_scaled(work->dua_res_ws_n, work->x,
    //                    work->data->n, 1.0-work->settings->alpha);  // += (1-alpha)*x_x
    //
    //     prea_vec_copy(work->z + work->data->n, work->dua_res_ws_m,
    //                   work->data->m);                       // dr = z_s
    //     vec_add_scaled(work->dua_res_ws_m, work->z_prev + work->data->n,
    //                    work->data->m, work->settings->alpha-2.0); // += (alpha-2)*z_prev_s
    //     vec_add_scaled(work->dua_res_ws_m, work->x + work->data->n,
    //                    work->data->m, 1.0-work->settings->alpha); // += (1-alpha)*x_s
    //     mat_tpose_vec(work->data->A, work->dua_res_ws_m,
    //                   work->dua_res_ws_n, 1, 0);
    //     return (work->settings->rho * vec_norm2(work->dua_res_ws_n, work->data->n));
    // }

// }


#ifndef SKIP_INFEASIBILITY
/**
 * Compute norm of infeasibility residual
 * @param  work Workspace
 * @return      Norm of infeasibility residual
 */
c_float compute_inf_res(Work * work){
    // c_int j;
    // c_float tmp, infeas_resid_sq=0;
    //
    // for (j = 0; j < work->data->m; j++) {
    //     if (work->x[work->data->n + j] < work->data->l[j]) {
    //         tmp = work->z[work->data->n + j] - work->data->l[j];
    //     } else if (work->x[work->data->n + j] > work->data->u[j]) {
    //         tmp = work->data->u[j] - work->z[work->data->n + j];
    //     } else {
    //         tmp = work->x[work->data->n + j] - work->z[work->data->n + j];
    //     }
    //     infeas_resid_sq += tmp*tmp;
    // }
    // return c_sqrt(infeas_resid_sq);
    return vec_norm2_diff(work->delta_y, work->delta_y_prev, work->data->m);
}
#endif


/**
 * Store the QP solution
 * @param work Workspace
 */
void store_solution(Work *work) {
    prea_vec_copy(work->x, work->solution->x, work->data->n);       // primal
    prea_vec_copy(work->y, work->solution->y, work->data->m);  // dual
    // vec_mult_scalar(work->solution->y, work->settings->rho, work->data->m);

    if(work->settings->scaling) // Unscale solution if scaling has been performed
        unscale_solution(work);
}

/**
 * Update solver information
 * @param work   Workspace
 * @param iter   Number of iterations
 * @param polish Called from polish function (1) or from elsewhere (0)
 */
void update_info(Work *work, c_int iter, c_int polish){
    if (work->data->m == 0) {  // No constraints in the problem (no polishing)
        work->info->iter = iter; // Update iteration number
        work->info->obj_val = compute_obj_val(work, 0);
        work->info->pri_res = 0.;  // Always primal feasible
        work->info->dua_res = compute_dua_res(work, 0);
        #ifdef PROFILING
            work->info->solve_time = toc(work->timer);
        #endif
    }
    else{ // Problem has constraints
        if (!polish) { // No polishing
            work->info->iter = iter; // Update iteration number
            work->info->obj_val = compute_obj_val(work, 0);
            work->info->pri_res = compute_pri_res(work, 0);
            work->info->dua_res = compute_dua_res(work, 0);
            #ifndef SKIP_INFEASIBILITY
            work->info->inf_res = compute_inf_res(work);
            #endif
            #ifdef PROFILING
                work->info->solve_time = toc(work->timer);
            #endif
        } else { // Polishing
            work->pol->obj_val = compute_obj_val(work, 1);
            work->pol->pri_res = compute_pri_res(work, 1);
            work->pol->dua_res = compute_dua_res(work, 1);
        }
    }
}


/**
 * Update solver status (string)
 * @param work Workspace
 */
void update_status_string(Info *info){
    // Update status string depending on status val

    if(info->status_val == OSQP_SOLVED)
        strcpy(info->status, "Solved");
    else if (info->status_val == OSQP_INFEASIBLE)
        strcpy(info->status, "Infeasible");
    else if (info->status_val == OSQP_UNSOLVED)
        strcpy(info->status, "Unsolved");
    else if (info->status_val == OSQP_MAX_ITER_REACHED)
        strcpy(info->status, "Maximum Iterations Reached");
}



/**
 * Check if residuals norm meet the required tolerance
 * @param  work Workspace
 * @return      Redisuals check
 */
c_int residuals_check(Work *work){
    c_float eps_pri, eps_dua;
    c_int exitflag = 0;
    c_int pri_check = 0, dua_check = 0;
    #if SKIP_INFEASIBILITY == 0
    c_int inf_check = 0;
    #endif


    // Check residuals
    if (work->data->m == 0){
        pri_check = 1;  // No contraints -> Primal feasibility always satisfied
    }
    else {
        // Compute primal tolerance
        eps_pri = c_sqrt(work->data->m) * work->settings->eps_abs +
                  work->settings->eps_rel * vec_norm2(work->z, work->data->m);
        // Primal feasibility check
        if (work->info->pri_res < eps_pri) pri_check = 1;

        #if SKIP_INFEASIBILITY == 0
        // Infeasibility check
        if (work->info->inf_res < 1e-2*eps_pri &&
            vec_norm2(work->delta_y, work->data->m) > 1e2*eps_pri) {
            inf_check = 1;
            // c_print("Inf residual condition True\n");
            // c_print("Inf residual = %e\n", work->info->inf_res);
            // c_print("eps_dua = %e\n", eps_dua);
            // c_print("eps_pri = %e\n", eps_pri);
        }
        #endif
    }

    // Compute dual tolerance
    mat_tpose_vec(work->data->A, work->y, work->dua_res_vec, 0, 0); // ws = A'*u
    eps_dua = c_sqrt(work->data->n) * work->settings->eps_abs +
              work->settings->eps_rel * work->settings->rho *
              vec_norm2(work->dua_res_vec, work->data->n);
    // Dual feasibility check
    if (work->info->dua_res < eps_dua) dua_check = 1;


    // Compare checks to determine solver status
    if (pri_check && dua_check){
        // Update final information
        work->info->status_val = OSQP_SOLVED;
        exitflag = 1;
    }

    #if SKIP_INFEASIBILITY == 0
    else if ((!pri_check) & dua_check & inf_check){
        // Update final information
        work->info->status_val = OSQP_INFEASIBLE;
        exitflag = 1;
    }
    #endif



    // if (work->info->pri_res < eps_pri && work->info->dua_res < eps_dua) {
    //     // Update final information
    //     work->info->status_val = OSQP_SOLVED;
    //     exitflag = 1;
    // } else if (work->info->pri_res > eps_pri && work->info->dua_res < eps_dua
    //                                          && work->info->inf_res < eps_pri) {
    //    // Update final information
    //    work->info->status_val = OSQP_INFEASIBLE;
    //    exitflag = 1;
    // }

    return exitflag;

}


/**
 * Validate problem data
 * @param  data Data to be validated
 * @return      Exitflag to check
 */
c_int validate_data(const Data * data){
    int j;

    if(!data){
        #ifdef PRINTING
        c_print("Missing data!\n");
        #endif
        return 1;
    }

    // General dimensions Tests
    if (data->n <= 0 || data->m < 0){
        #ifdef PRINTING
        c_print("n must be positive and m nonnegative; n = %i, m = %i\n",
                 data->n, data->m);
        #endif
        return 1;
    }

    // Matrix P
    if (data->P->m != data->n ){
        #ifdef PRINTING
        c_print("P does not have dimension n x n with n = %i\n", data->n);
        #endif
        return 1;
    }
    if (data->P->m != data->P->n ){
        #ifdef PRINTING
        c_print("P is not square\n");
        #endif
        return 1;
    }

    // Matrix A
    if (data->A->m != data->m || data->A->n != data->n){
        #ifdef PRINTING
        c_print("A does not have dimension m x n with m = %i and n = %i\n",
                data->m, data->n);
        #endif
        return 1;
    }

    // Lower and upper bounds
    for (j = 0; j < data->m; j++) {
        if (data->l[j] > data->u[j]) {
            #ifdef PRINTING
            c_print("Lower bound at index %d is greater than upper bound: %.4e > %.4e\n",
                  j, data->l[j], data->u[j]);
            #endif
          return 1;
        }
    }

    // TODO: Complete with other checks

    return 0;
}


/**
 * Validate problem settings
 * @param  data Data to be validated
 * @return      Exitflag to check
 */
c_int validate_settings(const Settings * settings){
    if (!settings){
        #ifdef PRINTING
        c_print("Missing settings!\n");
        #endif
        return 1;
    }
    if (settings->scaling != 0 &&  settings->scaling != 1) {
        #ifdef PRINTING
        c_print("scaling must be either 0 or 1\n");
        #endif
        return 1;
    }
    if (settings->scaling_norm != 1 &&  settings->scaling_norm != 2) {
        #ifdef PRINTING
        c_print("scaling_norm must be either 1 or 2\n");
        #endif
        return 1;
    }
    if (settings->scaling_iter < 1) {
        #ifdef PRINTING
        c_print("scaling_iter must be greater than 0\n");
        #endif
        return 1;
    }
    if (settings->pol_refine_iter < 0) {
        #ifdef PRINTING
        c_print("pol_refine_iter must be nonnegative\n");
        #endif
        return 1;
    }

    if (settings->rho <= 0) {
        #ifdef PRINTING
        c_print("rho must be positive\n");
        #endif
        return 1;
    }
    if (settings->delta <= 0) {
        #ifdef PRINTING
        c_print("delta must be positive\n");
        #endif
        return 1;
    }
    if (settings->max_iter <= 0) {
        #ifdef PRINTING
        c_print("max_iter must be positive\n");
        #endif
        return 1;
    }
    if (settings->eps_abs <= 0) {
        #ifdef PRINTING
        c_print("eps_abs must be positive\n");
        #endif
        return 1;
    }
    if (settings->eps_rel <= 0) {
        #ifdef PRINTING
        c_print("eps_rel must be positive\n");
        #endif
        return 1;
    }
    if (settings->alpha <= 0 || settings->alpha >= 2) {
        #ifdef PRINTING
        c_print("alpha must be between 0 and 2\n");
        #endif
        return 1;
    }
    if (settings->verbose != 0 && settings->verbose != 1) {
        #ifdef PRINTING
        c_print("verbose must be either 0 or 1\n");
        #endif
        return 1;
    }
    if (settings->warm_start != 0 && settings->warm_start != 1) {
        #ifdef PRINTING
        c_print("warm_start must be either 0 or 1\n");
        #endif
        return 1;
    }

    return 0;

}
