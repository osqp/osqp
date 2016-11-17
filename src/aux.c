#include "aux.h"
#include "util.h"

/***********************************************************
 * Auxiliary functions needed to compute ADMM iterations * *
 ***********************************************************/

/**
 * Cold start workspace variables
 * @param work Workspace
 */
void cold_start(Work *work) {
    memset(work->x, 0, (work->data->n + work->data->m) * sizeof(c_float));
    memset(work->z, 0, (work->data->n + work->data->m) * sizeof(c_float));
    memset(work->u, 0, (work->data->m) * sizeof(c_float));
}


/**
 * Update RHS during first tep of ADMM iteration (store it into x)
 * @param  work Workspace
 */
void compute_rhs(Work *work){
    c_int i; // Index
    for (i=0; i < work->data->n; i++){
        // Cycle over part related to original x variables
        work->x[i] = work->settings->rho * work->z[i] - work->data->q[i];
    }
    for (i = work->data->n; i < work->data->n + work->data->m; i++){
        // Cycle over dual variable within first step (nu)
        work->x[i] = work->z[i] - work->u[i - work->data->n];
    }

}


/**
 * Update x variable (slacks s related part)
 * after solving linear system (first ADMM step)
 *
 * @param work Workspace
 */
void update_x(Work *work){
    c_int i; // Index
    for (i = work->data->n; i < work->data->n + work->data->m; i++){
        work->x[i] = 1./work->settings->rho * work->x[i] + work->z[i] - work->u[i - work->data->n];
        //TODO: Remove 1/rho operation (store 1/rho during setup)
    }
}


/**
 * Project x (second ADMM step)
 * @param work Workspace
 */
void project_x(Work *work){
    c_int i;

    for (i = 0; i < work->data->n; i++){
        // Part related to original x variables (no projection)
        work->z[i] = work->settings->alpha * work->x[i] +
                     (1.0 - work->settings->alpha) * work->z_prev[i];
    }

    for (i = work->data->n; i < work->data->n + work->data->m; i++){
        // Part related to slack variables
        work->z[i] = c_min(c_max(work->settings->alpha * work->x[i] +
                     (1.0 - work->settings->alpha) * work->z_prev[i] +
                     work->u[i - work->data->n], work->data->lA[i - work->data->n]), work->data->uA[i - work->data->n]);
    }

}

/**
 * Update u variable (third ADMM step)
 * @param work Workspace
 */
void update_u(Work *work){
    c_int i; // Index
    for (i = work->data->n; i < work->data->n + work->data->m; i++){
        work->u[i - work->data->n] += work->settings->alpha * work->x[i] +
                      (1.0 - work->settings->alpha) * work->z_prev[i] -
                      work->z[i];
    }
}

/**
 * Compute objective function from data at value x
 * @param  data Data structure
 * @param  x    Value x
 * @return      Objective function value
 */
c_float compute_obj_val(Work *work, c_int polish) {
    if (polish) {
        return quad_form(work->data->P, work->act->x) +
               vec_prod(work->data->q, work->act->x, work->data->n);
    } else {
        return quad_form(work->data->P, work->x) +
               vec_prod(work->data->q, work->x, work->data->n);
    }
}


/**
 * Return norm of primal residual
 * TODO: Use more tailored residual (not general one)
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
            if (work->act->Ax[j] < work->data->lA[j]) {
                tmp = work->data->lA[j] - work->act->Ax[j];
                prim_resid_sq += tmp*tmp;
            } else if (work->act->Ax[j] > work->data->uA[j]) {
                tmp = work->act->Ax[j] - work->data->uA[j];
                prim_resid_sq += tmp*tmp;
            }
        }
    } else {
        // Called from ADMM algorithm
        for (j = 0; j < work->data->m; j++) {
            if (work->x[work->data->n + j] < work->data->lA[j]) {
                tmp = work->data->lA[j] - work->x[work->data->n + j];
                prim_resid_sq += tmp*tmp;
            } else if (work->x[work->data->n + j] > work->data->uA[j]) {
                tmp = work->x[work->data->n + j] - work->data->uA[j];
                prim_resid_sq += tmp*tmp;
            }
        }
    }
    return c_sqrt(prim_resid_sq);
}



/**
 * Return norm of dual residual
 * TODO: Use more tailored residual (not general one)
 * @param  work   Workspace
 * @param  polish Called from polish() function (1) or from elsewhere (0)
 * @return        Norm of dual residual
 */
c_float compute_dua_res(Work * work, c_int polish){
    if (polish) {
        // Called from polish() function
        // dr = q + Ared'*lambda_red + P*x
        // NB: Only upper triangular part of P is stored.
        prea_vec_copy(work->data->q, work->dua_res_ws_n,
                      work->data->n);                    // dr = q
        mat_vec_tpose(work->act->Ared, work->act->lambda_red,
                      work->dua_res_ws_n, 1, 0);      // += Ared'*lambda_red
        mat_vec(work->data->P, work->act->x,
                work->dua_res_ws_n, 1);               // += Px (upper triang part)
        mat_vec_tpose(work->data->P, work->act->x,
                      work->dua_res_ws_n, 1, 1);      // += Px (lower triang part)
        return vec_norm2(work->dua_res_ws_n, work->data->n);
    } else {
        // TODO: Update computation of the dual residual
        // Called from ADMM algorithm
        // -dr = rho * [I  A']( z^{k+1} + (alpha-2)*z^k + (1-alpha)*x^{k+1} )
        // NB: I compute negative dual residual for the convenience
        prea_vec_copy(work->z, work->dua_res_ws_n, work->data->n);  // dr = z_x
        vec_add_scaled(work->dua_res_ws_n, work->z_prev,
                       work->data->n, work->settings->alpha-2.0);   // += (alpha-2)*z_prev_x
        vec_add_scaled(work->dua_res_ws_n, work->x,
                       work->data->n, 1.0-work->settings->alpha);  // += (1-alpha)*x_x

        prea_vec_copy(work->z + work->data->n, work->dua_res_ws_m,
                      work->data->m);                       // dr = z_s
        vec_add_scaled(work->dua_res_ws_m, work->z_prev + work->data->n,
                       work->data->m, work->settings->alpha-2.0); // += (alpha-2)*z_prev_s
        vec_add_scaled(work->dua_res_ws_m, work->x + work->data->n,
                       work->data->m, 1.0-work->settings->alpha); // += (1-alpha)*x_s
        mat_vec_tpose(work->data->A, work->dua_res_ws_m,
                      work->dua_res_ws_n, 1, 0);
        return (work->settings->rho * vec_norm2(work->dua_res_ws_n, work->data->n));
    }
}


/**
 * Store the QP solution
 * @param work Workspace
 */
void store_solution(Work *work) {
    prea_vec_copy(work->x, work->solution->x, work->data->n);   // primal
    vec_add_scaled(work->solution->lambda, work->u,             // dual
                   work->data->m, work->settings->rho);
}

/**
 * Update solver information
 * @param work   Workspace
 * @param iter   Number of iterations
 * @param polish Called from polish function (1) or from elsewhere (0)
 */
void update_info(Work *work, c_int iter, c_int polish){
    if (!polish) {
        work->info->iter = iter; // Update iteration number
        work->info->obj_val = compute_obj_val(work, 0);
        work->info->pri_res = compute_pri_res(work, 0);
        work->info->dua_res = compute_dua_res(work, 0);
        #if PROFILING > 0
            work->info->solve_time = toc(work->timer);
        #endif
    } else {
        work->act->obj_val = compute_obj_val(work, 1);
        work->act->pri_res = compute_pri_res(work, 1);
        work->act->dua_res = compute_dua_res(work, 1);
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

}



/**
 * Check if residuals norm meet the required tolerance
 * @param  work Workspace
 * @return      Redisuals check
 */
c_int residuals_check(Work *work){
    c_float eps_pri, eps_dua;
    c_int exitflag = 0;

    // Compute primal tolerance
    eps_pri = c_sqrt(work->data->m) * work->settings->eps_abs +
              work->settings->eps_rel *
              c_max(vec_norm2(work->x + work->data->n, work->data->m),
                    c_max(vec_norm2(work->data->lA, work->data->m),
                          vec_norm2(work->data->uA, work->data->m)));
    // Compute dual tolerance
    mat_vec_tpose(work->data->A, work->u, work->dua_res_ws_n, 0, 0); // ws = A'*u
    eps_dua = c_sqrt(work->data->n) * work->settings->eps_abs +
              work->settings->eps_rel * work->settings->rho *
              vec_norm2( work->dua_res_ws_n, work->data->n);

    if (work->info->pri_res < eps_pri && work->info->dua_res < eps_dua)
        exitflag = 1;

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
        c_print("Missing data!\n");
        return 1;
    }

    // General dimensions Tests
    if (data->n <= 0 || data->m < 0){
        c_print("n must be positive and m nonnegative; n = %i, m = %i\n",
                 data->n, data->m);
        return 1;
    }

    // Matrix P
    if (data->P->m != data->n ){
        c_print("P does not have dimension n x n with n = %i\n", data->n);
        return 1;
    }
    if (data->P->m != data->P->n ){
        c_print("P is not square\n");
        return 1;
    }

    // Matrix A
    if (data->A->m != data->m || data->A->n != data->n){
        c_print("A does not have dimension m x n with m = %i and n = %i\n",
                data->m, data->n);
        return 1;
    }

    // Lower and upper bounds
    for (j = 0; j < data->m; j++) {
        if (data->lA[j] > data->uA[j]) {
          c_print("Lower bound at index %d is greater than upper bound: %.4e > %.4e\n",
                  j, data->lA[j], data->uA[j]);
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
        c_print("Missing settings!\n");
        return 1;
    }
    if (settings->normalize != 0 &&  settings->normalize != 1) {
        c_print("normalize must be either 0 or 1\n");
        return 1;
    }
    if (settings->rho <= 0) {
        c_print("rho must be positive\n");
        return 1;
    }


    if (settings->max_iter <= 0) {
        c_print("max_iter must be positive\n");
        return 1;
    }
    if (settings->eps_abs <= 0) {
        c_print("eps_abs must be positive\n");
        return 1;
    }
    if (settings->eps_rel <= 0) {
        c_print("eps_rel must be positive\n");
        return 1;
    }
    if (settings->alpha <= 0) {
        c_print("alpha must be positive\n");
        return 1;
    }
    if (settings->verbose != 0 &&  settings->verbose != 1) {
        c_print("verbose must be either 0 or 1\n");
        return 1;
    }
    if (settings->warm_start != 0 &&  settings->warm_start != 1) {
        c_print("warm_start must be either 0 or 1\n");
        return 1;
    }

    // TODO: Complete with other checks


    return 0;

}
