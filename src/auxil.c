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
        work->xz_tilde[i] = work->settings->sigma * work->x[i] - work->data->q[i];
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
* Update also delta_x (For unboundedness)
* @param work Workspace
*/
void update_x(Work * work){
    c_int i;

    // update x
    for (i = 0; i < work->data->n; i++){
        work->x[i] = work->settings->alpha * work->xz_tilde[i] +
                     (1.0 - work->settings->alpha) * work->x_prev[i];
    }

    // update delta_x
    for (i = 0; i < work->data->n; i++){
        work->delta_x[i] = work->x[i] - work->x_prev[i];
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
 * Update also delta_y to check for infeasibility
 * @param work Workspace
 */
void update_y(Work *work){
    c_int i; // Index
    for (i = 0; i < work->data->m; i++){

        work->delta_y[i] = work->settings->rho *
            (work->settings->alpha * work->xz_tilde[i + work->data->n] +
            (1.0 - work->settings->alpha) * work->z_prev[i] - work->z[i]);
        work->y[i] += work->delta_y[i];

    }
}

/**
 * Compute objective function from data at value x
 * @param  data Data structure
 * @param  x       Value x
 * @return         Objective function value
 */
c_float compute_obj_val(Data *data, c_float * x) {
        return quad_form(data->P, x) +
               vec_prod(data->q, x, data->n);
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
            if (work->pol->z[j] < work->data->l[j]) {
                tmp = work->data->l[j] - work->pol->z[j];
                prim_resid_sq += tmp*tmp;
            } else if (work->pol->z[j] > work->data->u[j]) {
                tmp = work->pol->z[j] - work->data->u[j];
                prim_resid_sq += tmp*tmp;
            }
        }
        return c_sqrt(prim_resid_sq);
    } else {
        // Called from ADMM algorithm (store temporary vector in z_prev)
        mat_vec(work->data->A, work->x, work->z_prev, 0);
        return vec_norm2_diff(work->z_prev, work->z, work->data->m);
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

    // N.B. Use x_prev as temporary vector

    if (!polish){ // Normal call
        // dual_res = q
        prea_vec_copy(work->data->q, work->x_prev, work->data->n);

        // += A' * y
        mat_tpose_vec(work->data->A, work->y, work->x_prev, 1, 0);

        // += P * x (upper triangular part)
        mat_vec(work->data->P, work->x, work->x_prev, 1);

        // += P' * x (lower triangular part with no diagonal)
        mat_tpose_vec(work->data->P, work->x, work->x_prev, 1, 1);

        // Return norm
        return vec_norm2(work->x_prev, work->data->n);

    } else {  // Call after polish
        // Called from polish() function
        // dr = q + Ared'*y_red + P*x
        // NB: Only upper triangular part of P is stored.
        prea_vec_copy(work->data->q, work->x_prev,
                      work->data->n);                    // dr = q
        mat_tpose_vec(work->pol->Ared, work->pol->y_red,
                      work->x_prev, 1, 0);      // += Ared'*y_red
        mat_vec(work->data->P, work->pol->x,
                work->x_prev, 1);               // += Px (upper triang part)
        mat_tpose_vec(work->data->P, work->pol->x,
                      work->x_prev, 1, 1);      // += Px (lower triang part)
        return vec_norm2(work->x_prev, work->data->n);
    }
}


/**
 * Check if problem is infeasible
 * @param  work Workspace
 * @return      Integer for True or False
 */
c_int is_infeasible(Work * work){
    c_int i; // Index for loops
    c_float norm_delta_y, ineq_lhs = 0;

    // Compute norm of delta_y
    norm_delta_y = vec_norm2(work->delta_y, work->data->m);

    if (norm_delta_y > work->settings->eps_inf*work->settings->eps_inf){ // ||delta_y|| > 0
        // Normalize delta_y by its norm
        vec_mult_scalar(work->delta_y, 1./norm_delta_y, work->data->m);

        // Compute check
        // ineq_lhs = u'*max(delta_y, 0) + l'*min(delta_y, 0) < 0
        for (i = 0; i < work->data->m; i++){
            ineq_lhs += work->data->u[i] * c_max(work->delta_y[i], 0) + work->data->l[i] * c_min(work->delta_y[i], 0);
        }

        if (ineq_lhs < -work->settings->eps_inf){ // Condition satisfied
            // Compute and return ||A'delta_y|| < eps_inf
            mat_tpose_vec(work->data->A, work->delta_y, work->Atdelta_y, 0, 0);
            return vec_norm2(work->Atdelta_y, work->data->n) < work->settings->eps_inf;
        }
    }

    // Conditions not satisfied -> not infeasible
    return 0;

}

/**
 * Check if problem is unbounded
 * @param  work Workspace
 * @return        Integer for True or False
 */
c_int is_unbounded(Work * work){
    c_int i; // Index for loops
    c_float norm_delta_x;

    // Compute norm of delta_x
    norm_delta_x = vec_norm2(work->delta_x, work->data->n);

    // Prevent 0 division || delta_x || > 0
    if (norm_delta_x > work->settings->eps_unb*work->settings->eps_unb){

        // Normalize delta_x by its norm
        vec_mult_scalar(work->delta_x, 1./norm_delta_x, work->data->n);

        // Check first if q'*delta_x < 0
        if (vec_prod(work->data->q, work->delta_x, work->data->n) <
            -work->settings->eps_unb){

            // Compute product P * delta_x
            mat_vec(work->data->P, work->delta_x, work->Pdelta_x, 0);

            // Check if || P * delta_x || = 0
            if (vec_norm2(work->Pdelta_x, work->data->n) < work->settings->eps_unb){

                // Compute A * delta_x
                mat_vec(work->data->A, work->delta_x, work->Adelta_x, 0);

                // De Morgan Law Applied to Unboundedness conditions for A * x
                // See Section "Detecting infeasibility and unboundedness" of
                // OSQP Paper
                // N.B. Note that 1e-03 is used to adjust the infinity value
                //      in case the problem is scaled.
                for (i = 0; i < work->data->m; i++){
                    if (((work->data->u[i] < OSQP_INFTY*1e-03) && (work->Adelta_x[i] >  work->settings->eps_unb)) ||
                    ((work->data->l[i] > -OSQP_INFTY*1e-03) && (work->Adelta_x[i] < -work->settings->eps_unb))){
                        // At least one condition not satisfied
                        return 0;
                    }
                }

                // All conditions passed -> Unbounded
                return 1;
            }
        }
    }

    // Conditions not satisfied -> not unbounded
    return 0;

}


/**
 * Store the QP solution
 * @param work Workspace
 */
void store_solution(Work *work) {
    if ((work->info->status_val != OSQP_INFEASIBLE) &&
        (work->info->status_val != OSQP_UNBOUNDED)){
        prea_vec_copy(work->x, work->solution->x, work->data->n);   // primal
        prea_vec_copy(work->y, work->solution->y, work->data->m);  // dual

        if(work->settings->scaling) // Unscale solution if scaling has been performed
            unscale_solution(work);
    } else { // Problem infeasible or unbounded. Solution is NaN
        vec_set_scalar(work->solution->x, OSQP_NAN, work->data->n);
        vec_set_scalar(work->solution->y, OSQP_NAN, work->data->m);
    }
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
        work->info->obj_val = compute_obj_val(work->data, work->x);
        work->info->pri_res = 0.;          // Always primal feasible
        work->info->dua_res = compute_dua_res(work, 0);
        #ifdef PROFILING
            work->info->solve_time = toc(work->timer);
        #endif
    }
    else{ // Problem has constraints
        if (polish) { // polishing

            work->pol->obj_val = compute_obj_val(work->data, work->pol->x);
            work->pol->pri_res = compute_pri_res(work, 1);
            work->pol->dua_res = compute_dua_res(work, 1);

        } else { // normal update

            work->info->iter = iter; // Update iteration number
            work->info->obj_val = compute_obj_val(work->data, work->x);
            work->info->pri_res = compute_pri_res(work, 0);
            work->info->dua_res = compute_dua_res(work, 0);

            #ifdef PROFILING
                work->info->solve_time = toc(work->timer);
            #endif
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
    else if (info->status_val == OSQP_UNBOUNDED)
        strcpy(info->status, "Unbounded");
    else if (info->status_val == OSQP_MAX_ITER_REACHED)
        strcpy(info->status, "Maximum Iterations Reached");
}



/**
 * Check if termination conditions are satisfied
 * @param  work Workspace
 * @return      Redisuals check
 */
c_int check_termination(Work *work){
    c_float eps_pri, eps_dua;
    c_int exitflag = 0;
    c_int pri_check = 0, dua_check = 0, inf_check = 0, unb_check = 0;

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

        // Infeasibility check
        inf_check = is_infeasible(work);
    }

    // Compute dual tolerance
    mat_tpose_vec(work->data->A, work->y, work->x_prev, 0, 0); // ws = A'*u
    eps_dua = c_sqrt(work->data->n) * work->settings->eps_abs +
              work->settings->eps_rel * work->settings->rho *
              vec_norm2(work->x_prev, work->data->n);
    // Dual feasibility check
    if (work->info->dua_res < eps_dua) dua_check = 1;


    // Check unboundedness
    unb_check = is_unbounded(work);

    // Compare checks to determine solver status
    if (pri_check && dua_check){
        // Update final information
        work->info->status_val = OSQP_SOLVED;
        exitflag = 1;
    }
    else if (inf_check){
        // Update final information
        work->info->status_val = OSQP_INFEASIBLE;
        work->info->obj_val = OSQP_INFTY;
        exitflag = 1;
    }
    else if (unb_check){
        // Update final information
        work->info->status_val = OSQP_UNBOUNDED;
        work->info->obj_val = -OSQP_INFTY;
        exitflag = 1;
    }

    return exitflag;

}


/**
 * Validate problem data
 * @param  data Data to be validated
 * @return      Exitflag to check
 */
c_int validate_data(const Data * data){
    c_int j;

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
                 (int)data->n, (int)data->m);
        #endif
        return 1;
    }

    // Matrix P
    if (data->P->m != data->n ){
        #ifdef PRINTING
        c_print("P does not have dimension n x n with n = %i\n", (int)data->n);
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
                (int)data->m, (int)data->n);
        #endif
        return 1;
    }

    // Lower and upper bounds
    for (j = 0; j < data->m; j++) {
        if (data->l[j] > data->u[j]) {
            #ifdef PRINTING
            c_print("Lower bound at index %d is greater than upper bound: %.4e > %.4e\n",
                  (int)j, data->l[j], data->u[j]);
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
