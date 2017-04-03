#include "auxil.h"

/***********************************************************
 * Auxiliary functions needed to compute ADMM iterations * *
 ***********************************************************/
 #ifndef EMBEDDED
 /**
  * Automatically compute rho
  * @param work Workspace
  */
 void compute_rho(OSQPWorkspace * work){
    c_float trP, trAtA, ratio;

    if (work->data->m == 0){ // No consraints. Use max rho
        work->settings->rho = AUTO_RHO_MAX;
        return;
    }

    // Compute tr(P)
    trP = mat_trace(work->data->P);

    // Compute tr(AtA) = fro(A) ^ 2
    trAtA = mat_fro_sq(work->data->A);
    trAtA *= trAtA;

    if (trAtA < 1e-05){ // tr(AtA) = 0
        work->settings->rho = AUTO_RHO_MAX;
        return;
    }

    // Compute ratio
    ratio = trP / trAtA;

    // Compute rho
    work->settings->rho = AUTO_RHO_OFFSET + AUTO_RHO_SLOPE * ratio;

 }
 #endif // ifndef EMBEDDED


 /**
  * Swap c_float vector pointers
  * @param a first vector
  * @param b second vector
  */
 void swap_vectors(c_float ** a, c_float ** b){
     c_float * temp;
     temp = *b;
     *b = *a;
     *a = temp;
 }


/**
 * Cold start workspace variables
 * @param work Workspace
 */
void cold_start(OSQPWorkspace *work) {

    vec_set_scalar(work->x, 0., work->data->n);
    vec_set_scalar(work->z, 0., work->data->m);
    vec_set_scalar(work->y, 0., work->data->m);

}


/**
 * Update RHS during first tep of ADMM iteration. Store it into (x,z).
 * @param  work Workspace
 */
static void compute_rhs(OSQPWorkspace *work){
    c_int i; // Index
    for (i=0; i < work->data->n; i++){
        // Cycle over part related to x variables
        work->xz_tilde[i] = work->settings->sigma * work->x_prev[i] - work->data->q[i];
    }
    for (i = 0; i < work->data->m; i++){
        // Cycle over dual variable in the first step (nu)
        work->xz_tilde[i + work->data->n] = work->z_prev[i] - (c_float) 1./work->settings->rho * work->y[i];
    }

}


/**
 * Update z_tilde variable after solving linear system (first ADMM step)
 *
 * @param work Workspace
 */
static void update_z_tilde(OSQPWorkspace *work){
    c_int i; // Index
    for (i = 0; i < work->data->m; i++){
        work->xz_tilde[i + work->data->n] = work->z_prev[i] + (c_float) 1./work->settings->rho * (work->xz_tilde[i + work->data->n] - work->y[i]);
    }
}


/**
 * Update x_tilde and z_tilde variable (first ADMM step)
 * @param work [description]
 */
void update_xz_tilde(OSQPWorkspace * work){
    // Compute right-hand side
    compute_rhs(work);

    // Solve linear system
    solve_lin_sys(work->settings, work->priv, work->xz_tilde);

    // Update z_tilde variable after solving linear system
    update_z_tilde(work);
}


/**
* Update x (second ADMM step)
* Update also delta_x (For dual infeasibility)
* @param work Workspace
*/
void update_x(OSQPWorkspace * work){
    c_int i;

    // update x
    for (i = 0; i < work->data->n; i++){
        work->x[i] = work->settings->alpha * work->xz_tilde[i] +
                     ((c_float) 1.0 - work->settings->alpha) * work->x_prev[i];
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
void update_z(OSQPWorkspace *work){
    c_int i;

    // update z
    for (i = 0; i < work->data->m; i++){
        work->z[i] = work->settings->alpha * work->xz_tilde[i + work->data->n] +
                     ((c_float) 1.0 - work->settings->alpha) * work->z_prev[i] +
                     (c_float) 1./work->settings->rho * work->y[i];
    }

    // project z
    project_z(work);

}



/**
 * Update y variable (third ADMM step)
 * Update also delta_y to check for infeasibility
 * @param work Workspace
 */
void update_y(OSQPWorkspace *work){
    c_int i; // Index
    for (i = 0; i < work->data->m; i++){

        work->delta_y[i] = work->settings->rho *
            (work->settings->alpha * work->xz_tilde[i + work->data->n] +
            ((c_float) 1.0 - work->settings->alpha) * work->z_prev[i] - work->z[i]);
        work->y[i] += work->delta_y[i];

    }
}

/**
 * Compute objective function from data at value x
 * @param  data OSQPData structure
 * @param  x       Value x
 * @return         Objective function value
 */
c_float compute_obj_val(OSQPData *data, c_float * x) {
        return quad_form(data->P, x) +
               vec_prod(data->q, x, data->n);
}


/**
 * Return norm of primal residual
 * @param  work   Workspace
 * @param  polish Called from polish function (1) or from elsewhere (0)
 * @return        Norm of primal residual
 */
c_float compute_pri_res(OSQPWorkspace * work, c_int polish){

    // If embedded we cannot access polish members
    #ifndef EMBEDDED
    c_int j;
    c_float tmp, prim_resid_sq = 0.0;
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
    #endif
        // Called from ADMM algorithm (store temporary vector in z_prev)
        mat_vec(work->data->A, work->x, work->z_prev, 0);
        return vec_norm2_diff(work->z_prev, work->z, work->data->m);

    #ifndef EMBEDDED
    }
    #endif
}



/**
 * Return norm of dual residual
 * TODO: Use more tailored residual (not general one)
 * @param  work   Workspace
 * @param  polish Called from polish() function (1) or from elsewhere (0)
 * @return        Norm of dual residual
 */
c_float compute_dua_res(OSQPWorkspace * work, c_int polish){

    // N.B. Use x_prev as temporary vector

    #ifndef EMBEDDED
    if (!polish){ // Normal call
    #endif
        // dual_res = q
        prea_vec_copy(work->data->q, work->x_prev, work->data->n);

        // += A' * y
        if (work->data->m > 0)
            mat_tpose_vec(work->data->A, work->y, work->x_prev, 1, 0);

        // += P * x (upper triangular part)
        mat_vec(work->data->P, work->x, work->x_prev, 1);

        // += P' * x (lower triangular part with no diagonal)
        mat_tpose_vec(work->data->P, work->x, work->x_prev, 1, 1);

        // Return norm
        return vec_norm2(work->x_prev, work->data->n);

    #ifndef EMBEDDED
    } else {  // Call after polish
        // Called from polish() function
        // dr = q + Ared'*y_red + P*x
        // NB: Only upper triangular part of P is stored.
        prea_vec_copy(work->data->q, work->x_prev,
                      work->data->n);                    // dr = q
        mat_tpose_vec(work->pol->Ared, work->pol->y_red,
                      work->x_prev, 1, 0);  // += Ared'*y_red
        mat_vec(work->data->P, work->pol->x,
                work->x_prev, 1);               // += Px (upper triang part)
        mat_tpose_vec(work->data->P, work->pol->x,
                      work->x_prev, 1, 1);      // += Px (lower triang part)
        return vec_norm2(work->x_prev, work->data->n);
    }
    #endif
}


/**
 * Check if problem is primal infeasible
 * @param  work Workspace
 * @return      Integer for True or False
 */
c_int is_primal_infeasible(OSQPWorkspace * work){
    c_int i; // Index for loops
    c_float norm_delta_y, ineq_lhs = 0;

    // Compute norm of delta_y
    norm_delta_y = vec_norm2(work->delta_y, work->data->m);

    if (norm_delta_y > work->settings->eps_prim_inf*work->settings->eps_prim_inf){ // ||delta_y|| > 0
        // Normalize delta_y by its norm
        vec_mult_scalar(work->delta_y, (c_float) 1./norm_delta_y, work->data->m);

        // Compute check
        // ineq_lhs = u'*max(delta_y, 0) + l'*min(delta_y, 0) < 0
        for (i = 0; i < work->data->m; i++){
            ineq_lhs += work->data->u[i] * c_max(work->delta_y[i], 0) + work->data->l[i] * c_min(work->delta_y[i], 0);
        }

        if (ineq_lhs < -work->settings->eps_prim_inf){ // Condition satisfied
            // Compute and return ||A'delta_y|| < eps_prim_inf
            mat_tpose_vec(work->data->A, work->delta_y, work->Atdelta_y, 0, 0);
            return vec_norm2(work->Atdelta_y, work->data->n) < work->settings->eps_prim_inf;
        }
    }

    // Conditions not satisfied -> not primal infeasible
    return 0;

}

/**
 * Check if problem is dual infeasible
 * @param  work Workspace
 * @return        Integer for True or False
 */
c_int is_dual_infeasible(OSQPWorkspace * work){
    c_int i; // Index for loops
    c_float norm_delta_x;

    // Compute norm of delta_x
    norm_delta_x = vec_norm2(work->delta_x, work->data->n);

    // Prevent 0 division || delta_x || > 0
    if (norm_delta_x > work->settings->eps_dual_inf*work->settings->eps_dual_inf){

        // Normalize delta_x by its norm
        vec_mult_scalar(work->delta_x, (c_float) 1./norm_delta_x, work->data->n);

        // Check first if q'*delta_x < 0
        if (vec_prod(work->data->q, work->delta_x, work->data->n) <
            -work->settings->eps_dual_inf){

            // Compute product P * delta_x
            mat_vec(work->data->P, work->delta_x, work->Pdelta_x, 0);

            // Check if || P * delta_x || = 0
            if (vec_norm2(work->Pdelta_x, work->data->n) < work->settings->eps_dual_inf){

                // Compute A * delta_x
                mat_vec(work->data->A, work->delta_x, work->Adelta_x, 0);

                // De Morgan Law Applied to dual infeasibility conditions for A * x
                // N.B. Note that 1e-03 is used to adjust the infinity value
                //      in case the problem is scaled.
                for (i = 0; i < work->data->m; i++){
                    if (((work->data->u[i] < OSQP_INFTY*1e-03) && (work->Adelta_x[i] >  work->settings->eps_dual_inf)) ||
                    ((work->data->l[i] > -OSQP_INFTY*1e-03) && (work->Adelta_x[i] < -work->settings->eps_dual_inf))){
                        // At least one condition not satisfied
                        return 0;
                    }
                }

                // All conditions passed -> dual infeasible
                return 1;
            }
        }
    }

    // Conditions not satisfied -> not dual infeasible
    return 0;

}


/**
 * Store the QP solution
 * @param work Workspace
 */
void store_solution(OSQPWorkspace *work) {
    if ((work->info->status_val != OSQP_PRIMAL_INFEASIBLE) &&
        (work->info->status_val != OSQP_DUAL_INFEASIBLE)){
        prea_vec_copy(work->x, work->solution->x, work->data->n);   // primal
        prea_vec_copy(work->y, work->solution->y, work->data->m);  // dual

        if(work->settings->scaling) // Unscale solution if scaling has been performed
            unscale_solution(work);
    } else { // Problem primal or dual infeasible. Solution is NaN
        vec_set_scalar(work->solution->x, OSQP_NAN, work->data->n);
        vec_set_scalar(work->solution->y, OSQP_NAN, work->data->m);

        // Cold start iterates to 0 for next runs
        cold_start(work);
    }
}


/**
* Update solver information
* @param work               Workspace
* @param iter               Iteration number
* @param compute_objective  Boolean (if compute the objective or not)
* @param polish             Boolean (if called from polish)
*/
void update_info(OSQPWorkspace *work, c_int iter, c_int compute_objective, c_int polish){

    #ifndef EMBEDDED
    if (polish) { // polish

        // Always compute objective value when called from polish
        work->pol->obj_val = compute_obj_val(work->data, work->pol->x);

        if (work->data->m == 0) {
            // No constraints -> Always primal feasible
            work->pol->pri_res = 0.;
        } else {
            work->pol->pri_res = compute_pri_res(work, 1);
        }
        work->pol->dua_res = compute_dua_res(work, 1);

    } else { // normal update
    #endif

        work->info->iter = iter; // Update iteration number

        // Check if we need to compute the objective
        if (compute_objective){
            work->info->obj_val = compute_obj_val(work->data, work->x);
        }

        if (work->data->m == 0) {
            // No constraints -> Always primal feasible
              work->info->pri_res = 0.;
        } else {
              work->info->pri_res = compute_pri_res(work, 0);
        }
        work->info->dua_res = compute_dua_res(work, 0);

        #ifdef PROFILING
            work->info->solve_time = toc(work->timer);
        #endif

    #ifndef EMBEDDED
    }
    #endif
}


/**
 * Update solver status (value and string)
 * @param work Workspace
 */
void update_status(OSQPInfo *info, c_int status_val) {
    // Update status value
    info->status_val = status_val;

    // Update status string depending on status val
    if(status_val == OSQP_SOLVED)
        c_strcpy(info->status, "Solved");
    else if (status_val == OSQP_PRIMAL_INFEASIBLE)
        c_strcpy(info->status, "Primal infeasible");
    else if (status_val == OSQP_UNSOLVED)
        c_strcpy(info->status, "Unsolved");
    else if (status_val == OSQP_DUAL_INFEASIBLE)
        c_strcpy(info->status, "Dual infeasible");
    else if (status_val == OSQP_MAX_ITER_REACHED)
        c_strcpy(info->status, "Maximum iterations reached");
    else if (status_val == OSQP_SIGINT)
        c_strcpy(info->status, "Interrupted");
}



/**
 * Check if termination conditions are satisfied
 * @param  work Workspace
 * @return      Redisuals check
 */
c_int check_termination(OSQPWorkspace *work){
    c_float eps_pri, eps_dua;
    c_int exitflag = 0;
    c_int pri_res_check = 0, dua_res_check = 0, prim_inf_check = 0, dual_inf_check = 0;

    // Check residuals
    if (work->data->m == 0){
        pri_res_check = 1;  // No contraints -> Primal feasibility always satisfied
    }
    else {
        // Compute primal tolerance
        eps_pri = c_sqrt((c_float) work->data->m) * work->settings->eps_abs +
                  work->settings->eps_rel * vec_norm2(work->z, work->data->m);
        // Primal feasibility check
        if (work->info->pri_res < eps_pri) pri_res_check = 1;

        // Primal infeasibility check
        prim_inf_check = is_primal_infeasible(work);
    }

    // Compute dual tolerance
    mat_tpose_vec(work->data->A, work->y, work->x_prev, 0, 0); // ws = A'*u
    eps_dua = c_sqrt((c_float) work->data->n) * work->settings->eps_abs +
              work->settings->eps_rel * work->settings->rho *
              vec_norm2(work->x_prev, work->data->n);
    // Dual feasibility check
    if (work->info->dua_res < eps_dua) dua_res_check = 1;


    // Check dual infeasibility
    dual_inf_check = is_dual_infeasible(work);

    // Compare checks to determine solver status
    if (pri_res_check && dua_res_check){
        // Update final information
        update_status(work->info, OSQP_SOLVED);
        exitflag = 1;
    }
    else if (prim_inf_check){
        // Update final information
        update_status(work->info, OSQP_PRIMAL_INFEASIBLE);
        work->info->obj_val = OSQP_INFTY;
        exitflag = 1;
    }
    else if (dual_inf_check){
        // Update final information
        update_status(work->info, OSQP_DUAL_INFEASIBLE);
        work->info->obj_val = -OSQP_INFTY;
        exitflag = 1;
    }

    return exitflag;

}


#ifndef EMBEDDED

/**
 * Validate problem data
 * @param  data OSQPData to be validated
 * @return      Exitflag to check
 */
c_int validate_data(const OSQPData * data){
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
 * @param  data OSQPData to be validated
 * @return      Exitflag to check
 */
c_int validate_settings(const OSQPSettings * settings){
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
    if (settings->auto_rho != 0 &&  settings->auto_rho != 1) {
        #ifdef PRINTING
        c_print("auto_rho must be either 0 or 1\n");
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
    if (settings->eps_prim_inf <= 0) {
        #ifdef PRINTING
        c_print("eps_prim_inf must be positive\n");
        #endif
        return 1;
    }
    if (settings->eps_dual_inf <= 0) {
        #ifdef PRINTING
        c_print("eps_dual_inf must be positive\n");
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
    if (settings->early_terminate != 0 && settings->early_terminate != 1) {
        #ifdef PRINTING
        c_print("early_terminate must be either 0 or 1\n");
        #endif
        return 1;
    }
    if (settings->early_terminate_interval <= 0) {
        #ifdef PRINTING
        c_print("early_terminate_interval must be positive\n");
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

#endif  // #ifndef EMBEDDED
