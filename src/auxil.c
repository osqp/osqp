#include "auxil.h"

/***********************************************************
 * Auxiliary functions needed to compute ADMM iterations * *
 ***********************************************************/
#if EMBEDDED != 1
void set_rho_vec(OSQPWorkspace * work){
    c_int i;
    work->settings->rho = c_min(c_max(work->settings->rho, RHO_MIN), RHO_MAX);

    for(i=0; i < work->data->m; i++){
        if ( (work->data->l[i] < -OSQP_INFTY*MIN_SCALING) && (work->data->u[i] > OSQP_INFTY*MIN_SCALING) ) {
            // Loose bounds
            work->constr_type[i] = -1;
            work->rho_vec[i] = RHO_MIN;
        } else if (work->data->u[i] - work->data->l[i] < RHO_TOL) {
            // Equality constraints
            work->constr_type[i] = 1;
            work->rho_vec[i] = RHO_MAX;
        } else {
            // Inequality constraints
            work->constr_type[i] = 0;
            work->rho_vec[i] = work->settings->rho;
        }
        work->rho_inv_vec[i] = 1. / work->rho_vec[i];
    }
}

c_int update_rho_vec(OSQPWorkspace * work){
    c_int i, constr_type_changed = 0;

    for(i=0; i < work->data->m; i++){
        if ( (work->data->l[i] < -OSQP_INFTY*MIN_SCALING) && (work->data->u[i] > OSQP_INFTY*MIN_SCALING) ) {
            // Loose bounds
            if (work->constr_type[i] != -1){
                work->constr_type[i] = -1;
                work->rho_vec[i] = RHO_MIN;
                work->rho_inv_vec[i] = 1. / RHO_MIN;
                constr_type_changed = 1;
            }
        } else if (work->data->u[i] - work->data->l[i] < RHO_TOL) {
            // Equality constraints
            if (work->constr_type[i] != 1){
                work->constr_type[i] = 1;
                work->rho_vec[i] = RHO_MAX;
                work->rho_inv_vec[i] = 1. / RHO_MAX;
                constr_type_changed = 1;
            }
        } else {
            // Inequality constraints
            if (work->constr_type[i] != 0){
                work->constr_type[i] = 0;
                work->rho_vec[i] = work->settings->rho;
                work->rho_inv_vec[i] = 1. / work->settings->rho;
                constr_type_changed = 1;
            }
        }
    }

    return constr_type_changed;
}
#endif // EMBEDDED



void swap_vectors(c_float ** a, c_float ** b){
    c_float * temp;
    temp = *b;
    *b = *a;
    *a = temp;
 }


void cold_start(OSQPWorkspace *work) {

    vec_set_scalar(work->x, 0., work->data->n);
    vec_set_scalar(work->z, 0., work->data->m);
    vec_set_scalar(work->y, 0., work->data->m);

}


static void compute_rhs(OSQPWorkspace *work){
    c_int i; // Index
    for (i=0; i < work->data->n; i++){
        // Cycle over part related to x variables
        work->xz_tilde[i] = work->settings->sigma * work->x_prev[i] - work->data->q[i];
    }
    for (i = 0; i < work->data->m; i++){
        // Cycle over dual variable in the first step (nu)
        work->xz_tilde[i + work->data->n] = work->z_prev[i] - work->rho_inv_vec[i] * work->y[i];
    }

}


static void update_z_tilde(OSQPWorkspace *work){
    c_int i; // Index
    for (i = 0; i < work->data->m; i++){
        work->xz_tilde[i + work->data->n] = work->z_prev[i] + work->rho_inv_vec[i] * (work->xz_tilde[i + work->data->n] - work->y[i]);
    }
}


void update_xz_tilde(OSQPWorkspace * work){
    // Compute right-hand side
    compute_rhs(work);

    // Solve linear system
    work->linsys_solver->solve(work->linsys_solver, work->xz_tilde, work->settings);

    // Update z_tilde variable after solving linear system
    update_z_tilde(work);
}


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

void update_z(OSQPWorkspace *work){
    c_int i;

    // update z
    for (i = 0; i < work->data->m; i++){
        work->z[i] = work->settings->alpha * work->xz_tilde[i + work->data->n] +
                     ((c_float) 1.0 - work->settings->alpha) * work->z_prev[i] +
                     work->rho_inv_vec[i] * work->y[i];
    }

    // project z
    project(work, work->z);

}



void update_y(OSQPWorkspace *work){
    c_int i; // Index
    for (i = 0; i < work->data->m; i++){

        work->delta_y[i] = work->rho_vec[i] *
              (work->settings->alpha * work->xz_tilde[i + work->data->n] +
               ((c_float) 1.0 - work->settings->alpha) * work->z_prev[i] - work->z[i]);
        work->y[i] += work->delta_y[i];

    }
}


c_float compute_obj_val(OSQPData *data, c_float * x) {
        return quad_form(data->P, x) +
               vec_prod(data->q, x, data->n);
}


c_float compute_pri_res(OSQPWorkspace * work, c_float * x, c_float * z){

    // NB: Use z_prev as temporary vector
    // pr = Ax - z

    mat_vec(work->data->A, x, work->z_prev, 0);
    vec_add_scaled(work->z_prev, z, work->data->m, -1);

    // If scaling active -> rescale residual
    if (work->settings->scaling && !work->settings->scaled_termination){
        vec_ew_prod(work->scaling->Einv, work->z_prev, work->z_prev, work->data->m);
    }

    // Return norm of the residual
    return vec_norm_inf(work->z_prev, work->data->m);
}



c_float compute_dua_res(OSQPWorkspace * work, c_float * x, c_float * y){

    // NB: Use x_prev as temporary vector
    // NB: Only upper triangular part of P is stored.
    // dr = q + A'*y + P*x

    // dr = q
    prea_vec_copy(work->data->q, work->x_prev, work->data->n);

    // dr += P * x (upper triangular part)
    mat_vec(work->data->P, x, work->x_prev, 1);

    // dr += P' * x (lower triangular part with no diagonal)
    mat_tpose_vec(work->data->P, x, work->x_prev, 1, 1);

    // dr += A' * y
    if (work->data->m > 0)
        mat_tpose_vec(work->data->A, y, work->x_prev, 1, 0);

    // If scaling active -> rescale residual
    if (work->settings->scaling && !work->settings->scaled_termination){
        vec_ew_prod(work->scaling->Dinv, work->x_prev, work->x_prev, work->data->n);
    }

    return vec_norm_inf(work->x_prev, work->data->n);
}


c_int is_primal_infeasible(OSQPWorkspace * work){

    // This function checks for the primal infeasibility termination criteria.
    //
    // 1) A' * delta_y < eps * ||delta_y||
    //
    // 2) u'*max(delta_y, 0) + l'*min(delta_y, 0) < -eps * ||delta_y||
    //

    c_int i; // Index for loops
    c_float norm_delta_y;
    c_float ineq_lhs;
    c_float eps_prim_inf;

    eps_prim_inf = work->settings->eps_prim_inf;

    // Compute infinity norm of delta_y
    if (work->settings->scaling && !work->settings->scaled_termination){ // Unscale if necessary
        // Use work->Adelta_x as temporary vector
        vec_ew_prod(work->scaling->E, work->delta_y, work->Adelta_x, work->data->m);
        norm_delta_y = vec_norm_inf(work->Adelta_x, work->data->m);
    } else{
        norm_delta_y = vec_norm_inf(work->delta_y, work->data->m);
    }

    if (norm_delta_y > eps_prim_inf){ // ||delta_y|| > 0

        ineq_lhs = 0;
        for (i = 0; i < work->data->m; i++){
            ineq_lhs += work->data->u[i] * c_max(work->delta_y[i], 0) + \
                        work->data->l[i] * c_min(work->delta_y[i], 0);
        }

        // Check if the condition is satisfied: ineq_lhs < -eps
        if (ineq_lhs < -eps_prim_inf * norm_delta_y ){
            // Compute and return ||A'delta_y|| < eps_prim_inf
            mat_tpose_vec(work->data->A, work->delta_y, work->Atdelta_y, 0, 0);
            if (work->settings->scaling && !work->settings->scaled_termination){ // Unscale if necessary
                vec_ew_prod(work->scaling->Dinv, work->Atdelta_y, work->Atdelta_y, work->data->n);
            }
            return vec_norm_inf(work->Atdelta_y, work->data->n) < eps_prim_inf * norm_delta_y;
        }

    }

    // Conditions not satisfied -> not primal infeasible
    return 0;

}


c_int is_dual_infeasible(OSQPWorkspace * work){
    // This function checks for the scaled dual infeasibility termination criteria.
    //
    // 1) q * delta_x < - eps * || delta_x ||
    //
    // 2) ||P * delta_x || < eps * || delta_x ||
    //
    // 3) -> (A * delta_x)_i > -eps * || delta_x ||,    l_i != -inf
    //    -> (A * delta_x)_i <  eps * || delta_x ||,    u_i != inf
    //


    c_int i; // Index for loops
    c_float norm_delta_x;
    c_float eps_dual_inf;


    eps_dual_inf = work->settings->eps_dual_inf;

    // Compute norm of delta_x
    if (work->settings->scaling && !work->settings->scaled_termination){ // Unscale if necessary
        // Use work->Atdelta_y as temporary vector
        vec_ew_prod(work->scaling->D, work->delta_x, work->Atdelta_y, work->data->n);
        norm_delta_x = vec_norm_inf(work->Atdelta_y, work->data->n);
    } else{
        norm_delta_x = vec_norm_inf(work->delta_x, work->data->n);
    }

    // Prevent 0 division || delta_x || > 0
    if (norm_delta_x > eps_dual_inf){

        // Normalize delta_x by its norm
        /* vec_mult_scalar(work->delta_x, 1./norm_delta_x, work->data->n); */

        // Check first if q'*delta_x < 0
        if (vec_prod(work->data->q, work->delta_x, work->data->n) < -eps_dual_inf * norm_delta_x){

            // Compute product P * delta_x
            mat_vec(work->data->P, work->delta_x, work->Pdelta_x, 0);

            // Scale if necessary
            if (work->settings->scaling && !work->settings->scaled_termination){
                vec_ew_prod(work->scaling->Dinv, work->Pdelta_x, work->Pdelta_x, work->data->n);
            }

            // Check if || P * delta_x || = 0
            if (vec_norm_inf(work->Pdelta_x, work->data->n) < eps_dual_inf * norm_delta_x){

                // Compute A * delta_x
                mat_vec(work->data->A, work->delta_x, work->Adelta_x, 0);

                // Scale if necessary
                if (work->settings->scaling && !work->settings->scaled_termination){
                    vec_ew_prod(work->scaling->Einv, work->Adelta_x, work->Adelta_x, work->data->m);
                }

                // De Morgan Law Applied to dual infeasibility conditions for A * x
                // N.B. Note that 1e-06 is used to adjust the infinity value
                //      in case the problem is scaled.
                for (i = 0; i < work->data->m; i++){
                    if (((work->data->u[i] < OSQP_INFTY*1e-06) && (work->Adelta_x[i] >  eps_dual_inf * norm_delta_x)) ||
                    ((work->data->l[i] > -OSQP_INFTY*1e-06) && (work->Adelta_x[i] < -eps_dual_inf * norm_delta_x))){
                        // At least one condition not satisfied -> not dual infeasible
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


void update_info(OSQPWorkspace *work, c_int iter, c_int compute_objective, c_int polish){
    c_float * x, * z, * y;  // Allocate pointers to variables
    c_float * obj_val, * pri_res, *dua_res;  // objective value, residuals

    #ifdef PROFILING
    c_float *run_time;  // Execution time
    #endif

    #ifndef EMBEDDED
    if (polish){
        x = work->pol->x;
        y = work->pol->y;
        z = work->pol->z;
        obj_val = &work->pol->obj_val;
        pri_res = &work->pol->pri_res;
        dua_res = &work->pol->dua_res;
        #ifdef PROFILING
        run_time = &work->info->polish_time;
        #endif
    } else {
    #endif // EMBEDDED
        x = work->x;
        y = work->y;
        z = work->z;
        obj_val = &work->info->obj_val;
        pri_res = &work->info->pri_res;
        dua_res = &work->info->dua_res;
        work->info->iter = iter; // Update iteration number
        #ifdef PROFILING
        run_time = &work->info->solve_time;
        #endif
    #ifndef EMBEDDED
    }
    #endif


    // Compute the objective if needed
    if (compute_objective){
        *obj_val = compute_obj_val(work->data, x);
    }

    // Compute primal residual
    if (work->data->m == 0) {
        // No constraints -> Always primal feasible
        *pri_res = 0.;
    } else {
        *pri_res = compute_pri_res(work, x, z);
    }

    // Compute dual residual
    *dua_res = compute_dua_res(work, x, y);

    // Update timing
    #ifdef PROFILING
    *run_time = toc(work->timer);
    #endif

    #ifdef PRINTING
    work->summary_printed = 0;  // The just updated info have not been printed
    #endif
}


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



c_int check_termination(OSQPWorkspace *work){
    c_float eps_prim, eps_dual;
    c_int exitflag;
    c_int prim_res_check, dual_res_check, prim_inf_check, dual_inf_check;
    c_float eps_abs, eps_rel;
    c_float max_rel_eps, temp_rel_eps; // Temporary variables to compute maximums


    // Initialize variables to 0
    exitflag = 0;
    prim_res_check = 0; dual_res_check = 0;
    prim_inf_check = 0; dual_inf_check = 0;

    // Initialize tolerances
    eps_abs = work->settings->eps_abs;
    eps_rel = work->settings->eps_rel;

    // Check residuals
    if (work->data->m == 0){
        prim_res_check = 1;  // No contraints -> Primal feasibility always satisfied
    }
    else {
        // Compute primal tolerance

        // max_rel_eps = max(||z||, ||A x||)
        if (work->settings->scaling && !work->settings->scaled_termination){
            // ||Einv * z||
            vec_ew_prod(work->scaling->Einv, work->z, work->z_prev, work->data->m);
            max_rel_eps = vec_norm_inf(work->z_prev, work->data->m);
            // ||Einv * A * x||
            mat_vec(work->data->A, work->x, work->z_prev, 0);
            vec_ew_prod(work->scaling->Einv, work->z_prev, work->z_prev, work->data->m);
            temp_rel_eps = vec_norm_inf(work->z_prev, work->data->m);
            // Choose maximum
            if (temp_rel_eps > max_rel_eps) max_rel_eps = temp_rel_eps;
        } else { // No unscaling required
            // ||z||
            max_rel_eps = vec_norm_inf(work->z, work->data->m);
            // ||A * x||
            mat_vec(work->data->A, work->x, work->z_prev, 0);
            temp_rel_eps = vec_norm_inf(work->z_prev, work->data->m);
            // Choose maximum
            if (temp_rel_eps > max_rel_eps) max_rel_eps = temp_rel_eps;
        }


        // eps_prim
        eps_prim = eps_abs + eps_rel * max_rel_eps;

        // Primal feasibility check
        if (work->info->pri_res < eps_prim) {
            prim_res_check = 1;
        } else {
            // Primal infeasibility check
            prim_inf_check = is_primal_infeasible(work);
        }
    }  // End check if m == 0

    // Compute dual tolerance
    // max_rel_eps = max(||q||, ||A' y|, ||P x||)
    if (work->settings->scaling && !work->settings->scaled_termination){
        // || Dinv q||
        vec_ew_prod(work->scaling->Dinv, work->data->q, work->x_prev, work->data->n);
        max_rel_eps = vec_norm_inf(work->x_prev, work->data->n);
        // || Dinv A' y ||
        mat_tpose_vec(work->data->A, work->y, work->x_prev, 0, 0);
        vec_ew_prod(work->scaling->Dinv, work->x_prev, work->x_prev, work->data->n);
        temp_rel_eps = vec_norm_inf(work->x_prev, work->data->n);
        if (temp_rel_eps > max_rel_eps) max_rel_eps = temp_rel_eps;
        // || Dinv P x||
        // P * x (upper triangular part)
        mat_vec(work->data->P, work->x, work->x_prev, 0);
        // P' * x (lower triangular part with no diagonal)
        mat_tpose_vec(work->data->P, work->x, work->x_prev, 1, 1);
        vec_ew_prod(work->scaling->Dinv, work->x_prev, work->x_prev, work->data->n);
        temp_rel_eps = vec_norm_inf(work->x_prev, work->data->n);
        if (temp_rel_eps > max_rel_eps) max_rel_eps = temp_rel_eps;

    } else { // No scaling required
        // ||q||
        max_rel_eps = vec_norm_inf(work->data->q, work->data->n);
        mat_tpose_vec(work->data->A, work->y, work->x_prev, 0, 0);
        // ||A'*y||
        temp_rel_eps = vec_norm_inf(work->x_prev, work->data->n);
        if (temp_rel_eps > max_rel_eps) max_rel_eps = temp_rel_eps;
        // ||P*x||
        // P * x (upper triangular part)
        mat_vec(work->data->P, work->x, work->x_prev, 0);
        // P' * x (lower triangular part with no diagonal)
        mat_tpose_vec(work->data->P, work->x, work->x_prev, 1, 1);
        temp_rel_eps = vec_norm_inf(work->x_prev, work->data->n);
        if (temp_rel_eps > max_rel_eps) max_rel_eps = temp_rel_eps;

    }

    // eps_dual
    eps_dual = eps_abs + eps_rel * max_rel_eps;

    // Dual feasibility check
    if (work->info->dua_res < eps_dual) {
        dual_res_check = 1;
    } else {
        // Check dual infeasibility
        dual_inf_check = is_dual_infeasible(work);
    }

    // Compare checks to determine solver status
    if (prim_res_check && dual_res_check){
        // Update final information
        update_status(work->info, OSQP_SOLVED);
        exitflag = 1;
    }
    else if (prim_inf_check){
        // Update final information
        update_status(work->info, OSQP_PRIMAL_INFEASIBLE);
        if (work->settings->scaling && !work->settings->scaled_termination){
            // Update infeasibility certificate
            vec_ew_prod(work->scaling->E, work->delta_y, work->delta_y, work->data->m);
        }
        work->info->obj_val = OSQP_INFTY;
        exitflag = 1;
    }
    else if (dual_inf_check){
        // Update final information
        update_status(work->info, OSQP_DUAL_INFEASIBLE);
        if (work->settings->scaling && !work->settings->scaled_termination){
            // Update infeasibility certificate
            vec_ew_prod(work->scaling->D, work->delta_x, work->delta_x, work->data->n);
        }
        work->info->obj_val = -OSQP_INFTY;
        exitflag = 1;
    }

    return exitflag;

}


#ifndef EMBEDDED


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


c_int validate_linsys_solver(c_int linsys_solver){
    if (linsys_solver != SUITESPARSE_LDL){
        return 1;
    }
    // TODO: Add more solvers in case

    // Valid solver
    return 0;
}


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
    if (settings->scaling_iter < 1) {
        #ifdef PRINTING
        c_print("scaling_iter must be greater than 0\n");
        #endif
        return 1;
    }
    if (settings->scaling_norm != 1 && settings->scaling_norm != 2 &&
        settings->scaling_norm != -1) {
        #ifdef PRINTING
        c_print("scaling_norm must be wither 1, 2 or -1 (for infinity norm)\n");
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
    if (validate_linsys_solver(settings->linsys_solver)){
        #ifdef PRINTING
        c_print("linsys_solver not recognized\n");
        #endif
        return 1;
    }
    if (settings->verbose != 0 && settings->verbose != 1) {
        #ifdef PRINTING
        c_print("verbose must be either 0 or 1\n");
        #endif
        return 1;
    }
    if (settings->scaled_termination != 0 && settings->scaled_termination != 1) {
        #ifdef PRINTING
        c_print("scaled_termination must be either 0 or 1\n");
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
