#include "polish.h"

/**
 * Form reduced matrix A that contains only rows that are active at the solution.
 * The set of active constraints is guessed from the primal and dual solution
 * returned by the ADMM.
 * @param  work Workspace
 * @return      Number of rows in Ared
 */
static c_int form_Ared(Work *work) {
    c_int j, ptr, mred=0, Ared_nnz=0;

    // Initialize counters for active/inactive constraints
    work->pol->n_lAct = 0;
    work->pol->n_uAct = 0;

    /* Guess which linear constraints are lower-active, upper-active and free
     *    A2Ared[j] = -1    (if j-th row of A is not inserted in Ared)
     *    A2Ared[j] =  i    (if j-th row of A is inserted at i-th row of Ared)
     */
    for (j = 0; j < work->data->m; j++) {
        if ( work->z[j] - work->data->l[j] < - work->y[j] ) {     // lower-active
                work->pol->ind_lAct[work->pol->n_lAct++] = j;
                work->pol->A2Ared[j] = mred++;
        }
        else if ( work->data->u[j] - work->z[j] < work->y[j] ) {  // upper-active
                    work->pol->ind_uAct[work->pol->n_uAct++] = j;
                    work->pol->A2Ared[j] = mred++;
        }
        else {
            work->pol->A2Ared[j] = -1;                          // free
        }
    }

    // TODO: Add check if mred == 0

    // Count number of elements in Ared
    for (j = 0; j < work->data->A->nzmax; j++) {
        if (work->pol->A2Ared[work->data->A->i[j]] != -1)
            Ared_nnz++;
    }

    // Form Ared
    work->pol->Ared = csc_spalloc(mred, work->data->n, Ared_nnz, 1, 0);
    Ared_nnz = 0;
    for (j = 0; j < work->data->n; j++) {  // Cycle over columns of A
        work->pol->Ared->p[j] = Ared_nnz;
        for (ptr = work->data->A->p[j]; ptr < work->data->A->p[j + 1]; ptr++) {
            if (work->pol->A2Ared[work->data->A->i[ptr]] != -1) {
                // If row of A should be added to Ared
                work->pol->Ared->i[Ared_nnz] = work->pol->A2Ared[work->data->A->i[ptr]];
                work->pol->Ared->x[Ared_nnz++] = work->data->A->x[ptr];
            }
        }
    }
    // Update the last element in Ared->p
    work->pol->Ared->p[work->data->n] = Ared_nnz;

    // Return number of rows in Ared
    return mred;
}

/**
 * Form reduced right-hand side rhs_red
 * @param  work Workspace
 * @param  mred number of active constraints
 * @return      reduced rhs
 */
static c_float * form_rhs_red(Work * work, c_int mred){
    c_int j;

    // Form the rhs of the reduced KKT linear system
    c_float *rhs = c_malloc(sizeof(c_float) * (work->data->n + mred));
    for (j = 0; j < work->data->n; j++) {
        rhs[j] = -work->data->q[j];
    }
    for (j = 0; j < work->pol->n_lAct; j++) {
        rhs[work->data->n + j] = work->data->l[work->pol->ind_lAct[j]];
    }
    for (j = 0; j < work->pol->n_uAct; j++) {
        rhs[work->data->n + work->pol->n_lAct + j] =
            work->data->u[work->pol->ind_uAct[j]];
    }

    return rhs;

}

/**
 * Perform iterative refinement on the polished solution:
 *    (repeat)
 *    1. (K + dK) * dz = b - K*z
 *    2. z <- z + dz
 * @param  work Solver workspace
 * @param  p    Private variable for solving linear system
 * @param  b    RHS of the linear system
 * @return      More accurate solution
 */
static void iterative_refinement(Work *work, Priv *p, c_float *z, c_float *b) {
    if (work->settings->pol_refine_iter > 0) {
        c_int i, j, n;
        n = work->data->n + work->pol->Ared->m;
        c_float *dz = c_malloc(sizeof(c_float) * n);
        c_float *rhs = c_malloc(sizeof(c_float) * n);

        for (i=0; i<work->settings->pol_refine_iter; i++) {

            // Form the RHS for the iterative refinement:  b - K*z
            prea_vec_copy(b, rhs, n);

            // Upper Part: R^{n}
            // -= Px (upper triang)
            mat_vec(work->data->P, z, rhs, -1);

            // -= Px (lower triang)
            mat_tpose_vec(work->data->P, z, rhs, -1, 1);

            // -= Ared'*y_red
            mat_tpose_vec(work->pol->Ared, z + work->data->n, rhs, -1, 0);

            // Lower Part: R^{m}
            mat_vec(work->pol->Ared, z, rhs + work->data->n, -1);

            // Solve linear system. Store solution in rhs
            solve_lin_sys(work->settings, p, rhs);

            // Update solution
            for (j=0; j<n; j++) {
                z[j] += rhs[j];
            }
        }
        c_free(dz);
        c_free(rhs);
    }
}


/**
 * Compute dual variable y from reduced on y_red
 * @param work Workspace
 */
static void compute_y_from_y_red(Work * work){
    c_int j;
    for (j = 0; j < work->data->m; j++) {
        if (work->pol->A2Ared[j] != -1) {
            work->y[j] = work->pol->y_red[work->pol->A2Ared[j]];
        } else {
            work->y[j] = 0.0;
        }
    }
}

// Solution polishing: Solve equality constrained QP with assumed active constr.
c_int polish(Work *work) {
    c_int mred, polish_successful;
    c_float * rhs_red;
    Priv *plsh;

    #ifdef PROFILING
    tic(work->timer); // Start timer
    #endif

    // Form Ared by assuming the active constraints and store in work->pol->Ared
    mred = form_Ared(work);

    if (mred > 0) {  // There are active constraints -> Do polishing
        // Form and factorize reduced KKT
        plsh = init_priv(work->data->P, work->pol->Ared, work->settings, 1);

        // Form reduced right-hand side rhs_red
        rhs_red = form_rhs_red(work, mred);

        // Solve the reduced KKT system
        c_float *pol_sol = vec_copy(rhs_red, work->data->n + mred);
        solve_lin_sys(work->settings, plsh, pol_sol);

        // Perform iterative refinement to compensate for the regularization error
        iterative_refinement(work, plsh, pol_sol, rhs_red);

        // Store the polished solution
        work->pol->y_red = c_malloc(mred * sizeof(c_float));
        prea_vec_copy(pol_sol, work->pol->x, work->data->n);
        prea_vec_copy(pol_sol + work->data->n, work->pol->y_red, mred);

        // Compute z = A*x needed for computing the primal residual
        mat_vec(work->data->A, work->pol->x, work->pol->z, 0);

        // Compute primal and dual residuals at the polished solution
        update_info(work, 0, 1);

        // Check if polishing was successful
        polish_successful = (work->pol->pri_res < work->info->pri_res &&
            work->pol->dua_res < work->info->dua_res) || // Residuals are reduced
            (work->pol->pri_res < work->info->pri_res &&
             work->info->dua_res < 1e-10) ||             // Dual residual already tiny
            (work->pol->dua_res < work->info->dua_res &&
             work->info->pri_res < 1e-10);               // Primal residual already tiny


        if (polish_successful) {

                // Update solver information
                work->info->obj_val = work->pol->obj_val;
                work->info->pri_res = work->pol->pri_res;
                work->info->dua_res = work->pol->dua_res;
                work->info->status_polish = 1;

                // Update (x, z, y) in ADMM iterations

                // Update x
                prea_vec_copy(work->pol->x, work->x, work->data->n);

                // Update z
                prea_vec_copy(work->pol->z, work->z, work->data->m);

                // Reconstruct y from y_red and active constraints
                compute_y_from_y_red(work);

                // Print summary
                #ifdef PRINTING
                if (work->settings->verbose)
                    print_polishing(work->info);
                #endif

        } else { // Polishing failed
            work->info->status_polish = -1;
            // TODO: Try to find a better solution on the line connecting ADMM
            //       and polished solution
        }

        /* Update timing */
        #ifdef PROFILING
        work->info->polish_time = toc(work->timer);
        #endif

        // Memory clean-up
        free_priv(plsh);
        csc_spfree(work->pol->Ared);
        c_free(work->pol->y_red);
        c_free(rhs_red);
        c_free(pol_sol);
    }

    return 0;
}
