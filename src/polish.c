#include "polish.h"

/**
 * Form reduced matrix A that contains only rows that are active at the solution.
 * The set of active constraints is guessed from the primal and dual solution
 * returned by the ADMM.
 * @param  work Workspace
 * @return      Number of rows in Ared
 */
c_int form_Ared(Work *work) {
    c_int j, ptr, mred=0, Ared_nnz=0;

    // Initialize counters for active/inactive constraints
    work->pol->n_lAct = 0;
    work->pol->n_uAct = 0;

    /* Guess which linear constraints are lower-active, upper-active and free
     *    A2Ared[j] = -1    (if j-th row of A is not inserted in Ared)
     *    A2Ared[j] =  i    (if j-th row of A is inserted at i-th row of Ared)
     */
    for (j = 0; j < work->data->m; j++) {
        if ( work->z[work->data->n + j] - work->data->lA[j] <
             -work->settings->rho * work->u[j] ) {              // lower-active
                work->pol->ind_lAct[work->pol->n_lAct++] = j;
                work->pol->A2Ared[j] = mred++;
        }
        else if ( work->data->uA[j] - work->z[work->data->n + j] <
                  work->settings->rho * work->u[j] ) {          // upper-active
                    work->pol->ind_uAct[work->pol->n_uAct++] = j;
                    work->pol->A2Ared[j] = mred++;
        }
        else {
            work->pol->A2Ared[j] = -1;                          // free
        }
    }

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
 * Perform iterative refinement on the polished solution:
 *    (repeat)
 *    1. (K + dK) * dz = b - K*z
 *    2. z <- z + dz
 * @param  work Solver workspace
 * @param  p    Private variable for solving linear system
 * @param  b    RHS of the linear system
 * @return      More accurate solution
 */
void iterative_refinement(Work *work, Priv *p, c_float *z, c_float *b) {
    if (work->settings->pol_refine_iter > 0) {
        c_int i, j, n;
        n = work->data->n + work->pol->Ared->m;
        c_float *dz = c_malloc(sizeof(c_float) * n);
        c_float *rhs = c_malloc(sizeof(c_float) * n);

        for (i=0; i<work->settings->pol_refine_iter; i++) {
            // Form the RHS for the iterative refinement:  b - K*z
            prea_vec_copy(b, rhs, n);
            mat_vec(work->data->P, z, rhs, -1);          // -= Px (upper triang)
            mat_tpose_vec(work->data->P, z, rhs, -1, 1); // -= Px (lower triang)
            mat_tpose_vec(work->pol->Ared, z + work->data->n,
                          rhs, -1, 0);                   // -= Ared'*lambda_red
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


// Solution polishing: Solve equality constrained QP with assumed active constr.
c_int polish(Work *work) {
    c_int j, mred;
    Priv *plsh;

    #if PROFILING > 0
    tic(work->timer); // Start timer
    #endif

    // Form Ared by assuming the active constraints and store in work->pol->Ared
    mred = form_Ared(work);

    // Form and factorize reduced KKT
    plsh = init_priv(work->data->P, work->pol->Ared, work->settings, 1);

    // Form the rhs of the reduced KKT linear system
    c_float *rhs = c_malloc(sizeof(c_float) * (work->data->n + mred));
    for (j = 0; j < work->data->n; j++) {
        rhs[j] = -work->data->q[j];
    }
    for (j = 0; j < work->pol->n_lAct; j++) {
        rhs[work->data->n + j] = work->data->lA[work->pol->ind_lAct[j]];
    }
    for (j = 0; j < work->pol->n_uAct; j++) {
        rhs[work->data->n + work->pol->n_lAct + j] =
            work->data->uA[work->pol->ind_uAct[j]];
    }

    // Solve the reduced KKT system
    c_float *pol_sol = vec_copy(rhs, work->data->n + mred);
    solve_lin_sys(work->settings, plsh, pol_sol);

    // Perform iterative refinement to compensate for the regularization error
    iterative_refinement(work, plsh, pol_sol, rhs);

    // Store the polished solution
    work->pol->lambda_red = c_malloc(mred * sizeof(c_float));
    prea_vec_copy(pol_sol, work->pol->x, work->data->n);
    prea_vec_copy(pol_sol + work->data->n, work->pol->lambda_red, mred);

    // Compute A*x needed for computing the primal residual
    mat_vec(work->data->A, work->pol->x, work->pol->Ax, 0);

    // Compute primal and dual residuals at the polished solution
    update_info(work, 0, 1);

    // Check if polishing was successful
    if ((work->pol->pri_res < work->info->pri_res &&
         work->pol->dua_res < work->info->dua_res) ||
        (work->pol->pri_res < work->info->pri_res &&
         work->info->dua_res < 1e-10) ||          // dual residual is tiny
        (work->pol->dua_res < work->info->dua_res &&
         work->info->pri_res < 1e-10)) {          // primal residual is tiny
            // Update solver information
            work->info->obj_val = work->pol->obj_val;
            work->info->pri_res = work->pol->pri_res;
            work->info->dua_res = work->pol->dua_res;
            work->info->status_polish = 1;
            // Update primal and dual variables
            prea_vec_copy(work->pol->x, work->solution->x, work->data->n);
            for (j = 0; j < work->data->m; j++) {
                if (work->pol->A2Ared[j] != -1) {
                    work->solution->lambda[j] = work->pol->lambda_red[work->pol->A2Ared[j]];
                } else {
                    work->solution->lambda[j] = 0.0;
                }
            }
            // Print summary
            #if PRINTLEVEL > 1
            if (work->settings->verbose)
                print_polishing(work->info);
            #endif
    } else {
        // Polishing failed
        work->info->status_polish = 0;
        // TODO: Try to find a better solution on the line connecting ADMM
        //       and polished solution
    }

    /* Update timing */
    #if PROFILING > 0
    work->info->polish_time = toc(work->timer);
    #endif

    // Memory clean-up
    free_priv(plsh);
    csc_spfree(work->pol->Ared);
    c_free(work->pol->lambda_red);
    c_free(rhs);
    c_free(pol_sol);

    return 0;
}
