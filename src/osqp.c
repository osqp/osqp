#include "osqp.h"
#include "util.h"


/*******************************
 * Secondary functions needed  *
 *******************************/

/**
 * Cold start workspace variables
 * @param work Workspace
 */
static void cold_start(Work *work) {
    memset(work->x, 0, work->data->n * sizeof(c_float));
    memset(work->z, 0, work->data->n * sizeof(c_float));
    memset(work->u, 0, work->data->n * sizeof(c_float));
}


/**
 * Update RHS during first tep of ADMM iteration
 * @param  work Workspace
 * @return      RHS
 */
void compute_rhs(Work *work){
    c_int i; // Index
    for (i=0; i < work->data->n; i++){
        // Cycle over part related to original x variables
        work->rhs[i] = work->settings->rho * (work->z[i] - work->u[i])
                       - work->data->q[i];
    }
    for (i = work->data->n; i < work->data->n + work->data->m; i++){
        // Cycle over dual variable within first step (nu)
        work->rhs[i] = work->z[i] - work->u[i];
    }
}


/**
 * Update x variable after solving linear system (first ADMM step)
 * @param work Workspace
 */
void update_x(Work *work){
    c_int i; // Index
    for (i = 0; i < work->data->n; i++){ // Update x directly from RHS
        work->x[i] = work->rhs[i];
    }
    for (i = work->data->n; i < work->data->n + work->data->m; i++){
        work->x[i] = 1./work->settings->rho * work->rhs[i] + work->z[i] - work->u[i];
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
        // Part related to original x variables
        work->z[i] = c_min(c_max(work->settings->alpha * work->x[i] +
                     (1.0 - work->settings->alpha) * work->z_prev[i] +
                     work->u[i], work->data->lx[i]), work->data->ux[i]);
    }

    for (i = work->data->n; i < work->data->n + work->data->m; i++){
        // Part related to slack variables
        work->z[i] = c_min(c_max(work->settings->alpha * work->x[i] +
                     (1.0 - work->settings->alpha) * work->z_prev[i] +
                     work->u[i], work->data->lA[i]), work->data->uA[i]);
    }


}

/**
 * Update u variable
 * @param work Workspace
 */
void update_u(Work *work){
    c_int i; // Index
    for (i = 0; i < work->data->n + work->data->m; i++){
        work->u[i] += work->settings->alpha * work->x[i] +
                      (1.0 - work->settings->alpha) * work->z_prev[i] -
                      work->z[i];
    }

}

c_float compute_obj_val(Data * data, c_float * x){
    c_float obj_val = 0;
    obj_val = quad_form(data->P, x) + 

}

/**
 * Update solver information
 * @param work Workspace
 */
void update_info(Work *work, c_int iter){
    work->info->iter = iter; // Update iteration number
    work->obj_val =
    //TODO: COntinue from here. N.B. STATUS and STATUS VAL SET AT SETUP TIME
    // FORGET ABOUT TIMERS for now
}


/**********************
 * Main API Functions *
 **********************/


/**
 * Initialize OSQP solver allocating memory.
 *
 * It also sets the linear system solver:
 * - direct solver: KKT matrix factorization is performed here
 *
 *
 * N.B. This is the only function that allocates dynamic memory. During code
 * generation it is going to be removed.
 *
 * @param  data         Problem data
 * @param  settings     Solver settings
 * @return              Solver workspace
 */
Work * osqp_setup(const Data * data, Settings *settings){
    Work * work; // Workspace

    //TODO: Add timing in setup phase

    // Allocate empty workspace
    work = c_calloc(1, sizeof(Work));
    if (!work){
        c_print("ERROR: allocating work failure!\n");
    }

    // TODO: Add validation for problem data

    // Copy problem data into workspace
    work->data = c_malloc(sizeof(Data));
    work->data->n = data->n;    // Number of variables
    work->data->m = data->m;    // Number of linear constraints
    work->data->P = csc_to_triu(data->P);         // Cost function matrix
    work->data->q = vec_copy(data->q, data->n);    // Linear part of cost function
    work->data->A = copy_csc_mat(data->A);         // Linear constraints matrix
    work->data->lA = vec_copy(data->lA, data->m);  // Lower bounds on constraints
    work->data->uA = vec_copy(data->uA, data->m);  // Upper bounds on constraints
    work->data->lx = vec_copy(data->lx, data->n);  // lower bounds on variables
    work->data->ux = vec_copy(data->ux, data->n);  // upper bounds on variables

    /* Allocate internal solver variables (ADMM steps)
     *
     * N.B. Augmented variables with slacks (n+m)
     */
    work->x = c_malloc((work->data->n + work->data->m) * sizeof(c_float));
    work->z = c_malloc((work->data->n + work->data->m) * sizeof(c_float));
    work->u = c_malloc((work->data->n + work->data->m) * sizeof(c_float));
    work->rhs = c_malloc((work->data->n + work->data->m) * sizeof(c_float));
    work->z_prev = c_malloc((work->data->n + work->data->m) * sizeof(c_float));

    // TODO: Add Validaiton for settings
    // Copy settings
    work->settings = copy_settings(settings);

    // Initialize linear system solver private structure
    work->priv = init_priv(work->data->P, work->data->A, work->settings);

    // Allocate scaling
    if (settings->normalize){
        //TODO: Add normalization (now doing nothing)
        work->scaling = OSQP_NULL;
    }
    else {
        work->scaling = OSQP_NULL;
    }

    // Allocate solution
    work->solution = c_calloc(1, sizeof(Solution));
    work->solution->x = c_calloc(1, work->data->n * sizeof(c_float)); // Allocate primal solution
    work->solution->u = c_calloc(1, work->data->m * sizeof(c_float)); // Allocate dual solution


    // Allocate information
    work->info = c_calloc(1, sizeof(Info));


    // Print header
    if (work->settings->verbose) print_setup_header(work->data, settings);

    return work;
}





/**
 * Solve Quadratic Program
 *
 * Main ADMM iteration.
 * Iteration variables are the usual ADMM ones: x, z, u
 *
 * @param  work Workspace allocated
 * @return      Exitflag for errors
 */
c_int osqp_solve(Work * work){
    c_int exitflag = 0;
    c_int iter;

    if (work->settings->verbose){
        // Print Header for every column
        print_header();
    }

    // Initialize variables (cold start or warm start depending on settings)
    // TODO: Add proper warmstart
    cold_start(work);

    // Main ADMM algorithm
    for (iter = 0; iter < work->settings->max_iter; iter ++ ){
        // Update z_prev (preallocated, no malloc)
        prea_vec_copy(work->z, work->z_prev, work->data->n + work->data->m);


        /* ADMM STEPS */
        /* First step: x_{k+1} */
        compute_rhs(work);
        solve_lin_sys(work->settings, work->priv, work->rhs);
        // TODO: Can't we store solution directly in first part of x?
        // Then there would be no need to run update_x copying x again
        update_x(work);

        /* Second step: z_{k+1} */
        project_x(work);

        /* Third step: u_{k+1} */
        update_u(work);

        /* End of ADMM Steps */


        /* Update information */
        update_info(work, iter);

        /* Print summary */
        print_summary(work->info);


    }



    return exitflag;
}




/**
 * Cleanup workspace
 * @param  work Workspace
 * @return      Exitflag for errors
 */
c_int osqp_cleanup(Work * work){
    c_int exitflag=0;

    // TODO: Add checks for proper cleaning!

    // Free Data
    csc_spfree(work->data->P);
    c_free(work->data->q);
    csc_spfree(work->data->A);
    c_free(work->data->lA);
    c_free(work->data->uA);
    c_free(work->data->lx);
    c_free(work->data->ux);
    c_free(work->data);

    // Free work Variables
    c_free(work->x);
    c_free(work->u);
    c_free(work->z);

    // Free Settings
    c_free(work->settings);

    // Free solution
    c_free(work->solution->x);
    c_free(work->solution->u);
    c_free(work->solution);

    // Free information
    c_free(work->info);

    return exitflag;
}
