#include "auxil.h"
#include "util.h"
#include "osqp.h"

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

    // Validate data
    if (validate_data(data)){
        #ifdef PRINTING
        c_print("ERROR: Data validation returned failure!\n");
        #endif
        return OSQP_NULL;
    }

    // Validate settings
    if (validate_settings(settings)){
        #ifdef PRINTING
        c_print("ERROR: Settings validation returned failure!\n");
        #endif
        return OSQP_NULL;
    }

    // Allocate empty workspace
    work = c_calloc(1, sizeof(Work));
    if (!work){
        #ifdef PRINTING
        c_print("ERROR: allocating work failure!\n");
        #endif
        return OSQP_NULL;
    }

    // Start and allocate directly timer
    #ifdef PROFILING
    work->timer = c_malloc(sizeof(Timer));
    tic(work->timer);
    #endif


    // Copy problem data into workspace
    work->data = c_malloc(sizeof(Data));
    work->data->n = data->n;    // Number of variables
    work->data->m = data->m;    // Number of linear constraints
    work->data->P = csc_to_triu(data->P);         // Cost function matrix
    work->data->q = vec_copy(data->q, data->n);    // Linear part of cost function
    work->data->A = copy_csc_mat(data->A);         // Linear constraints matrix
    work->data->l = vec_copy(data->l, data->m);  // Lower bounds on constraints
    work->data->u = vec_copy(data->u, data->m);  // Upper bounds on constraints


    /*
     *  Allocate internal solver variables (ADMM steps)
     */
    work->x = c_calloc(work->data->n, sizeof(c_float));
    work->z = c_calloc(work->data->m, sizeof(c_float));
    work->xz_tilde = c_calloc((work->data->n + work->data->m), sizeof(c_float));
    work->x_prev = c_calloc(work->data->n, sizeof(c_float));
    work->z_prev = c_calloc(work->data->m, sizeof(c_float));
    work->y = c_calloc(work->data->m, sizeof(c_float));

    // Infeasibility variables
    work->delta_y = c_calloc(work->data->m, sizeof(c_float));
    work->Atdelta_y = c_calloc(work->data->n, sizeof(c_float));

    work->first_run = 1;

    // Copy settings
    work->settings = copy_settings(settings);
    if (work->data->m == 0) work->settings->polishing = 0;     // If no constraints->disable polishing

    // Perform scaling
    if (settings->scaling) {
        scale_data(work);
    }
    else {
        work->scaling = OSQP_NULL;
    }

    // Initialize linear system solver private structure
    work->priv = init_priv(work->data->P, work->data->A, work->settings, 0);

    // Initialize active constraints structure
    work->pol = c_malloc(sizeof(Polish));
    work->pol->ind_lAct = c_malloc(work->data->m * sizeof(c_int));
    work->pol->ind_uAct = c_malloc(work->data->m * sizeof(c_int));
    work->pol->A2Ared = c_malloc(work->data->m * sizeof(c_int));
    work->pol->x = c_malloc(work->data->n * sizeof(c_float));
    work->pol->z = c_malloc(work->data->m * sizeof(c_float));


    // Allocate solution
    work->solution = c_calloc(1, sizeof(Solution));
    work->solution->x = c_calloc(1, work->data->n * sizeof(c_float));
    work->solution->y = c_calloc(1, work->data->m * sizeof(c_float));

    // Allocate information
    work->info = c_calloc(1, sizeof(Info));
    work->info->status_val = OSQP_UNSOLVED;
    update_status_string(work->info);

    // Allocate timing information
    #ifdef PROFILING
    work->info->solve_time = 0.0;  // Solve time to zero
    work->info->polish_time = 0.0; // Polish time to zero
    work->info->run_time = 0.0;    // Total run time to zero
    work->info->setup_time = toc(work->timer); // Updater timer information
    #endif

    // Print header
    #ifdef PRINTING
    if (work->settings->verbose)
        print_setup_header(work->data, settings);
    #endif

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

    // Check if workspace has been initialized
    if (!work){
        #ifdef PRINTING
        c_print("ERROR: Workspace not initialized!\n");
        #endif
        return -1;
    }

    #ifdef PROFILING
    tic(work->timer); // Start timer
    #endif

    #ifdef PRINTING
    if (work->settings->verbose){
        // Print Header for every column
        print_header();
    }
    #endif

    // Initialize variables (cold start or warm start depending on settings)
    if (!work->settings->warm_start)
        cold_start(work);     // If not warm start -> set z, u to zero

    // Main ADMM algorithm
    for (iter = 1; iter <= work->settings->max_iter; iter ++) {
        // Update x_prev, z_prev (preallocated, no malloc)
        prea_vec_copy(work->x, work->x_prev, work->data->n);
        prea_vec_copy(work->z, work->z_prev, work->data->m);

        /* ADMM STEPS */
        /* Compute \tilde{x}^{k+1}, \tilde{z}^{k+1} */
        update_xz_tilde(work);

        /* Compute x^{k+1} */
        update_x(work);

        /* Compute z^{k+1} */
        update_z(work);

        /* Compute y^{k+1} */
        update_y(work);

        /* End of ADMM Steps */


        /* Update information */
        update_info(work, iter, 0);

        /* Print summary */
        #ifdef PRINTING
        if (work->settings->verbose && iter % PRINT_INTERVAL == 0)
            print_summary(work->info);
        #endif

        if (residuals_check(work)){
            // Terminate algorithm
            break;
        }

    }


    /* Print summary for last iteration */
    #ifdef PRINTING
    if (work->settings->verbose && iter % PRINT_INTERVAL != 0)
        print_summary(work->info);
    #endif

    /* if max iterations reached, change status accordingly */
    if (iter == work->settings->max_iter + 1){
        work->info->status_val = OSQP_MAX_ITER_REACHED;
    }

    /* Update final status */
    update_status_string(work->info);

    /* Update solve time */
    #ifdef PROFILING
    work->info->solve_time = toc(work->timer);
    #endif

    // Polish the obtained solution
    if (work->settings->polishing && work->info->status_val == OSQP_SOLVED)
        polish(work);

    /* Update total time */
    #ifdef PROFILING
    if (work->first_run == 0) {
        // total time: setup + solve + polish
        work->info->run_time = work->info->setup_time +
                               work->info->solve_time +
                               work->info->polish_time;
    } else {
        // total time: solve + polish
        work->info->run_time = work->info->solve_time +
                               work->info->polish_time;
    }
    #endif

    /* Print final footer */
    #ifdef PRINTING
    if(work->settings->verbose)
        print_footer(work->info, work->settings->polishing);
    #endif

    // Store solution
    store_solution(work);

    // Indicate that the solve function has already been executed
    if (work->first_run) work->first_run = 0;

    return exitflag;
}


/**
 * Cleanup workspace
 * @param  work Workspace
 * @return      Exitflag for errors
 */
c_int osqp_cleanup(Work * work){
    c_int exitflag=0;

    if (work) { // If workspace has been allocated
        // Free Data
        if (work->data) {
            if (work->data->P)
                csc_spfree(work->data->P);
            if (work->data->A)
                csc_spfree(work->data->A);
            if (work->data->q)
                c_free(work->data->q);
            if (work->data->l)
                c_free(work->data->l);
            if (work->data->u)
                c_free(work->data->u);
            c_free(work->data);
        }

        // Free scaling
        if (work->settings->scaling) {
            if (work->scaling->D)
                c_free(work->scaling->D);
            if (work->scaling->Dinv)
                c_free(work->scaling->Dinv);
            if (work->scaling->E)
                c_free(work->scaling->E);
            if (work->scaling->Einv)
                c_free(work->scaling->Einv);
            c_free(work->scaling);
        }

        // Free private structure for linear system solver_solution
        free_priv(work->priv);

        // Free active constraints structure
        if (work->pol) {
            if (work->pol->ind_lAct)
                c_free(work->pol->ind_lAct);
            if (work->pol->ind_uAct)
                c_free(work->pol->ind_uAct);
            if (work->pol->A2Ared)
                c_free(work->pol->A2Ared);
            if (work->pol->x)
                c_free(work->pol->x);
            if (work->pol->z)
                c_free(work->pol->z);
            c_free(work->pol);
        }

        // Free other Variables
        if (work->x)
            c_free(work->x);
        if (work->z)
            c_free(work->z);
        if (work->xz_tilde)
            c_free(work->xz_tilde);
        if (work->x_prev)
            c_free(work->x_prev);
        if (work->z_prev)
            c_free(work->z_prev);
        if (work->y)
            c_free(work->y);

        if (work->delta_y)
            c_free(work->delta_y);
        if (work->Atdelta_y)
            c_free(work->Atdelta_y);

        // Free Settings
        if (work->settings)
            c_free(work->settings);

        // Free solution
        if (work->solution) {
            if (work->solution->x)
                c_free(work->solution->x);
            if (work->solution->y)
                c_free(work->solution->y);
            c_free(work->solution);
        }

        // Free information
        if (work->info)
            c_free(work->info);

        // Free timer
        #ifdef PROFILING
        if (work->timer)
            c_free(work->timer);
        #endif

        // Free work
        c_free(work);
    }

    return exitflag;
}



/************************
 * Update problem data  *
 ************************/

/**
 * Update linear cost in the problem
 * @param  work  Workspace
 * @param  q_new New linear cost
 * @return       Exitflag for errors and warnings
 */
c_int osqp_update_lin_cost(Work * work, c_float * q_new) {

    // Replace q by the new vector
    prea_vec_copy(q_new, work->data->q, work->data->n);

    // Scaling
    if (work->settings->scaling) {
        vec_ew_prod(work->scaling->D, work->data->q, work->data->n);
    }

    return 0;
}

/**
 * Update lower and upper bounds in the problem constraints
 * @param  work   Workspace
 * @param  l_new New lower bound
 * @param  u_new New upper bound
 * @return        Exitflag: 1 if new lower bound is not <= than new upper bound
 */
c_int osqp_update_bounds(Work * work, c_float * l_new, c_float * u_new) {
    c_int i;

    // Check if lower bound is smaller than upper bound
    for (i=0; i<work->data->m; i++) {
        if (l_new[i] > u_new[i]) {
            #ifdef PRINTING
            c_print("lower bound must be lower than or equal to upper bound\n");
            #endif
            return 1;
        }
    }

    // Replace lA and uA by the new vectors
    prea_vec_copy(l_new, work->data->l, work->data->m);
    prea_vec_copy(u_new, work->data->u, work->data->m);

    // Scaling
    if (work->settings->scaling) {
        vec_ew_prod(work->scaling->E, work->data->l, work->data->m);
        vec_ew_prod(work->scaling->E, work->data->u, work->data->m);
    }

    return 0;
}

/**
 * Update lower bound in the problem constraints
 * @param  work   Workspace
 * @param  l_new New lower bound
 * @return        Exitflag: 1 if new lower bound is not <= than upper bound
 */
c_int osqp_update_lower_bound(Work * work, c_float * l_new) {
    c_int i;

    // Replace lA by the new vector
    prea_vec_copy(l_new, work->data->l, work->data->m);

    // Scaling
    if (work->settings->scaling) {
        vec_ew_prod(work->scaling->E, work->data->l, work->data->m);
    }

    // Check if lower bound is smaller than upper bound
    for (i=0; i<work->data->m; i++) {
        if (work->data->l[i] > work->data->u[i]) {
            #ifdef PRINTING
            c_print("upper bound must be greater than or equal to lower bound\n");
            #endif
            return 1;
        }
    }

    return 0;
}


/**
 * Update upper bound in the problem constraints
 * @param  work   Workspace
 * @param  u_new New upper bound
 * @return        Exitflag: 1 if new upper bound is not >= than lower bound
 */
c_int osqp_update_upper_bound(Work * work, c_float * u_new) {
    c_int i;

    // Replace uA by the new vector
    prea_vec_copy(u_new, work->data->u, work->data->m);

    // Scaling
    if (work->settings->scaling) {
        vec_ew_prod(work->scaling->E, work->data->u, work->data->m);
    }

    // Check if upper bound is greater than lower bound
    for (i=0; i<work->data->m; i++) {
        if (work->data->u[i] < work->data->l[i]) {
            #ifdef PRINTING
            c_print("lower bound must be lower than or equal to upper bound\n");
            #endif
            return 1;
        }
    }
    return 0;
}


/****************************
 * Update problem settings  *
 ****************************/


/**
 * Update max_iter setting
 * @param  work         Workspace
 * @param  max_iter_new New max_iter setting
 * @return              Exitflag
 */
c_int osqp_update_max_iter(Work * work, c_int max_iter_new) {
    // Check that max_iter is positive
    if (max_iter_new <= 0) {
        #ifdef PRINTING
        c_print("max_iter must be positive\n");
        #endif
        return 1;
    }
    // Update max_iter
    work->settings->max_iter = max_iter_new;

    return 0;
}

/**
 * Update absolute tolernace value
 * @param  work        Workspace
 * @param  eps_abs_new New absolute tolerance value
 * @return             Exitflag
 */
c_int osqp_update_eps_abs(Work * work, c_float eps_abs_new) {
    // Check that eps_abs is positive
    if (eps_abs_new <= 0.) {
        #ifdef PRINTING
        c_print("eps_abs must be positive\n");
        #endif
        return 1;
    }
    // Update eps_abs
    work->settings->eps_abs = eps_abs_new;

    return 0;
}

/**
 * Update relative tolernace value
 * @param  work        Workspace
 * @param  eps_rel_new New relative tolerance value
 * @return             Exitflag
 */
c_int osqp_update_eps_rel(Work * work, c_float eps_rel_new) {
    // Check that eps_rel is positive
    if (eps_rel_new <= 0.) {
        #ifdef PRINTING
        c_print("eps_rel must be positive\n");
        #endif
        return 1;
    }
    // Update eps_rel
    work->settings->eps_rel = eps_rel_new;

    return 0;
}

/**
 * Update relaxation parameter alpha
 * @param  work  Workspace
 * @param  alpha New relaxation parameter value
 * @return       Exitflag
 */
c_int osqp_update_alpha(Work * work, c_float alpha_new) {
    // Check that alpha is between 0 and 2
    if (alpha_new <= 0. || alpha_new >= 2.) {
        #ifdef PRINTING
        c_print("alpha must be between 0 and 2\n");
        #endif
        return 1;
    }
    // Update alpha
    work->settings->alpha = alpha_new;

    return 0;
}

/**
 * Update regularization parameter in polishing
 * @param  work      Workspace
 * @param  delta_new New regularization parameter
 * @return           Exitflag
 */
c_int osqp_update_delta(Work * work, c_float delta_new) {
    // Check that delta is positive
    if (delta_new <= 0.) {
        #ifdef PRINTING
        c_print("delta must be positive\n");
        #endif
        return 1;
    }
    // Update delta
    work->settings->delta = delta_new;

    return 0;
}

/**
 * Update polishing setting
 * @param  work          Workspace
 * @param  polishing_new New polishing setting
 * @return               Exitflag
 */
c_int osqp_update_polishing(Work * work, c_int polishing_new) {
    // Check that polishing is either 0 or 1
    if (polishing_new != 0 && polishing_new != 1) {
      #ifdef PRINTING
      c_print("polishing should be either 0 or 1\n");
      #endif
      return 1;
    }
    // Update polishing
    work->settings->polishing = polishing_new;

    #ifdef PROFILING
    // Reset polish time to zero
    work->info->polish_time = 0.0;
    #endif

    return 0;
}

/**
 * Update number of iterative refinement steps in polishing
 * @param  work                Workspace
 * @param  pol_refine_iter_new New iterative reginement steps
 * @return                     Exitflag
 */
c_int osqp_update_pol_refine_iter(Work * work, c_int pol_refine_iter_new) {
    // Check that pol_refine_iter is nonnegative
    if (pol_refine_iter_new < 0) {
        #ifdef PRINTING
        c_print("pol_refine_iter must be nonnegative\n");
        #endif
        return 1;
    }
    // Update pol_refine_iter
    work->settings->pol_refine_iter = pol_refine_iter_new;

    return 0;
}


/**
 * Update verbose setting
 * @param  work        Workspace
 * @param  verbose_new New verbose setting
 * @return             Exitflag
 */
c_int osqp_update_verbose(Work * work, c_int verbose_new) {
    // Check that verbose is either 0 or 1
    if (verbose_new != 0 && verbose_new != 1) {
      #ifdef PRINTING
      c_print("verbose should be either 0 or 1\n");
      #endif
      return 1;
    }
    // Update verbose
    work->settings->verbose = verbose_new;

    return 0;
}


/**
 * Update warm_start setting
 * @param  work           Workspace
 * @param  warm_start_new New warm_start setting
 * @return                Exitflag
 */
c_int osqp_update_warm_start(Work * work, c_int warm_start_new) {
    // Check that warm_start is either 0 or 1
    if (warm_start_new != 0 && warm_start_new != 1) {
      #ifdef PRINTING
      c_print("warm_start should be either 0 or 1\n");
      #endif
      return 1;
    }
    // Update warm_start
    work->settings->warm_start = warm_start_new;

    return 0;
}
