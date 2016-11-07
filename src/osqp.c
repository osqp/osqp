#include "osqp.h"
#include "aux.h"
#include "util.h"

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

    #if PROFILING > 0
    timer tsetup; // Timer
    tic(&tsetup); // Start timer
    #endif

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
    work->solution->u = c_calloc(1, (work->data->n + work->data->m) * sizeof(c_float)); // Allocate dual solution


    // Allocate information
    work->info = c_calloc(1, sizeof(Info));
    work->info->status_val = OSQP_UNSOLVED;
    update_status_string(work->info);

    // Allocate timing information
    #if PROFILING > 0
    work->info->solve_time = 0.0; // Solve time to zero
    work->info->setup_time = toc(&tsetup); // Updater timer information
    #endif

    // Print header
    #if PRINTLEVEL > 1
    if (work->settings->verbose) print_setup_header(work->data, settings);
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

    #if PROFILING > 0
    timer tsolve; // Timer
    tic(&tsolve); // Start timer
    #endif

    #if PRINTLEVEL > 1
    if (work->settings->verbose){
        // Print Header for every column
        print_header();
    }
    #endif

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
        solve_lin_sys(work->settings, work->priv, work->x);
        update_x(work);

        /* Second step: z_{k+1} */
        project_x(work);

        /* Third step: u_{k+1} */
        update_u(work);
        /* End of ADMM Steps */


        /* Update information */
        update_info(work, iter);

        /* Print summary */
        #if PRINTLEVEL > 1
        if (work->settings->verbose && iter % PRINT_INTERVAL == 0)
            print_summary(work->info);
        #endif

        if (residuals_check(work)){
            // Update final information
            work->info->status_val = OSQP_SOLVED;
            break;
        }

    }


    /* Print summary for last iteration */
    #if PRINTLEVEL > 1
    print_summary(work->info);
    #endif


    /* Update final status */
    update_status_string(work->info);

    /* Update timing */
    #if PROFILING > 0
    work->info->solve_time += toc(&tsolve);
    #endif

    /* Print final footer */
    #if PRINTLEVEL > 0
    print_footer(work->info);
    #endif

    // Store solution
    prea_vec_copy(work->z, work->solution->x, work->data->n);
    prea_vec_copy(work->u, work->solution->u, work->data->n + work->data->m);

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
    c_free(work->z_prev);


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
