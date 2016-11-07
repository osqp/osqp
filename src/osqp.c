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

    // Allocate internal solver variables (ADMM steps)
    work->x = c_calloc(1, (work->data->n + work->data->m) * sizeof(c_float)); // Augmented variables with slacks (n+m)
    work->z = c_calloc(1, (work->data->n + work->data->m) * sizeof(c_float));
    work->u = c_calloc(1, (work->data->n + work->data->m) * sizeof(c_float));

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
 * @param  work Workspace allocated
 * @return      Exitflag for errors
 */
c_int osqp_solve(Work * work){

    c_int exitflag = 0;

    if (work->settings->verbose){
        // Print Header for every column
        print_header();
    }

    // Initialize variables (cold start or warm start depending on settings)
    // TODO: Add proper warmstart
    cold_start(work);













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
