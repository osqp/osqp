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

    //TODO: Add timing in setup phase

    // Allocate empty workspace
    work = c_calloc(1, sizeof(Work));
    if (!work){
        c_print("ERROR: allocating work failure!\n");
    }

    // TODO: Add validation for problem data

    // Allocate problem data
    work->n = data->n;    // Number of variables
    work->m = data->m;    // Number of linear constraints

    // Copy problem data into workspace
    work->P = copy_csc_mat(data->P);         // Cost function matrix    
    work->q = vec_copy(data->q, data->n);    // Linear part of cost function
    work->A = copy_csc_mat(data->A);         // Linear constraints matrix
    work->lA = vec_copy(data->lA, data->m);  // Lower bounds on constraints
    work->uA = vec_copy(data->uA, data->m);  // Upper bounds on constraints
    work->lx = vec_copy(data->lx, data->n);  // lower bounds on variables
    work->ux = vec_copy(data->ux, data->n);  // upper bounds on variables

    // Initialize linear system solver private structure
    work->priv = init_priv(work->P, work->A, work->settings);

    // Allocate internal solver variables (ADMM steps)
    work->x = c_calloc(1, (work->n + work->m) * sizeof(c_float)); // Augmented variables with slacks (n+m)
    work->z = c_calloc(1, (work->n + work->m) * sizeof(c_float));
    work->u = c_calloc(1, (work->n + work->m) * sizeof(c_float));

    // TODO: Add Validaiton for settings
    // Allocate settings
    work->settings = settings;

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
    work->solution->x = c_calloc(1, work->n * sizeof(c_float)); // Allocate primal solution
    work->solution->u = c_calloc(1, work->m * sizeof(c_float)); // Allocate dual solution


    // Allocate information
    work->info = c_calloc(1, sizeof(Info));




    return work;
}
