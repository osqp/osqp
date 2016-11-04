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

    // Allocate problem data
    work->n = data->n;    // Number of variables
    work->m = data->m;    // Number of linear constraints

    //TODO: Mem copy everything!
    work->P = data->P;    // Cost function matrix
    work->q = data->q;    // Linear part of cost function
    work->A = data->A;    // Linear constraints matrix
    work->lA = data->lA;  // Lower bounds on constraints
    work->uA = data->uA;  // Upper bounds on constraints
    work->lx = work->lx;  // lower bounds on variables
    work->ux = work->ux;  // upper bounds on variables

    // Initialize linear system solver private structure
    work->priv = init_priv(work->P, work->A, work->settings);

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


    // Allocate information




    return work;
}
