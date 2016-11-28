#ifndef OSQP_H
#define OSQP_H


/* Includes */
#include "constants.h"
#include "glob_opts.h"
#include <string.h>
#include "lin_alg.h"
#include "lin_sys.h"
#include "cs.h"
#include "util.h"
#include "polish.h"
#include "scaling.h"


/*****************************
 * Structures and Data Types *
 *****************************/

/* Problem data struct */
struct OSQP_PROBLEM_DATA {
        /* these cannot change for multiple runs for the same call to osqp_init */
        c_int n; /* Number of variables n, */
        c_int m; /* Number of constraints m */
        csc *P, *A; /* P:  in CSC format */

        /* these can change for multiple runs for the same call to osqp_init */
        c_float *q; /* dense array for linear part of cost function (size n) */
        c_float *lA, *uA; /* dense arrays for bounds lA, uA (size m)*/
};


/* Settings struct */
struct OSQP_SETTINGS {
        /* these *cannot* change for multiple runs with the same call to osqp_init */
        c_float rho; /* ADMM step rho */
        c_int scaling; /* boolean, heuristic data rescaling */
        c_int scaling_norm; /* scaling norm */
        c_int max_scaling_iter; /* maximum scaling_steps */
        c_float scaling_eps; /* scaling tolerance */

        /* these can change for multiple runs with the same call to osqp_init */
        c_int max_iter; /* maximum iterations to take */
        c_float eps_abs;  /* absolute convergence tolerance  */
        c_float eps_rel;  /* relative convergence tolerance  */
        c_float alpha; /* relaxation parameter */
        c_float delta; /* regularization parameter for polishing */
        c_int polishing; /* boolean, polish ADMM solution */
        c_int verbose; /* boolean, write out progress  */
        c_int warm_start; /* boolean, warm start */
};


/* OSQP Environment */
struct OSQP_WORK {
        // Problem Data to work on (possibly Scaled)
        Data * data;

        // Linear System solver structure
        Priv * priv;

        // Polish` structure
        Polish * pol;

        // Internal solver variables
        c_float *x, *z, *u, *z_prev;

        // Workspaces for computing dual residual
        c_float *dua_res_ws_n;  // n-dimensional workspace
        c_float *dua_res_ws_m;  // m-dimensional workspace

        // Other internal structures
        Settings *settings;              // Problem settings
        Scaling *scaling;                // Scaling Vectors
        Solution *solution;              // Problem Solution
        Info *info;                      // Solver information

        #if PROFILING > 0
        Timer * timer;  // Timer object
        #endif
};

/* Problem scaling */
struct OSQP_SCALING {
        c_float *D, *E;        /* for normalization */
        c_float *Dinv, *Einv;  /* for rescaling */
};

/* Primal and dual solutions */
struct OSQP_SOLUTION {
        c_float *x;       // Primal solution
        c_float *lambda;  // Lagrange multiplier associated to lA <= Ax <= uA
};

/* Solver Information */
struct OSQP_INFO {
        c_int iter;          /* number of iterations taken */
        char status[32];     /* status string, e.g. 'Solved' */
        c_int status_val;    /* status as c_int, defined in constants.h */
        c_int status_polish; /* polish status: successful (1), not (0) */
        c_float obj_val;     /* primal objective */
        c_float pri_res;     /* norm of primal residual */
        c_float dua_res;     /* norm of dual residual */
        c_float inf_res;     /* norm of infeasibility residual */

        #if PROFILING > 0
        c_float setup_time;  /* time taken for setup phase (milliseconds) */
        c_float solve_time;  /* time taken for solve phase (milliseconds) */
        c_float polish_time; /* time taken for polish phase (milliseconds) */
        c_float run_time;    /* total time  (milliseconds) */
        #endif
};

/* Active constraints */
struct OSQP_POLISH {
    csc *Ared;            // Matrix A containing only actiev rows
    c_int *ind_lAct;      // indices of lower-active constraints
    c_int *ind_uAct;      // indices of upper-active constraints
    c_int n_lAct;         // number of lower-active constraints
    c_int n_uAct;         // number of upper-active constraints
    c_int *A2Ared;        // Table of indices that maps A to Ared
    c_float *x;           // optimal solution obtained by polishing
    c_float *Ax;          // workspace for storing A*x
    c_float *lambda_red;  // optimal dual variables associated to Ared obtained
                          //   by polishing
    c_float obj_val;      // objective value at polished solution
    c_float pri_res;      // primal residual at polished solution
    c_float dua_res;      // dual residual at polished solution
};


/********************
 * Main Solver API  *
 ********************/

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
 * @return              Solver environment
 */
Work * osqp_setup(const Data * data, Settings *settings);


/**
 * Solve Quadratic Program
 * @param  work Workspace allocated
 * @return      Exitflag for errors
 */
c_int osqp_solve(Work * work);


/**
 * Cleanup workspace
 * @param  work Workspace
 * @return      Exitflag for errors
 */
c_int osqp_cleanup(Work * work);



/********************************************
 * Sublevel API                             *
 *                                          *
 * Edit data without performing setup again *
 ********************************************/
//TODO: Add sublevel API functions







#endif
