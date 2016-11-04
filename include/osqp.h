#ifndef OSQP_H
#define OSQP_H


/* Includes */
#include "constants.h"
#include "glob_opts.h"
#include "lin_alg.h"
#include "lin_sys.h"

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
        c_float *l, *u; /* dense arrays for bounds l, u (size m)*/

        Settings *settings; /* contains solver settings specified by user */
};


/* Settings struct */
struct OSQP_SETTINGS {
        /* these *cannot* change for multiple runs with the same call to osqp_init */
        // c_int normalize; /* boolean, heuristic data rescaling: 1 */
        c_float rho; /* ADMM step rho*/

        /* these can change for multiple runs with the same call to osqp_init */
        c_int max_iter; /* maximum iterations to take: 2500 */
        c_float eps;  /* convergence tolerance  */
        c_float alpha; /* relaxation parameter */
        c_int verbose; /* boolean, write out progress: 1 */
        c_int warm_start; /* boolean, warm start (put initial guess in Sol
                               struct): 0 */
};


/* Workspace */
struct OSQP_WORK {
        // Problem Data (possibly Scaled)
        c_int n; // Number of variables
        c_int m; // Number of constraints
        csc * P; // Cost Function Matrix, dimension n x n
        c_float *q; // Cost function vector q, dimension n

        csc * A; // Constraints matrix, dimension (m x n)
        c_float *lb, *ub; // Lower and upper bounds vectors for constraints, dimension m

        Priv *p; // Linear System solver structure

        Settings *settings; // Problem settings
        Scaling *scaling; // Scaling Vectors
        Solution *solution; // Problem Solution
        Info * info; // Solver information

};

/* Problem scaling */
struct OSQP_SCALING {
        c_float *D, *E; /* for normalization */
};

/* Primal and dual solutions */
struct OSQP_SOLUTION {
        c_float *x, *u;
};


/* Solver Information */
struct OSQP_INFO {
        c_int iter;      /* number of iterations taken */
        char status[32]; /* status string, e.g. 'Solved' */
        c_int status_val; /* status as c_int, defined in constants.h */
        c_int obj_val;  /* primal objective */
        c_float pri_res; /* primal residual */
        c_float dual_res; /* dual residual */
        c_float setup_time; /* time taken for setup phase (milliseconds) */
        c_float solve_time; /* time taken for solve phase (milliseconds) */
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
 * @param  data   Problem data
 * @param  info   Solver information
 * @return        Solver workspace
 */
Work * osqp_setup(const Data * data, Info *info);


/**
 * Solve Quadratic Program
 * @param  work Workspace allocated
 * @return      Exitflag for errors
 */
c_int osqp_solve(Work * work);


//TODO: Add cleanup functions
//TODO: Complete osqp.c with true functions


/********************************************
 * Sublevel API                             *
 *                                          *
 * Edit data without performing setup again *
 ********************************************/
//TODO: Add sublevel API functions







#endif
