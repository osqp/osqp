#ifndef OSQP_H
#define OSQP_H


/* Includes */
#include "constants.h"
#include "glob_opts.h"
#include "lin_alg.h"

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








#endif
