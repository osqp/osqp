#ifndef OSQP_H
#define OSQP_H

#ifdef __cplusplus
extern "C" {
#endif


/* Includes */
#include "glob_opts.h"
#include "constants.h"
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
        c_float *l, *u; /* dense arrays for bounds lA, uA (size m)*/
};


/* Settings struct */
struct OSQP_SETTINGS {
        /* these *cannot* change for multiple runs with the same call to osqp_init */
        c_float rho;     /* ADMM step rho */
        c_float sigma;   /* ADMM step sigma */
        c_int scaling;   /* boolean, heuristic data rescaling */
        c_int scaling_norm; /* scaling norm */
        c_int scaling_iter; /* scaling iterations */

        /* these can change for multiple runs with the same call to osqp_init */
        c_int max_iter; /* maximum iterations to take */
        c_float eps_abs;  /* absolute convergence tolerance  */
        c_float eps_rel;  /* relative convergence tolerance  */
        c_float eps_inf;  /* infeasibility tolerance  */
        c_float eps_unb;  /* unboundedness tolerance  */
        c_float alpha; /* relaxation parameter */
        c_float delta; /* regularization parameter for polishing */
        c_int polishing; /* boolean, polish ADMM solution */
        c_int pol_refine_iter; /* iterative refinement steps in polishing */
        c_int verbose; /* boolean, write out progress  */
        c_int warm_start; /* boolean, warm start */
};


/* OSQP Environment */
struct OSQP_WORK {
        // Problem Data to work on (possibly Scaled)
        Data * data;

        // Linear System solver structure
        Priv * priv;

        // Polish structure
        Polish * pol;

        // Internal solver variables
        c_float *x, *y, *z, *xz_tilde; // Iterates
        c_float *x_prev, *z_prev;      // Previous x and x.
                                       // N.B. Used also as workspace vectors for residuals.
        c_float *delta_y, *Atdelta_y;  // Infeasibility variables delta_y and
                                       // A' * delta_y
        c_float *delta_x, *Pdelta_x, *Adelta_x; // Unboundedness variables
                                                // delta_x, P * delta_x and
                                                // A * delta_x

        // flag indicating whether the solve function has been run before
        c_int first_run;

        // Other internal structures
        Settings *settings;              // Problem settings
        Scaling *scaling;                // Scaling Vectors
        Solution *solution;              // Problem Solution
        Info *info;                      // Solver information

        #ifdef PROFILING
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
        c_float *y;       // Lagrange multiplier associated to l <= Ax <= u
};

/* Solver Information */
struct OSQP_INFO {
        c_int iter;          /* number of iterations taken */
        char status[32];     /* status string, e.g. 'Solved' */
        c_int status_val;    /* status as c_int, defined in constants.h */
        c_int status_polish; /* polish status: successful (1), unperformed (0), (-1) unsuccessful */
        c_float obj_val;     /* primal objective */
        c_float pri_res;     /* norm of primal residual */
        c_float dua_res;     /* norm of dual residual */

        #ifdef PROFILING
        c_float setup_time;  /* time taken for setup phase (milliseconds) */
        c_float solve_time;  /* time taken for solve phase (milliseconds) */
        c_float polish_time; /* time taken for polish phase (milliseconds) */
        c_float run_time;    /* total time  (milliseconds) */
        #endif
};

/* Active constraints */
struct OSQP_POLISH {
    csc *Ared;            // Active rows of A
                          //    Ared = vstack[Alow, Aupp]
    c_int n_low;          // number of lower-active rows
    c_int n_upp;          // number of upper-active rows
    c_int *A_to_Alow;     // Maps indices in A to indices in Alow
    c_int *A_to_Aupp;     // Maps indices in A to indices in Aupp
    c_int *Alow_to_A;     // Maps indices in Alow to indices in A
    c_int *Aupp_to_A;     // Maps indices in Aupp to indices in A
    c_float *x;           // optimal x-solution obtained by polishing
    c_float *z;           // optimal z-solution obtained by polishing
    c_float *y_red;       // optimal dual variables associated to Ared obtained
                          //    by polishing
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



//TODO: Add sublevel API functions
/********************************************
 * Sublevel API                             *
 *                                          *
 * Edit data without performing setup again *
 ********************************************/

/**
 * Update linear cost in the problem
 * @param  work  Workspace
 * @param  q_new New linear cost
 * @return       Exitflag for errors and warnings
 */
c_int osqp_update_lin_cost(Work * work, c_float * q_new);


/**
 * Update lower and upper bounds in the problem constraints
 * @param  work   Workspace
 * @param  l_new New lower bound
 * @param  u_new New upper bound
 * @return        Exitflag: 1 if new lower bound is not <= than new upper bound
 */
c_int osqp_update_bounds(Work * work, c_float * l_new, c_float * u_new);


/**
 * Update lower bound in the problem constraints
 * @param  work   Workspace
 * @param  l_new New lower bound
 * @return        Exitflag: 1 if new lower bound is not <= than upper bound
 */
c_int osqp_update_lower_bound(Work * work, c_float * l_new);


/**
 * Update upper bound in the problem constraints
 * @param  work   Workspace
 * @param  u_new New upper bound
 * @return        Exitflag: 1 if new upper bound is not >= than lower bound
 */
c_int osqp_update_upper_bound(Work * work, c_float * u_new);



/************************************************
 * Edit settings without performing setup again *
 ************************************************/

/**
* Update max_iter setting
* @param  work         Workspace
* @param  max_iter_new New max_iter setting
* @return              Exitflag
*/
c_int osqp_update_max_iter(Work * work, c_int max_iter_new);


/**
 * Update absolute tolernace value
 * @param  work        Workspace
 * @param  eps_abs_new New absolute tolerance value
 * @return             Exitflag
 */
c_int osqp_update_eps_abs(Work * work, c_float eps_abs_new);


/**
 * Update relative tolernace value
 * @param  work        Workspace
 * @param  eps_rel_new New relative tolerance value
 * @return             Exitflag
 */
c_int osqp_update_eps_rel(Work * work, c_float eps_rel_new);


/**
 * Update relaxation parameter alpha
 * @param  work  Workspace
 * @param  alpha New relaxation parameter value
 * @return       Exitflag
 */
c_int osqp_update_alpha(Work * work, c_float alpha_new);


/**
 * Update regularization parameter in polishing
 * @param  work      Workspace
 * @param  delta_new New regularization parameter
 * @return           Exitflag
 */
c_int osqp_update_delta(Work * work, c_float delta_new);


/**
 * Update polishing setting
 * @param  work          Workspace
 * @param  polishing_new New polishing setting
 * @return               Exitflag
 */
c_int osqp_update_polishing(Work * work, c_int polishing_new);


/**
 * Update number of iterative refinement steps in polishing
 * @param  work                Workspace
 * @param  pol_refine_iter_new New iterative reginement steps
 * @return                     Exitflag
 */
c_int osqp_update_pol_refine_iter(Work * work, c_int pol_refine_iter_new);


/**
 * Update verbose setting
 * @param  work        Workspace
 * @param  verbose_new New verbose setting
 * @return             Exitflag
 */
c_int osqp_update_verbose(Work * work, c_int verbose_new);


/**
 * Update warm_start setting
 * @param  work           Workspace
 * @param  warm_start_new New warm_start setting
 * @return                Exitflag
 */
c_int osqp_update_warm_start(Work * work, c_int warm_start_new);

#ifdef __cplusplus
}
#endif

#endif
