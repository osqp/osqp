#ifndef OSQP_TYPES_H
#define OSQP_TYPES_H

#ifdef __cplusplus
extern "C" {
#endif

// #include "cs.h"        // matrix types
#include "glob_opts.h"


/******************
 * Internal types *
 ******************/

/* matrix in compressed-column or triplet form */
typedef struct {
     c_int nzmax;     /* maximum number of entries */
     c_int m;         /* number of rows */
     c_int n;         /* number of columns */
     c_int *p;        /* column pointers (size n+1) or col indices (size nzmax) start from 0 */
     c_int *i;        /* row indices, size nzmax starting from 0*/
     c_float *x;      /* numerical values, size nzmax */
     c_int nz;       /* # of entries in triplet matrix, -1 for compressed-col */
} csc;

/* Linear system solver private structure (internal functions deal with it) */
typedef struct c_priv Priv;

/* Define OSQP Timer */
typedef struct OSQP_TIMER OSQPTimer;

/* Problem scaling */
typedef struct {
        c_float *D, *E;        /* for normalization */
        c_float *Dinv, *Einv;  /* for rescaling */
} OSQPScaling;

/* Primal and dual solutions */
typedef struct {
        c_float *x;       // Primal solution
        c_float *y;       // Lagrange multiplier associated to l <= Ax <= u
} OSQPSolution;

/* Solver Information */
typedef struct {
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
} OSQPInfo;


/* Polish structure */
typedef struct {
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
} OSQPPolish;




/**********************************
 * Main structures and Data Types *
 **********************************/


/* Data struct */
typedef struct {
        /* these cannot change for multiple runs for the same call to osqp_init */
        c_int n; /* Number of variables n, */
        c_int m; /* Number of constraints m */
        csc *P, *A; /* P:  in CSC format */

        /* these can change for multiple runs for the same call to osqp_init */
        c_float *q; /* dense array for linear part of cost function (size n) */
        c_float *l, *u; /* dense arrays for bounds l, u (size m)*/
} OSQPData;


/* Settings struct */
typedef struct {
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
} OSQPSettings;


/* OSQP Environment */
typedef struct {
        // Problem Data to work on (possibly Scaled)
        OSQPData * data;

        // Linear System solver structure
        Priv * priv;

        // Polish structure
        OSQPPolish * pol;

        // Internal solver variables
        c_float *x, *y, *z, *xz_tilde;          // Iterates
        c_float *x_prev, *z_prev;               // Previous x and x.
                                                // N.B. Used also as workspace vectors
                                                //      for residuals.
        c_float *delta_y, *Atdelta_y;           // Infeasibility variables delta_y and
                                                // A' * delta_y
        c_float *delta_x, *Pdelta_x, *Adelta_x; // Unboundedness variables
                                                // delta_x, P * delta_x and
                                                // A * delta_x
        c_float *P_x, *A_x;                     // Used in scaling:
                                                //  - preallocate values of P->x, A->x
        c_float *D_temp, *E_temp;               //  - temporary scaling vectors

        // flag indicating whether the solve function has been run before
        c_int first_run;

        // Other internal structures
        OSQPSettings *settings;          // Problem settings
        OSQPScaling *scaling;                // Scaling Vectors
        OSQPSolution *solution;              // Problem Solution
        OSQPInfo *info;                      // Solver information

        #ifdef PROFILING
        OSQPTimer * timer;  // Timer object
        #endif
} OSQPWorkspace;







#ifdef __cplusplus
}
#endif

#endif
