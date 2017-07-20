#ifndef OSQP_TYPES_H
#define OSQP_TYPES_H

#ifdef __cplusplus
extern "C" {
#endif

#include "glob_opts.h"


/******************
 * Internal types *
 ******************/

/**
 *  Matrix in compressed-column or triplet form
 */
typedef struct {
     c_int nzmax;     ///< maximum number of entries.
     c_int m;         ///< number of rows
     c_int n;         ///< number of columns
     c_int *p;        ///< column pointers (size n+1) (col indices (size nzmax) start from 0 when using triplet format (direct KKT matrix formation))
     c_int *i;        ///< row indices, size nzmax starting from 0
     c_float *x;      ///< numerical values, size nzmax
     c_int nz;       ///< # of entries in triplet matrix, -1 for csc
} csc;

/**
 * Linear system solver private structure (internal functions deal with it)
 */
typedef struct c_priv Priv;

/**
 * OSQP Timer for statistics
 */
typedef struct OSQP_TIMER OSQPTimer;

/**
 * Problem scaling matrices stored as vectors
 */
typedef struct {
        c_float *D;       ///< primal variable scaling
        c_float  *E;      ///< dual variable scaling
        c_float *Dinv;    ///< primal variable rescaling
        c_float *Einv;    ///< dual variable rescaling
} OSQPScaling;

/**
 * Solution structure
 */
typedef struct {
        c_float *x;       ///< Primal solution
        c_float *y;       ///< Lagrange multiplier associated to \f$l <= Ax <= u\f$
} OSQPSolution;


/**
 * Solver return nformation
 */
typedef struct {
        c_int iter;          ///< number of iterations taken
        char status[32];     ///< status string, e.g. 'Solved'
        c_int status_val;    ///< status as c_int, defined in constants.h

        #ifndef EMBEDDED
        c_int status_polish; ///< polish status: successful (1), unperformed (0), (-1) unsuccessful
        #endif

        c_float obj_val;     ///< primal objective
        c_float pri_res;     ///< norm of primal residual
        c_float dua_res;     ///< norm of dual residual

        #ifdef PROFILING
        c_float setup_time;  ///< time taken for setup phase (milliseconds)
        c_float solve_time;  ///< time taken for solve phase (milliseconds)
        c_float polish_time; ///< time taken for polish phase (milliseconds)
        c_float run_time;    ///< total time  (milliseconds)
        #endif
} OSQPInfo;


#ifndef EMBEDDED
/**
 * Polish structure
 */
typedef struct {
    csc *Ared;            ///< Active rows of A.
                          ///<    Ared = vstack[Alow, Aupp]
    c_int n_low;          ///< number of lower-active rows
    c_int n_upp;          ///< number of upper-active rows
    c_int *A_to_Alow;     ///< Maps indices in A to indices in Alow
    c_int *A_to_Aupp;     ///< Maps indices in A to indices in Aupp
    c_int *Alow_to_A;     ///< Maps indices in Alow to indices in A
    c_int *Aupp_to_A;     ///< Maps indices in Aupp to indices in A
    c_float *x;           ///< optimal x-solution obtained by polish
    c_float *z;           ///< optimal z-solution obtained by polish
    c_float *y_red;       ///< optimal dual variables associated to Ared obtained
                          ///<    by polish
    c_float obj_val;      ///< objective value at polished solution
    c_float pri_res;      ///< primal residual at polished solution
    c_float dua_res;      ///< dual residual at polished solution
} OSQPPolish;
#endif



/**********************************
 * Main structures and Data Types *
 **********************************/

/**
 * Data structure
 */
typedef struct {
        c_int n;             ///< number of variables n,
        c_int m;             ///< number of constraints m
        csc *P;              ///< P: in csc format (size n x n)
        csc *A;              ///< A: in csc format (size m x n)
        c_float *q;          ///< dense array for linear part of cost function (size n)
        c_float *l;          ///< dense array for lower bound (size m)
        c_float *u;          ///< dense array for upper bound (size m)
} OSQPData;


/**
 * Settings struct
 */
typedef struct {
        /**
         * @name These *cannot* change for multiple runs with the same call to osqp_setup
         * @{
         */
        c_float rho;     ///< ADMM step rho
        c_float sigma;   ///< ADMM step sigma
        c_int scaling;   ///< boolean, heuristic data rescaling

        #if EMBEDDED != 1
        c_int scaling_iter; ///< scaling iteration
        #endif
        /** @} */

        /**
         * @name These these can change for multiple runs with the same call to osqp_setup
         * @{
         */
        c_int max_iter; ///< maximum iterations to tak
        c_float eps_abs;  ///< absolute convergence tolerance
        c_float eps_rel;  ///< relative convergence tolerance
        c_float eps_prim_inf;  ///< primal infeasibility tolerance
        c_float eps_dual_inf;  ///< dual infeasibility tolerance
        c_float alpha; ///< relaxation paramete

        #ifndef EMBEDDED
        c_float delta; ///< regularization parameter for polis
        c_int polish; ///< boolean, polish ADMM solutio
        c_int pol_refine_iter; ///< iterative refinement steps in polis

        c_int verbose; ///< boolean, write out progres
        c_int auto_rho; ///< boolean, true if rho is chosen automatically
        #endif

        c_int scaled_termination;  ///< boolean, use scaled termination criteria
        c_int early_terminate;  ///< boolean, terminate if stopping criteria is met
        c_int early_terminate_interval; ///< boolean, interval for checking termination, if early_terminate == 1
        c_int warm_start; ///< boolean, warm start

        /** @} */

} OSQPSettings;


/**
 * OSQP Workspace
 */
typedef struct {
        /// Problem data to work on (possibly scaled)
        OSQPData * data;

        /// Linear System solver structure
        Priv * priv;

        #ifndef EMBEDDED
        /// Polish structure
        OSQPPolish * pol;
        #endif

        /**
         * @name Iterates
         * @{
         */
        c_float *x; ///< Iterate x
        c_float *y; ///< Iterate y
        c_float *z; ///< Iterate z
        c_float *xz_tilde; ///< Iterate xz_tilde

        c_float *x_prev;               ///< Previous x
                                       /**< N.B. Used also as workspace vector for dual residual */
        c_float *z_prev;               ///< Previous z
                                       /**< N.B. Used also as workspace vector for primal residual */

       /**
        * @name Primal infeasibility variables
        * @{
        */
        c_float *delta_y;           ///< Difference of dual iterates
        c_float *Atdelta_y;         ///< A' * delta_y

        /** @} */

        /**
         * @name Dual infeasibility variables
         * @{
         */
        c_float *delta_x;                     ///< Difference of consecutive primal iterates
        c_float *Pdelta_x;                    ///< P * delta_x
        c_float *Adelta_x;                    ///< A * delta_x

        /** @} */

        /**
         * @name Temporary vectors used in scaling
         * @{
         */

        c_float *D_temp;            ///< temporary primal variable scaling vectors
        c_float *D_temp_A;            ///< temporary primal variable scaling vectors storing norms of A columns
        c_float *E_temp;            ///< temporary constraints scaling vectors storing norms of A' columns

        /** @} */


        /** @} */

        OSQPSettings *settings;          ///< Problem settings
        OSQPScaling *scaling;                ///< Scaling Vectors
        OSQPSolution *solution;              ///< Problem Solution
        OSQPInfo *info;                      ///< Solver information

        #ifdef PROFILING
        OSQPTimer * timer;  ///< Timer object

        /// flag indicating whether the solve function has been run before
        c_int first_run;
        #endif

        #ifdef PRINTING
        c_int summary_printed;  ///< Has last summary been printed? (true/false)
        #endif
} OSQPWorkspace;







#ifdef __cplusplus
}
#endif

#endif
