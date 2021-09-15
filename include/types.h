#ifndef OSQP_TYPES_H
# define OSQP_TYPES_H

# ifdef __cplusplus
extern "C" {
# endif // ifdef __cplusplus

# include "glob_opts.h"
# include "osqp_api_constants.h"


/******************
* Internal types *
******************/

/**
 *  Matrix in compressed-column form.
 *  The structure is used internally to store matrices in the triplet form as well,
 *  but the API requires that the matrices are in the CSC format.
 */
typedef struct {
  c_int    nzmax; ///< maximum number of entries
  c_int    m;     ///< number of rows
  c_int    n;     ///< number of columns
  c_int   *p;     ///< column pointers (size n+1); col indices (size nzmax) start from 0 when using triplet format (direct KKT matrix formation)
  c_int   *i;     ///< row indices, size nzmax starting from 0
  c_float *x;     ///< numerical values, size nzmax
  c_int    nz;    ///< number of entries in triplet matrix, -1 for csc
} csc;

/**
 * Linear system solver structure (sublevel objects initialize it differently)
 */

typedef struct linsys_solver LinSysSolver;

/**
 * OSQP Timer for statistics
 */
typedef struct OSQP_TIMER OSQPTimer;

/**
 * Problem scaling matrices stored as vectors
 */
typedef struct {
  c_float  c;    ///< cost function scaling
  c_float *D;    ///< primal variable scaling
  c_float *E;    ///< dual variable scaling
  c_float  cinv; ///< cost function rescaling
  c_float *Dinv; ///< primal variable rescaling
  c_float *Einv; ///< dual variable rescaling
} OSQPScaling;


# ifndef EMBEDDED

/**
 * Polish structure
 */
typedef struct {
  csc *Ared;          ///< active rows of A
  ///<    Ared = vstack[Alow, Aupp]
  c_int    n_low;     ///< number of lower-active rows
  c_int    n_upp;     ///< number of upper-active rows
  c_int   *A_to_Alow; ///< Maps indices in A to indices in Alow
  c_int   *A_to_Aupp; ///< Maps indices in A to indices in Aupp
  c_int   *Alow_to_A; ///< Maps indices in Alow to indices in A
  c_int   *Aupp_to_A; ///< Maps indices in Aupp to indices in A
  c_float *x;         ///< optimal x-solution obtained by polish
  c_float *z;         ///< optimal z-solution obtained by polish
  c_float *y;         ///< optimal y-solution obtained by polish
  c_float  obj_val;   ///< objective value at polished solution
  c_float  pri_res;   ///< primal residual at polished solution
  c_float  dua_res;   ///< dual residual at polished solution
} OSQPPolish;
# endif // ifndef EMBEDDED


/**********************************
* Main structures and Data Types *
**********************************/

/**
 * Data structure
 */
typedef struct {
  c_int    n; ///< number of variables n
  c_int    m; ///< number of constraints m
  csc     *P; ///< the upper triangular part of the quadratic cost matrix P in csc format (size n x n).
  csc     *A; ///< linear constraints matrix A in csc format (size m x n)
  c_float *q; ///< dense array for linear part of cost function (size n)
  c_float *l; ///< dense array for lower bound (size m)
  c_float *u; ///< dense array for upper bound (size m)
} OSQPData;


#include "osqp_api_types.h"


/**
 * OSQP Workspace
 */
struct OSQPWorkspace_ {
  /// Problem data to work on (possibly scaled)
  OSQPData *data;

  /// Linear System solver structure
  LinSysSolver *linsys_solver;

# ifndef EMBEDDED
  /// Polish structure
  OSQPPolish *pol;
# endif // ifndef EMBEDDED

  /**
   * @name Vector used to store a vectorized rho parameter
   * @{
   */
  c_float *rho_vec;     ///< vector of rho values
  c_float *rho_inv_vec; ///< vector of inv rho values

  /** @} */

# if EMBEDDED != 1
  c_int *constr_type; ///< Type of constraints: loose (-1), equality (1), inequality (0)
# endif // if EMBEDDED != 1

  /**
   * @name Iterates
   * @{
   */
  c_float *x;        ///< Iterate x
  c_float *y;        ///< Iterate y
  c_float *z;        ///< Iterate z
  c_float *xz_tilde; ///< Iterate xz_tilde

  c_float *x_prev;   ///< Previous x

  /**< NB: Used also as workspace vector for dual residual */
  c_float *z_prev;   ///< Previous z

  /**< NB: Used also as workspace vector for primal residual */

  /**
   * @name Primal and dual residuals workspace variables
   *
   * Needed for residuals computation, tolerances computation,
   * approximate tolerances computation and adapting rho
   * @{
   */
  c_float *Ax;  ///< scaled A * x
  c_float *Px;  ///< scaled P * x
  c_float *Aty; ///< scaled A' * y

  /** @} */

  /**
   * @name Primal infeasibility variables
   * @{
   */
  c_float *delta_y;   ///< difference between consecutive dual iterates
  c_float *Atdelta_y; ///< A' * delta_y

  /** @} */

  /**
   * @name Dual infeasibility variables
   * @{
   */
  c_float *delta_x;  ///< difference between consecutive primal iterates
  c_float *Pdelta_x; ///< P * delta_x
  c_float *Adelta_x; ///< A * delta_x

  /** @} */

  /**
   * @name Temporary vectors used in scaling
   * @{
   */

  c_float *D_temp;   ///< temporary primal variable scaling vectors
  c_float *D_temp_A; ///< temporary primal variable scaling vectors storing norms of A columns
  c_float *E_temp;   ///< temporary constraints scaling vectors storing norms of A' columns


  /** @} */

  OSQPSettings *settings; ///< problem settings
  OSQPScaling  *scaling;  ///< scaling vectors
  OSQPSolution *solution; ///< problem solution
  OSQPInfo     *info;     ///< solver information

# ifdef PROFILING
  OSQPTimer *timer;       ///< timer object

  /// flag indicating whether the solve function has been run before
  c_int first_run;

  /// flag indicating whether the update_time should be cleared
  c_int clear_update_time;

  /// flag indicating that osqp_update_rho is called from osqp_solve function
  c_int rho_update_from_solve;
# endif // ifdef PROFILING

# ifdef PRINTING
  c_int summary_printed; ///< Has last summary been printed? (true/false)
# endif // ifdef PRINTING

};


/**
 * Define linsys_solver prototype structure
 *
 * NB: The details are defined when the linear solver is initialized depending
 *      on the choice
 */
struct linsys_solver {
  enum linsys_solver_type type;                 ///< linear system solver type functions
  c_int (*solve)(LinSysSolver *self,
                 c_float      *b);              ///< solve linear system

# ifndef EMBEDDED
  void (*free)(LinSysSolver *self);             ///< free linear system solver (only in desktop version)
# endif // ifndef EMBEDDED

# if EMBEDDED != 1
  c_int (*update_matrices)(LinSysSolver *s,
                           const csc *P,            ///< update matrices P
                           const csc *A);           //   and A in the solver

  c_int (*update_rho_vec)(LinSysSolver  *s,
                          const c_float *rho_vec);  ///< Update rho_vec
# endif // if EMBEDDED != 1

# ifndef EMBEDDED
  c_int nthreads; ///< number of threads active
# endif // ifndef EMBEDDED
};


# ifdef __cplusplus
}
# endif // ifdef __cplusplus

#endif // ifndef OSQP_TYPES_H
