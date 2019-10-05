#ifndef OSQP_TYPES_H
#define OSQP_TYPES_H

# ifdef __cplusplus
extern "C" {
# endif // ifdef __cplusplus

# include "glob_opts.h"
# include "osqp.h"       //includes user API types
# include "lin_alg.h"

/******************
* Internal types *
******************/

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
  c_float  c;         ///< cost function scaling
  OSQPVectorf *D;     ///< primal variable scaling
  OSQPVectorf *E;     ///< dual variable scaling
  c_float  cinv;      ///< cost function rescaling
  OSQPVectorf *Dinv;  ///< primal variable rescaling
  OSQPVectorf *Einv;  ///< dual variable rescaling
} OSQPScaling;




# ifndef EMBEDDED

/**
 * Polish structure
 */

typedef struct {
  OSQPMatrix *Ared;        ///< active rows of A
                           //   Ared = vstack[Alow, Aupp]
  c_int        n_active;      ///< total active constraints
  OSQPVectori  *active_flags;     ///< -1/0/1 to indicate  lower/ inactive / upper active constraints
  OSQPVectorf *x;          ///< optimal x-solution obtained by polish
  OSQPVectorf *z;          ///< optimal z-solution obtained by polish
  OSQPVectorf *y;          ///< optimal y-solution obtained by polish
  c_float     obj_val;     ///< objective value at polished solution
  c_float     pri_res;     ///< primal residual at polished solution
  c_float     dua_res;     ///< dual residual at polished solution
} OSQPPolish;
# endif // ifndef EMBEDDED


/**********************************
* Main structures and Data Types *
**********************************/

/**
 * QP problem data (possibly internally scaled)
 */
typedef struct {
  c_int    n; ///< number of variables n
  c_int    m; ///< number of constraints m
  OSQPMatrix  *P; ///< the upper triangular part of the quadratic cost matrix P (size n x n).
  OSQPMatrix  *A; ///< linear constraints matrix A (size m x n)
  OSQPVectorf *q; ///< dense array for linear part of cost function (size n)
  OSQPVectorf *l; ///< dense array for lower bound (size m)
  OSQPVectorf *u; ///< dense array for upper bound (size m)
} OSQPData;


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
  OSQPVectorf *rho_vec;     ///< vector of rho values
  OSQPVectorf *rho_inv_vec; ///< vector of inv rho values

  /** @} */

# if EMBEDDED != 1
  OSQPVectori *constr_type; ///< Type of constraints: loose (-1), equality (1), inequality (0)
# endif // if EMBEDDED != 1

  /**
   * @name Iterates
   * @{
   */
  OSQPVectorf *x;           ///< Iterate x
  OSQPVectorf *y;           ///< Iterate y
  OSQPVectorf *z;           ///< Iterate z
  OSQPVectorf *xz_tilde;    ///< Iterate xz_tilde
  OSQPVectorf *xtilde_view; ///< xtilde view into xz_tilde
  OSQPVectorf *ztilde_view; ///< ztilde view into xz_tilde

  OSQPVectorf *x_prev;   ///< Previous x

  /**< NB: Used also as workspace vector for dual residual */
  OSQPVectorf *z_prev;   ///< Previous z

  /**< NB: Used also as workspace vector for primal residual */

  /**
   * @name Primal and dual residuals workspace variables
   *
   * Needed for residuals computation, tolerances computation,
   * approximate tolerances computation and adapting rho
   * @{
   */
  OSQPVectorf *Ax;  ///< scaled A * x
  OSQPVectorf *Px;  ///< scaled P * x
  OSQPVectorf *Aty; ///< scaled A * x

  /** @} */

  /**
   * @name Primal infeasibility variables
   * @{
   */
  OSQPVectorf *delta_y;   ///< difference between consecutive dual iterates
  OSQPVectorf *Atdelta_y; ///< A' * delta_y

  /** @} */

  /**
   * @name Dual infeasibility variables
   * @{
   */
  OSQPVectorf *delta_x;  ///< difference between consecutive primal iterates
  OSQPVectorf *Pdelta_x; ///< P * delta_x
  OSQPVectorf *Adelta_x; ///< A * delta_x

  /** @} */

  /**
   * @name Temporary vectors used in scaling
   * @{
   */

  OSQPVectorf *D_temp;   ///< temporary primal variable scaling vectors
  OSQPVectorf *D_temp_A; ///< temporary primal variable scaling vectors storing norms of A columns
  OSQPVectorf *E_temp;   ///< temporary constraints scaling vectors storing norms of A' columns

  /** @} */
  OSQPScaling  *scaling;  ///< scaling vectors

  /// Scaled primal and dual residuals used for computing rho estimate.
  /// They are also passed to indirect linear system solvers for computing required accuracy.
  c_float scaled_pri_res;
  c_float scaled_dua_res;

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

// NB: "typedef struct OSQPWorkspace_ OSQPWorkspace" is declared already 
// in the osqp API where the main OSQPSolver is defined.


/**
 * Define linsys_solver prototype structure
 *
 * NB: The details are defined when the linear solver is initialized depending
 *      on the choice
 */
struct linsys_solver {
  enum linsys_solver_type type;             ///< linear system solver type functions
  c_int (*solve)(LinSysSolver *self,
                 OSQPVectorf  *b,
                 c_int         admm_iter);

  void (*warm_start)(LinSysSolver      *self,
                     const OSQPVectorf *x);

# ifndef EMBEDDED
  void (*free)(LinSysSolver *self);         ///< free linear system solver (only in desktop version)
# endif // ifndef EMBEDDED

# if EMBEDDED != 1
  c_int (*update_matrices)(LinSysSolver     *self,
                           const OSQPMatrix *P,            ///< update matrices P
                           const OSQPMatrix *A);           //   and A in the solver

  c_int (*update_rho_vec)(LinSysSolver      *self,
                          const OSQPVectorf *rho_vec);  ///< Update rho_vec
# endif // if EMBEDDED != 1

# ifndef EMBEDDED
  c_int nthreads; ///< number of threads active
# endif // ifndef EMBEDDED
};


# ifdef __cplusplus
}
# endif // ifdef __cplusplus

#endif // ifndef OSQP_TYPES_H
