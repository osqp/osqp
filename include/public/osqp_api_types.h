#ifndef OSQP_API_TYPES_H
#define OSQP_API_TYPES_H


# include "osqp_api_constants.h"

/*****************************
* OSQP API type definitions  *
******************************/

/* OSQP custom float definitions */
# ifdef OSQP_USE_LONG            // long integers
typedef long long OSQPInt; /* for indices */
# else // standard integers
typedef int OSQPInt;       /* for indices */
# endif /* ifdef OSQP_USE_LONG */


# ifndef OSQP_USE_FLOAT         // Doubles
typedef double OSQPFloat; /* for numerical values  */
# else                  // Floats
typedef float OSQPFloat;  /* for numerical values  */
# endif /* ifndef OSQP_USE_FLOAT */

#ifdef OSQP_PACK_SETTINGS
#define OSQP_ATTR_PACK __attribute__((packed))
#else
/* Don't put an attribute on when packing is disabled */
#define OSQP_ATTR_PACK
#endif

/**
 *  Matrix in compressed-column form.
 *  The structure is used internally to store matrices in the triplet form as well,
 *  but the API requires that the matrices are in the CSC format.
 */
typedef struct {
  OSQPInt    m;     ///< number of rows
  OSQPInt    n;     ///< number of columns
  OSQPInt   *p;     ///< column pointers (size n+1); col indices (size nzmax) starting from 0 for triplet format
  OSQPInt   *i;     ///< row indices, size nzmax starting from 0
  OSQPFloat *x;     ///< numerical values, size nzmax
  OSQPInt    nzmax; ///< maximum number of entries
  OSQPInt    nz;    ///< number of entries in triplet matrix, -1 for csc
  OSQPInt    owned; ///< 1 if the p, i and x pointers were allocated automatically, 0 if they are owned by the user
} OSQPCscMatrix;

/**
 * User settings
 */
typedef struct OSQP_ATTR_PACK {
  /* Note: If this struct is updated, ensure update_settings and validate_settings are also updated */
  // Linear algebra settings
  OSQPInt device;                             ///< device identifier; currently used for CUDA devices
  enum osqp_linsys_solver_type linsys_solver; ///< linear system solver to use

  // Control settings
  OSQPInt allocate_solution;                  ///< boolean; allocate solution in OSQPSolver during osqp_setup
  OSQPInt verbose;                            ///< boolean; write out progress
  OSQPInt profiler_level;                     ///< integer; level of detail for profiler annotations
  OSQPInt warm_starting;                      ///< boolean; warm start
  OSQPInt scaling;                            ///< data scaling iterations; if 0, then disabled
  OSQPInt polishing;                          ///< boolean; polish ADMM solution

  // ADMM parameters
  OSQPFloat rho;                    ///< ADMM penalty parameter
  OSQPInt   rho_is_vec;             ///< boolean; is rho scalar or vector?
  OSQPFloat sigma;                  ///< ADMM penalty parameter
  OSQPFloat alpha;                  ///< ADMM relaxation parameter

  // Restart parameters
  OSQPFloat beta;                                   ///< ADMM restart convergence rate (0,1]
  OSQPFloat lambd;                                  ///< Reflected Halpern smoothing rate [0,1]
  OSQPFloat restart_necessary;                      ///< Adaptive restarts necessary decay rate (0,1)
  OSQPFloat restart_artificial;                     ///< Adaptive restarts artificial decay rate (0,1)
  OSQPFloat xi;                                     ///< Averaged restarts parameter (0,1)
  OSQPInt ini_rest_len;                             ///< Iteration of first restart
  OSQPInt adaptive_rest;                            ///< boolean; include the two adaptive restart condition or not
  OSQPInt alpha_adjustment_reflected_halpern;       ///< boolean; include the alpha correction term in reflected halpern or not (allows for alpha other than 1.0 for reflected halpern)
  OSQPInt rho_custom_condition;                     ///< boolean; include extra rho adaptation conditions
  OSQPInt custom_average_rest;                      ///< boolean; reset the values to not the averaged but instead to a modified version or not
  OSQPInt vector_rho_in_averaged_KKT;               ///< boolean; use vectarized rho in the smoothed duality gap computation or not
  OSQPInt adapt_rho_on_restart;                     ///< boolean; adapt rho on every restart
  OSQPInt halpern_step_first_inner_iter;            ///< boolean; performe a halpern step on the first inner loop step
  OSQPInt pid_controller;                           ///< boolean; use a pid controller as the rho update scheme or not
  OSQPInt pid_controller_sqrt;                      ///< boolean; use the sqrt of the prim and dual residual ratios or not
  OSQPInt pid_controller_sqrt_mult;                 ///< boolean; use the sqrt of the prim and dual residual ratios as a multiple or not
  OSQPInt pid_controller_sqrt_mult_2;               ///< boolean; use the sqrt of the prim and dual residual ratios as a multiple or not
  OSQPInt pid_controller_log;                       ///< boolean; use the log of the prim and dual residual ratios or not
  OSQPFloat KP;                                     ///< Coefficient for proportionality, used in pid controller
  OSQPFloat KI;                                     ///< Coefficient for integrality, used in pid controller
  OSQPFloat KD;                                     ///< Coefficient for derivative, used in pid controller
  OSQPInt negate_K;                                 ///< boolean; negate all of pid controller values or not
  char restart_type[OSQP_MAX_RESTART_TYPE_LEN];     ///< Determines if we do no restarts, Halpern, Reflected, or Averaged restarts: {"none", "halpern", "reflected halpern", "averaged"}
  char halpern_scheme[OSQP_MAX_HALPERN_SCHEME_LEN]; ///< Determines if we adapt the Halpern smoothing parameter [(k + 1) / (k + 2)]: {"none", "adaptive"} 

  OSQPInt plot;   ///< boolean; save each iteration for ploting purposes


  // CG settings
  OSQPInt           cg_max_iter;      ///< maximum number of CG iterations per solve
  OSQPInt           cg_tol_reduction; ///< number of consecutive zero CG iterations before the tolerance gets halved
  OSQPFloat         cg_tol_fraction;  ///< CG tolerance (fraction of ADMM residuals)
  osqp_precond_type cg_precond;       ///< Preconditioner to use in the CG method

  // adaptive rho logic
  /**
   * rho stepsize adaption method
   */
  OSQPInt adaptive_rho;

  /**
   * Interval between rho adaptations
   *
   * When adaptive_rho == OSQP_ADAPTIVE_RHO_UPDATE_ITERATIONS, this is the number of iterations
   * between rho updates.
   *
   * Not used when adaptive_rho is any other value.
   */
  OSQPInt adaptive_rho_interval;

  /**
   * Adaptation parameter controlling when non-fixed rho adaptations occur.
   *
   * - When adaptive_rho == OSQP_ADAPTIVE_RHO_UPDATE_TIME, this is the fraction of the
   *   setup time to use as the rho adaptation interval.
   * - When adaptive_rho == OSQP_ADAPTIVE_RHO_UPDATE_KKT_ERROR, this is the fraction of
   *   the previous KKT error to adapt rho at.
   * - Not used for any other adaptive_rho value.
   */
  OSQPFloat adaptive_rho_fraction;

  /**
   * Tolerance applied when adapting rho.
   *
   * New rho must be X times larger or smaller than the current one to change it
   */
  OSQPFloat adaptive_rho_tolerance;
  OSQPFloat adaptive_rho_tolerance_greater;
  OSQPFloat adaptive_rho_tolerance_less;
  OSQPFloat rho_custom_tolerance;

  // termination parameters
  OSQPInt   max_iter;               ///< maximum number of iterations
  OSQPFloat eps_abs;                ///< absolute solution tolerance
  OSQPFloat eps_rel;                ///< relative solution tolerance
  OSQPFloat eps_prim_inf;           ///< primal infeasibility tolerance
  OSQPFloat eps_dual_inf;           ///< dual infeasibility tolerance
  OSQPInt   scaled_termination;     ///< boolean; use scaled termination criteria
  OSQPInt   check_termination;      ///< integer, check termination interval; if 0, checking is disabled
  OSQPInt   check_dualgap;          ///< Boolean; use duality gap termination criteria
  OSQPFloat time_limit;             ///< maximum time to solve the problem (seconds)

  // polishing parameters
  OSQPFloat delta;                  ///< regularization parameter for polishing
  OSQPInt   polish_refine_iter;     ///< number of iterative refinement steps in polishing
} OSQPSettings;


/**
 * Information about the solution process.
 */
typedef struct {
  // solver status
  char    status[32];     ///< Status string, e.g. 'solved'
  OSQPInt status_val;     ///< Status as OSQPInt, defined in osqp_api_constants.h
  OSQPInt status_polish;  ///< Polishing status: successful (1), unperformed (0), unsuccessful (-1)

  // solution quality
  OSQPFloat obj_val;                ///< Primal objective value
  OSQPFloat dual_obj_val;           ///< Dual objective value
  OSQPFloat prim_res;               ///< Norm of primal residual
  OSQPFloat dual_res;               ///< Norm of dual residual
  OSQPFloat duality_gap;            ///< Duality gap (Primal obj - Dual obj)
  OSQPFloat prim_normalized;        ///< Norm of primal residual / (1. + max_rel_eps)
  OSQPFloat dual_normalized;        ///< Norm of dual residual / (1. + max_rel_eps)
  OSQPFloat duality_gap_normalized; ///< Duality gap / (1. + max_rel_eps)
  OSQPFloat prim_integral;          ///< Primal residual integral
  OSQPFloat dual_integral;          ///< Dual residual integral
  OSQPFloat duality_gap_integral;   ///< Duality gap integral
  OSQPFloat total_integral;         ///< Integral sum (prim_integral + dual_integral + duality_gap_integral)

  // algorithm information
  OSQPInt   iter;             ///< Number of iterations taken
  OSQPInt   rho_updates;      ///< Number of rho updates performned
  OSQPFloat rho_estimate;     ///< Best rho estimate so far from residuals
  OSQPInt   restart;          ///< Number of restarts performed
  OSQPInt   inner_loop_iter;  ///< Number of iterations since restarting

  // timing information
  OSQPFloat setup_time;       ///< Setup phase time (seconds)
  OSQPFloat solve_time;       ///< Solve phase time (seconds)
  OSQPFloat update_time;      ///< Update phase time (seconds)
  OSQPFloat polish_time;      ///< Polish phase time (seconds)
  OSQPFloat run_time;         ///< Total solve time (seconds)
  OSQPFloat run_time_prev;    ///< Total solve time since last iter (seconds)
  OSQPFloat delta_solve_time; ///< Difference in time between consecutive iterations (seconds)

  // Convergence information
  OSQPFloat rel_kkt_error; ///< Relative KKT error
} OSQPInfo;


/**
 * Structure to hold the computed solution (if any), and any certificates of
 * infeasibility (if any) found by the solver.
 */
typedef struct {
  OSQPFloat* x;             ///< Primal solution
  OSQPFloat* y;             ///< Lagrange multiplier associated with \f$l \le Ax \le u\f$
  OSQPFloat* prim_inf_cert; ///< Primal infeasibility certificate
  OSQPFloat* dual_inf_cert; ///< Dual infeasibility certificate
} OSQPSolution;


/* Internal workspace */
typedef struct OSQPWorkspace_ OSQPWorkspace;


/**
 * Main OSQP solver structure that holds all information.
 */
typedef struct {
  /** @} */
  OSQPSettings*  settings; ///< Problem settings
  OSQPSolution*  solution; ///< Computed solution
  OSQPInfo*      info;     ///< Solver information
  OSQPWorkspace* work;     ///< Internal solver workspace (contents not public)
} OSQPSolver;



/**
 * Structure to hold the settings for the generated code
 */
typedef struct {
  OSQPInt embedded_mode;      ///< Embedded mode (1 = vector update, 2 = vector + matrix update)
  OSQPInt float_type;         ///< Use floats if 1, doubles if 0
  OSQPInt printing_enable;    ///< Enable printing if 1
  OSQPInt profiling_enable;   ///< Enable timing of code sections if 1
  OSQPInt interrupt_enable;   ///< Enable interrupt checking if 1
  OSQPInt derivatives_enable; ///< Enable deriatives if 1
} OSQPCodegenDefines;

#endif /* ifndef OSQP_API_TYPES_H */
