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
} OSQPCscMatrix;

/**
 * User settings
 */
typedef struct {
  /* Note: If this struct is updated, ensure update_settings is also updated */
  OSQPInt device;                             ///< device identifier; currently used for CUDA devices
  enum osqp_linsys_solver_type linsys_solver; ///< linear system solver to use
  OSQPInt verbose;                            ///< boolean; write out progress
  OSQPInt warm_starting;                      ///< boolean; warm start
  OSQPInt scaling;                            ///< data scaling iterations; if 0, then disabled
  OSQPInt polishing;                          ///< boolean; polish ADMM solution

  // ADMM parameters
  OSQPFloat rho;                    ///< ADMM penalty parameter
  OSQPInt   rho_is_vec;             ///< boolean; is rho scalar or vector?
  OSQPFloat sigma;                  ///< ADMM penalty parameter
  OSQPFloat alpha;                  ///< ADMM relaxation parameter

  // CG settings
  OSQPInt           cg_max_iter;      ///< maximum number of CG iterations per solve
  OSQPInt           cg_tol_reduction; ///< number of consecutive zero CG iterations before the tolerance gets halved
  OSQPFloat         cg_tol_fraction;  ///< CG tolerance (fraction of ADMM residuals)
  osqp_precond_type cg_precond;       ///< Preconditioner to use in the CG method

  // adaptive rho logic
  OSQPInt   adaptive_rho;           ///< boolean, is rho step size adaptive?
  OSQPInt   adaptive_rho_interval;  ///< number of iterations between rho adaptations; if 0, then it is timing-based
  OSQPFloat adaptive_rho_fraction;  ///< time interval for adapting rho (fraction of the setup time)
  OSQPFloat adaptive_rho_tolerance; ///< tolerance X for adapting rho; new rho must be X times larger or smaller than the current one to change it

  // TODO: allowing negative values for adaptive_rho_interval can eliminate the need for adaptive_rho

  // termination parameters
  OSQPInt   max_iter;               ///< maximum number of iterations
  OSQPFloat eps_abs;                ///< absolute solution tolerance
  OSQPFloat eps_rel;                ///< relative solution tolerance
  OSQPFloat eps_prim_inf;           ///< primal infeasibility tolerance
  OSQPFloat eps_dual_inf;           ///< dual infeasibility tolerance
  OSQPInt   scaled_termination;     ///< boolean; use scaled termination criteria
  OSQPInt   check_termination;      ///< integer, check termination interval; if 0, checking is disabled
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
  OSQPFloat obj_val;      ///< Primal objective value
  OSQPFloat prim_res;     ///< Norm of primal residual
  OSQPFloat dual_res;     ///< Norm of dual residual

  // algorithm information
  OSQPInt   iter;         ///< Number of iterations taken
  OSQPInt   rho_updates;  ///< Number of rho updates performned
  OSQPFloat rho_estimate; ///< Best rho estimate so far from residuals

  // timing information
  OSQPFloat setup_time;  ///< Setup phase time (seconds)
  OSQPFloat solve_time;  ///< Solve phase time (seconds)
  OSQPFloat update_time; ///< Update phase time (seconds)
  OSQPFloat polish_time; ///< Polish phase time (seconds)
  OSQPFloat run_time;    ///< Total solve time (seconds)
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
