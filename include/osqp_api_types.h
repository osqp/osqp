#ifndef OSQP_API_TYPES_H
#define OSQP_API_TYPES_H

# ifdef __cplusplus
extern "C" {
# endif // ifdef __cplusplus

# include "osqp_configure.h"
# include "osqp_api_constants.h"

/*****************************
* OSQP API type definitions  *
******************************/

/* OSQP custom float definitions */
# ifdef DLONG            // long integers
typedef long long c_int; /* for indices */
# else // standard integers
typedef int c_int;       /* for indices */
# endif /* ifdef DLONG */


# ifndef DFLOAT         // Doubles
typedef double c_float; /* for numerical values  */
# else                  // Floats
typedef float c_float;  /* for numerical values  */
# endif /* ifndef DFLOAT */


/**
 * User settings
 */
typedef struct {
  c_float rho;              ///< ADMM step rho
  c_int   rho_is_vec;       ///< boolean; defines whether rho is scalar or vector
  c_float sigma;            ///< ADMM step sigma
  c_int   scaling;          ///< heuristic data scaling iterations; if 0, then disabled.
  c_int   max_iter;         ///< maximum number of iterations
  c_float eps_abs;          ///< absolute convergence tolerance
  c_float eps_rel;          ///< relative convergence tolerance
  c_float eps_prim_inf;     ///< primal infeasibility tolerance
  c_float eps_dual_inf;     ///< dual infeasibility tolerance
  c_float  alpha;           ///< relaxation parameter
  c_int scaled_termination; ///< boolean, use scaled termination criteria
  c_int check_termination;  ///< integer, check termination interval; if 0, checking is disabled
  c_int warm_start;         ///< boolean, warm start
  enum linsys_solver_type linsys_solver; ///< linear system solver to use
  c_int algebra_device;     ///< algebra device identifier; currently used for cuda devices

# if EMBEDDED != 1
  c_int   adaptive_rho;           ///< boolean, is rho step size adaptive?
  c_int   adaptive_rho_interval;  ///< number of iterations between rho adaptations; if 0, then it is automatic
  c_float adaptive_rho_tolerance; ///< tolerance X for adapting rho. The new rho has to be X times larger or 1/X
                                  ///  times smaller than the current one to trigger a new factorization.
#  ifdef PROFILING
  c_float adaptive_rho_fraction;  ///< interval for adapting rho (fraction of the setup time)
#  endif // Profiling
# endif // EMBEDDED != 1

# ifndef EMBEDDED
  c_float delta;                         ///< regularization parameter for polishing
  c_int   polish;                        ///< boolean, polish ADMM solution
  c_int   polish_refine_iter;            ///< number of iterative refinement steps in polishing
  c_int verbose;                         ///< boolean, write out progress
# endif // ifndef EMBEDDED

# ifdef PROFILING
  c_float time_limit;                    ///< maximum seconds allowed to solve the problem; if 0, then disabled
# endif // ifdef PROFILING
} OSQPSettings;

/**
 * Solver return information
 */
typedef struct {
  c_int iter;          ///< number of iterations taken
  char  status[32];    ///< status string, e.g. 'solved'
  c_int status_val;    ///< status as c_int, defined in osqp_api_constants.h

# if EMBEDDED != 1
  c_int   rho_updates;  ///< number of rho updates
  c_float rho_estimate; ///< best rho estimate so far from residuals
# endif // if EMBEDDED != 1

# ifndef EMBEDDED
  c_int status_polish; ///< polish status: successful (1), unperformed (0), (-1) unsuccessful
# endif // ifndef EMBEDDED

  c_float obj_val;     ///< primal objective
  c_float pri_res;     ///< norm of primal residual
  c_float dua_res;     ///< norm of dual residual

# ifdef PROFILING
  c_float setup_time;  ///< time taken for setup phase (seconds)
  c_float solve_time;  ///< time taken for solve phase (seconds)
  c_float update_time; ///< time taken for update phase (seconds)
  c_float polish_time; ///< time taken for polish phase (seconds)
  c_float run_time;    ///< total time  (seconds)
# endif // ifdef PROFILING

} OSQPInfo;


/**
 * Solution structure
 */
typedef struct {
  c_float *x;             ///< primal solution
  c_float *y;             ///< Lagrange multiplier associated to \f$l <= Ax <= u\f$
  c_float *prim_inf_cert; ///< primal infeasibility certificate
  c_float *dual_inf_cert; ///< dual infeasibility certificate
} OSQPSolution;


/* Internal workspace */
typedef struct OSQPWorkspace_ OSQPWorkspace;


/*
 * OSQP Main Solver type
 */

typedef struct {
  /** @} */
  OSQPSettings  *settings; ///< problem settings
  OSQPSolution  *solution; ///< problem solution
  OSQPInfo      *info;     ///< solver information
  OSQPWorkspace *work;     ///< solver internal workspace
} OSQPSolver;



# ifdef __cplusplus
}
# endif // ifdef __cplusplus

#endif // ifndef OSQP_API_TYPES_H
