#ifndef OSQP_API_TYPES_H
#define OSQP_API_TYPES_H

# ifdef __cplusplus
extern "C" {
# endif /* ifdef __cplusplus */

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
  c_int algebra_device;                  ///< algebra device identifier; currently used for CUDA devices
  enum linsys_solver_type linsys_solver; ///< linear system solver to use
  c_int verbose;                         ///< boolean; write out progress
  c_int warm_starting;                   ///< boolean; warm start
  c_int scaling;                         ///< data scaling iterations; if 0, then disabled
  c_int polishing;                       ///< boolean; polish ADMM solution

  // algorithm parameters
  c_float rho;                    ///< ADMM penalty parameter
  c_int   rho_is_vec;             ///< boolean; is rho scalar or vector?
  c_float sigma;                  ///< ADMM penalty parameter
  c_float alpha;                  ///< ADMM relaxation parameter

  // TODO: CG settings
  // c_int cg_precondition;          ///< boolean; use preconditioned CG
  // c_int cg_max_iter;              ///< maximum number of CG iterations per solve

  // adaptive rho logic
  c_int   adaptive_rho;           ///< boolean, is rho step size adaptive?
  c_int   adaptive_rho_interval;  ///< number of iterations between rho adaptations; if 0, then it is timing-based
  c_float adaptive_rho_fraction;  ///< time interval for adapting rho (fraction of the setup time)
  c_float adaptive_rho_tolerance; ///< tolerance X for adapting rho; new rho must be X times larger or smaller than the current one to change it

  // TODO: allowing negative values for adaptive_rho_interval can eliminate the need for adaptive_rho

  // termination parameters
  c_int   max_iter;               ///< maximum number of iterations
  c_float eps_abs;                ///< absolute solution tolerance
  c_float eps_rel;                ///< relative solution tolerance
  c_float eps_prim_inf;           ///< primal infeasibility tolerance
  c_float eps_dual_inf;           ///< dual infeasibility tolerance
  c_int   scaled_termination;     ///< boolean; use scaled termination criteria
  c_int   check_termination;      ///< integer, check termination interval; if 0, checking is disabled
  c_float time_limit;             ///< maximum time to solve the problem (seconds)

  // polishing parameters
  c_float delta;                  ///< regularization parameter for polishing
  c_int   polish_refine_iter;     ///< number of iterative refinement steps in polishing
} OSQPSettings;


/**
 * Solver return information
 */
typedef struct {
  // solver status
  char  status[32];     ///< status string, e.g. 'solved'
  c_int status_val;     ///< status as c_int, defined in osqp_api_constants.h
  c_int status_polish;  ///< polishing status: successful (1), unperformed (0), unsuccessful (-1)

  // solution quality
  c_float obj_val;      ///< primal objective
  c_float prim_res;     ///< norm of primal residual
  c_float dual_res;     ///< norm of dual residual

  // algorithm information
  c_int   iter;         ///< number of iterations taken
  c_int   rho_updates;  ///< number of rho updates
  c_float rho_estimate; ///< best rho estimate so far from residuals

  // timing information
  c_float setup_time;  ///< setup  phase time (seconds)
  c_float solve_time;  ///< solve  phase time (seconds)
  c_float update_time; ///< update phase time (seconds)
  c_float polish_time; ///< polish phase time (seconds)
  c_float run_time;    ///< total time  (seconds)
} OSQPInfo;


/**
 * Solution structure
 */
typedef struct {
  c_float *x;             ///< primal solution
  c_float *y;             ///< Lagrange multiplier associated with \f$l \le Ax \le u\f$
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
