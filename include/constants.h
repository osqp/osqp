#ifndef CONSTANTS_H
#define CONSTANTS_H

#ifdef __cplusplus
extern "C" {
#endif


/*******************
 * OSQP Versioning *
 *******************/
#define OSQP_VERSION ("0.1.3") /* string literals automatically null-terminated */


/******************
 * Solver Status  *
 ******************/
 // TODO: Add other statuses
#define OSQP_DUAL_INFEASIBLE_INACCURATE (4)
#define OSQP_PRIMAL_INFEASIBLE_INACCURATE (3)
#define OSQP_SOLVED_INACCURATE (2)
#define OSQP_SOLVED (1)
#define OSQP_MAX_ITER_REACHED (-2)
#define OSQP_PRIMAL_INFEASIBLE (-3) /* primal infeasible  */
#define OSQP_DUAL_INFEASIBLE (-4) /* dual infeasible */
#define OSQP_SIGINT (-5) /* interrupted by user */
#define OSQP_UNSOLVED (-10)  /* Unsolved. Only setup function has been called */


/*************************
 * Linear System Solvers *
 *************************/
enum linsys_solver_type {SUITESPARSE_LDL_SOLVER, MKL_PARDISO_SOLVER};
static const char *LINSYS_SOLVER_NAME[] = {
  "suitesparse ldl", "mkl pardiso"
};
/**********************************
 * Solver Parameters and Settings *
 **********************************/

#define RHO (0.1)
#define SIGMA (1E-06)
#define MAX_ITER (2500)
#define EPS_ABS (1E-3)
#define EPS_REL (1E-3)
#define EPS_PRIM_INF (1E-4)
#define EPS_DUAL_INF (1E-4)
#define ALPHA (1.6)
#define LINSYS_SOLVER (SUITESPARSE_LDL_SOLVER)

#define RHO_MIN (1e-06)
#define RHO_MAX (1e06)
#define RHO_EQ_OVER_RHO_INEQ (1e03)
#define RHO_TOL (1e-04)


#ifndef EMBEDDED
#define DELTA (1E-6)
#define POLISH (0)
#define POLISH_REFINE_ITER (3)
#define VERBOSE (1)
#endif

#define SCALED_TERMINATION (0)
#define CHECK_TERMINATION (25)
#define WARM_START (1)
#define SCALING (10)

#if EMBEDDED != 1
#define MIN_SCALING (1e-03)  ///< Minimum scaling value
#define MAX_SCALING (1e+03)  ///< Maximum scaling value
#define SCALING_NORM (-1)     ///< Scaling norm

#define ADAPTIVE_RHO (1)
#define ADAPTIVE_RHO_INTERVAL (0)
#define ADAPTIVE_RHO_AUTO_INTERVAL_PERCENTAGE (0.7)  ///< Percentage of setup time after which we update rho
#endif

/* Printing */
#define PRINT_INTERVAL 200



#ifdef __cplusplus
}
#endif

#endif
