#ifndef OSQP_API_CONSTANTS_H
#define OSQP_API_CONSTANTS_H

# ifdef __cplusplus
extern "C" {
# endif // ifdef __cplusplus

#include "osqp_configure.h"

/*******************
* OSQP Versioning *
*******************/
#include "version.h"

/******************
* Solver Status  *
******************/
enum osqp_status_type {
    OSQP_SOLVED = 1,
    OSQP_SOLVED_INACCURATE,
    OSQP_PRIMAL_INFEASIBLE,
    OSQP_PRIMAL_INFEASIBLE_INACCURATE,
    OSQP_DUAL_INFEASIBLE,
    OSQP_DUAL_INFEASIBLE_INACCURATE,
    OSQP_MAX_ITER_REACHED,
    OSQP_TIME_LIMIT_REACHED,
    OSQP_NON_CVX,               /* problem non-convex */
    OSQP_SIGINT,                /* interrupted by user */
    OSQP_UNSOLVED               /* Unsolved; only setup function has been called */
};


/*************************
* Linear System Solvers *
*************************/
enum linsys_solver_type { QDLDL_SOLVER, MKL_PARDISO_SOLVER, CUDA_PCG_SOLVER, UNKNOWN_SOLVER=99 };
extern const char * LINSYS_SOLVER_NAME[];


/******************
* Solver Errors  *
******************/
enum osqp_error_type {
    OSQP_DATA_VALIDATION_ERROR = 1,  /* Start errors from 1 */
    OSQP_SETTINGS_VALIDATION_ERROR,
    OSQP_LINSYS_SOLVER_LOAD_ERROR,
    OSQP_LINSYS_SOLVER_INIT_ERROR,
    OSQP_NONCVX_ERROR,
    OSQP_MEM_ALLOC_ERROR,
    OSQP_WORKSPACE_NOT_INIT_ERROR,
    OSQP_ALGEBRA_LOAD_ERROR
};
extern const char * OSQP_ERROR_MESSAGE[];


/**********************************
* Solver Parameters and Settings *
**********************************/

#ifdef CUDA_SUPPORT
# define OSQP_LINSYS_SOLVER (CUDA_PCG_SOLVER)
#else
# define OSQP_LINSYS_SOLVER (QDLDL_SOLVER)
#endif

# define OSQP_VERBOSE               (1)
# define OSQP_WARM_STARTING         (1)
# define OSQP_SCALING               (10)
# define OSQP_POLISHING             (0)

// ADMM parameters
# define OSQP_RHO                   (0.1)
# define OSQP_SIGMA                 (1E-06)
# define OSQP_ALPHA                 (1.6)

# define OSQP_RHO_MIN               (1e-06)
# define OSQP_RHO_MAX               (1e06)
# define OSQP_RHO_TOL               (1e-04) ///< tolerance for detecting if an inequality is set to equality
# define OSQP_RHO_EQ_OVER_RHO_INEQ  (1e03)

#ifdef CUDA_SUPPORT
# define OSQP_RHO_IS_VEC            (0)
#else
# define OSQP_RHO_IS_VEC            (1)
#endif

// CG parameters
# define OSQP_CG_MAX_ITER           (20)
# define OSQP_CG_TOL_REDUCTION      (10)
# define OSQP_CG_TOL_FRACTION       (0.15)

// adaptive rho logic
# define OSQP_ADAPTIVE_RHO (1)

#ifdef CUDA_SUPPORT
#  define OSQP_ADAPTIVE_RHO_INTERVAL  (10)
#  define OSQP_ADAPTIVE_RHO_TOLERANCE (2.0)
#else
#  define OSQP_ADAPTIVE_RHO_INTERVAL  (0)
#  define OSQP_ADAPTIVE_RHO_TOLERANCE (5.0)          ///< tolerance for adopting new rho; minimum ratio between new rho and the current one
#endif

# define OSQP_ADAPTIVE_RHO_FRACTION (0.4)           ///< fraction of setup time after which we update rho
# define OSQP_ADAPTIVE_RHO_MULTIPLE_TERMINATION (4) ///< multiple of check_termination after which we update rho (if PROFILING disabled)
# define OSQP_ADAPTIVE_RHO_FIXED (100)              ///< number of iterations after which we update rho if termination_check  and PROFILING are disabled

// termination parameters
# define OSQP_MAX_ITER              (4000)
# define OSQP_EPS_ABS               (1E-3)
# define OSQP_EPS_REL               (1E-3)
# define OSQP_EPS_PRIM_INF          (1E-4)
# define OSQP_EPS_DUAL_INF          (1E-4)
# define OSQP_SCALED_TERMINATION    (0)
# define OSQP_TIME_LIMIT            (1e10)     ///< Disable time limit by default

#ifdef CUDA_SUPPORT
#  define OSQP_CHECK_TERMINATION (5)
#else
#  define OSQP_CHECK_TERMINATION    (25)
#endif

#  define OSQP_DELTA                (1E-6)
#  define OSQP_POLISH_REFINE_ITER   (3)


/*********************************
* Hard-coded values and settings *
**********************************/

# ifndef OSQP_NULL
#  define OSQP_NULL 0
# endif

# ifndef OSQP_NAN
#  define OSQP_NAN ((c_float)0x7fc00000UL)  // not a number
# endif

# ifndef OSQP_INFTY
#if defined(CUDA_SUPPORT) && defined(DFLOAT)
// Multiplying two floats that are in the order of 1e20 results in an overflow
#  define OSQP_INFTY ((c_float)1e17)
#else
#  define OSQP_INFTY ((c_float)1e30)        // infinity
#endif
# endif /* ifndef OSQP_INFTY */

# ifndef OSQP_DIVISION_TOL
#  define OSQP_DIVISION_TOL (1.0 / OSQP_INFTY)
# endif


# define OSQP_PRINT_INTERVAL 200

# define OSQP_MIN_SCALING   (1e-04) ///< minimum scaling value
# define OSQP_MAX_SCALING   (1e+04) ///< maximum scaling value

# define OSQP_CG_TOL_MIN    (1E-7)
# define OSQP_CG_POLISH_TOL (1e-5)

# ifdef __cplusplus
}
# endif // ifdef __cplusplus

#endif // ifndef OSQP_API_CONSTANTS_H
