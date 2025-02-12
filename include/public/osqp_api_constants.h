#ifndef OSQP_API_CONSTANTS_H
#define OSQP_API_CONSTANTS_H

#include "osqp_configure.h"

/***********************
* Solver capabilities *
***********************/
/**
 * Flags used to represent what capabilities the compiled OSQP solver has.
 *
 * These flags are used as bit flags, so capabilities can be tested using bit-wise operations.
 */
enum osqp_capabilities_type {
    /* This enum serves as a bit-flag definition, so each capability must be represented by
       a different bit in an int variable */
    OSQP_CAPABILITY_DIRECT_SOLVER   = 0x01,    /**<< A direct linear solver is present in the algebra. */
    OSQP_CAPABILITY_INDIRECT_SOLVER = 0x02,    /**<< An indirect linear solver is present in the algebra. */
    OSQP_CAPABILITY_CODEGEN         = 0x04,    /**<< Code generation is present. */
    OSQP_CAPABILITY_UPDATE_MATRICES = 0x08,    /**<< The problem matrices can be updated. */
    OSQP_CAPABILITY_DERIVATIVES     = 0x10     /**<< Solution derivatives w.r.t P/q/A/l/u are available. */
};


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
extern const char * OSQP_STATUS_MESSAGE[];

/******************
* Polish Status  *
******************/
enum osqp_polish_status_type {
    OSQP_POLISH_LINSYS_ERROR = -2,
    OSQP_POLISH_FAILED = -1,
    OSQP_POLISH_NOT_PERFORMED = 0,
    OSQP_POLISH_SUCCESS = 1,
    OSQP_POLISH_NO_ACTIVE_SET_FOUND = 2  /* No active set detected, polishing skipped (not an error) */
};

/*************************
* Linear System Solvers *
*************************/
enum osqp_linsys_solver_type {
    OSQP_UNKNOWN_SOLVER = 0,    /* Start from 0 for unknown solver because we index an array*/
    OSQP_DIRECT_SOLVER,
    OSQP_INDIRECT_SOLVER,
};

/*********************************
* Preconditioners for CG method *
*********************************/
typedef enum {
    OSQP_NO_PRECONDITIONER = 0,      /* Don't use a preconditioner */
    OSQP_DIAGONAL_PRECONDITIONER,    /* Diagonal (Jacobi) preconditioner */
} osqp_precond_type;

/******************
* Solver Errors  *
******************/
enum osqp_error_type {
    OSQP_NO_ERROR = 0,
    OSQP_DATA_VALIDATION_ERROR = 1,  /* Start errors from 1 */
    OSQP_SETTINGS_VALIDATION_ERROR,
    OSQP_LINSYS_SOLVER_INIT_ERROR,
    OSQP_NONCVX_ERROR,
    OSQP_MEM_ALLOC_ERROR,
    OSQP_WORKSPACE_NOT_INIT_ERROR,
    OSQP_ALGEBRA_LOAD_ERROR,
    OSQP_FOPEN_ERROR,
    OSQP_CODEGEN_DEFINES_ERROR,
    OSQP_DATA_NOT_INITIALIZED,
    OSQP_FUNC_NOT_IMPLEMENTED,      /**< Function not implemented in this library */
    OSQP_LAST_ERROR_PLACE,          /* This must always be the last item in the enum */
};
extern const char * OSQP_ERROR_MESSAGE[];


/**********************************
* Solver Parameters and Settings *
**********************************/

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

#ifdef OSQP_ALGEBRA_CUDA
# define OSQP_RHO_IS_VEC            (0)
#else
# define OSQP_RHO_IS_VEC            (1)
#endif

// CG parameters
# define OSQP_CG_MAX_ITER           (20)
# define OSQP_CG_TOL_REDUCTION      (10)
# define OSQP_CG_TOL_FRACTION       (0.15)

/*******************************
 * Adaptive rho update methods *
 *******************************/
#define OSQP_ADAPTIVE_RHO_UPDATE_DISABLED   (0) ///< Disable rho adaptation
#define OSQP_ADAPTIVE_RHO_UPDATE_ITERATIONS (1) ///< Fixed iteration interval
#define OSQP_ADAPTIVE_RHO_UPDATE_TIME       (2) ///< Time based
#define OSQP_ADAPTIVE_RHO_UPDATE_KKT_ERROR  (3) ///< KKT error decrease based

// Sentinel value, not for user use
#define _OSQP_ADAPTIVE_RHO_UPDATE_LAST_VALUE (4)

#define OSQP_ADAPTIVE_RHO_UPDATE_DEFAULT (OSQP_ADAPTIVE_RHO_UPDATE_ITERATIONS)

#define OSQP_ADAPTIVE_RHO_INTERVAL  (50)       ///< Default interval for iteration-based rho update

#ifdef OSQP_ALGEBRA_CUDA
#  define OSQP_ADAPTIVE_RHO_TOLERANCE (2.0)
#else
#  define OSQP_ADAPTIVE_RHO_TOLERANCE (5.0)          ///< tolerance for adopting new rho; minimum ratio between new rho and the current one
#endif

# define OSQP_ADAPTIVE_RHO_FRACTION (0.4)           ///< fraction of setup time after which we update rho

/**
 * Multiple of check_termination after which we update rho if using interval-based
 * rho adaptation and adaptive_rho_interval == 0.
 */
# define OSQP_ADAPTIVE_RHO_MULTIPLE_TERMINATION (4)

/**
 * Number of iterations after which we update rho if using interval-based rho adaptation
 * and adaptive_rho_interval == 0 and termination_check is disabled.
 */
# define OSQP_ADAPTIVE_RHO_FIXED (100)

// termination parameters
# define OSQP_MAX_ITER              (4000)
# define OSQP_EPS_ABS               (1E-3)
# define OSQP_EPS_REL               (1E-3)
# define OSQP_EPS_PRIM_INF          (1E-4)
# define OSQP_EPS_DUAL_INF          (1E-4)
# define OSQP_SCALED_TERMINATION    (0)
# define OSQP_TIME_LIMIT            (1e10)     ///< Disable time limit by default

// Disable the duality gap termination criteria on float builds by default for now, because
// floats can't always give the necessary precision in the current solver architecture.
#ifdef OSQP_USE_FLOAT
# define OSQP_CHECK_DUALGAP         (0)
#else
# define OSQP_CHECK_DUALGAP         (1)
#endif

#ifdef OSQP_ALGEBRA_CUDA
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
#  define OSQP_NAN ((OSQPFloat)0x7fc00000UL)  // not a number
# endif

# ifndef OSQP_INFTY
#if defined(OSQP_ALGEBRA_CUDA) && defined(OSQP_USE_FLOAT)
// Multiplying two floats that are in the order of 1e20 results in an overflow
#  define OSQP_INFTY ((OSQPFloat)1e17)
#else
#  define OSQP_INFTY ((OSQPFloat)1e30)        // infinity
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

#ifdef OSQP_USE_FLOAT
# define OSQP_ZERO_DEADZONE (1e-10) ///< Minimum permitted value
#else
# define OSQP_ZERO_DEADZONE (1e-15) ///< Minimum permitted value
#endif

#endif /* ifndef OSQP_API_CONSTANTS_H */
