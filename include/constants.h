#ifndef CONSTANTS_H
#define CONSTANTS_H

#ifdef __cplusplus
extern "C" {
#endif


/*******************
 * OSQP Versioning *
 *******************/
#define OSQP_VERSION ("0.1.1") /* string literals automatically null-terminated */


/******************
 * Solver Status  *
 ******************/
 // TODO: Add other statuses
#define OSQP_SOLVED (1)
#define OSQP_MAX_ITER_REACHED (-2)
// #define OSQP_SOLVED_INACCURATE (2)
#define OSQP_PRIMAL_INFEASIBLE (-3) /* primal infeasible  */
#define OSQP_DUAL_INFEASIBLE (-4) /* dual infeasible */
#define OSQP_SIGINT (-5) /* interrupted by user */
#define OSQP_UNSOLVED (-10)  /* Unsolved. Only setup function has been called */



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

#ifndef EMBEDDED
#define DELTA (1E-6)
#define POLISH (1)
#define POL_REFINE_ITER (3)
#define VERBOSE (1)
#define AUTO_RHO (1)


//Old
// #define AUTO_RHO_BETA0 (2.2074526972752477)   // Not settable by the user
// #define AUTO_RHO_BETA1 (0.78249418095777368)  // Not settable by the user
// #define AUTO_RHO_BETA2 (-0.83725587170072002) // Not settable by the user

// // Working on MPC examples
// #define AUTO_RHO_BETA0 (3.1723875550135223)   // Not settable by the user
// #define AUTO_RHO_BETA1 (0.29811867735531827)  // Not settable by the user
// #define AUTO_RHO_BETA2 (-0.55976668580992439) // Not settable by the user


// No regularization. interval [1, 1.2]
#define AUTO_RHO_BETA0 (2.2377322735057317)   // Not settable by the user
#define AUTO_RHO_BETA1 (0.73909558577990619)  // Not settable by the user
#define AUTO_RHO_BETA2 (-0.81428271821694909) // Not settable by the user







#define AUTO_RHO_MAX (1e06)                   // Not settable by user
#define AUTO_RHO_MIN (1e-06)                  // Not settable by user

#endif

#define EARLY_TERMINATE (1)
#define EARLY_TERMINATE_INTERVAL (25)
#define WARM_START (1)
#define SCALING (1)

#if EMBEDDED != 1
#define SCALING_NORM (2)
#define SCALING_ITER (3)
#define MAX_SCALING (1e6)  /// Maximum scaling
#define MIN_SCALING (1e-6)  /// Minimum scaling
#define SCALING_REG (1e-06)  /// Regularization in scaling iterations
#endif

/* Printing */
#define PRINT_INTERVAL 100



#ifdef __cplusplus
}
#endif

#endif
