#ifndef CONSTANTS_H
#define CONSTANTS_H

#ifdef __cplusplus
extern "C" {
#endif


/*******************
 * OSQP Versioning *
 *******************/
#define OSQP_VERSION ("0.0.0") /* string literals automatically null-terminated */


/******************
 * Solver Status  *
 ******************/
 // TODO: Add other statuses
#define OSQP_SOLVED (1)
#define OSQP_MAX_ITER_REACHED (-2)
// #define OSQP_SOLVED_INACCURATE (2)
#define OSQP_PRIMAL_INFEASIBLE (-3) /* primal infeasible  */
#define OSQP_DUAL_INFEASIBLE (-4) /* dual infeasible   */
#define OSQP_UNSOLVED (-10)  /* Unsolved. Only setup function has been called */



/**********************************
 * Solver Parameters and Settings *
 **********************************/

#define RHO (0.1)
#define SIGMA (0.001)
#define MAX_ITER (2500)
#define EPS_ABS (1E-3)
#define EPS_REL (1E-3)
#define EPS_PRIM_INF (1E-4)
#define EPS_DUAL_INF (1E-4)
#define ALPHA (1.6)

#ifndef EMBEDDED
#define DELTA (1E-7)
#define POLISH (1)
#define POL_REFINE_ITER (3)
#define VERBOSE (1)
#define AUTO_RHO (1)
#define AUTO_RHO_OFFSET (1.07838081E-03)    // Not settable by the user
#define AUTO_RHO_SLOPE (2.31511262)         // Not settable by the user
#define AUTO_RHO_MAX (10.)                  // Not settable by user
#endif

#define EARLY_TERMINATE (1)
#define EARLY_TERMINATE_INTERVAL (25)
#define WARM_START (1)
#define SCALING (1)

#if EMBEDDED != 1
#define SCALING_NORM (2)
#define SCALING_ITER (3)
#endif

/* Printing */
#define PRINT_INTERVAL 100



#ifdef __cplusplus
}
#endif

#endif
