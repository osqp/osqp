#ifndef SCALING_H
#define SCALING_H


// Functions to scale problem data
#include "osqp.h"
#include "types.h"
#include "lin_alg.h"

#ifdef __cplusplus
extern "C" {
#endif

// Enable data scaling if OSQP_EMBEDDED_MODE is disabled or if OSQP_EMBEDDED_MODE == 2
# if OSQP_EMBEDDED_MODE != 1

/**
 * Scale problem matrices
 * @param  solver OSQP solver
 * @return      exitflag
 */
OSQPInt scale_data(OSQPSolver* solver);
# endif // if OSQP_EMBEDDED_MODE != 1


/**
 * Unscale problem matrices
 * @param  solver OSQP solver
 * @return      exitflag
 */
OSQPInt unscale_data(OSQPSolver* solver);


/**
 * Unscale solution
   @param  usolx unscaled x result
   @param  usoly unscaled y result
   @param  solx  x solution to be unscaled
   @param  solx  y solution to be unscaled
 * @param  work Workspace
 * @return      exitflag
 */
OSQPInt unscale_solution(OSQPVectorf*       usolx,
                         OSQPVectorf*       usoly,
                         const OSQPVectorf* solx,
                         const OSQPVectorf* soly,
                         OSQPWorkspace*     work);

#ifdef __cplusplus
}
#endif

#endif /* ifndef SCALING_H */
