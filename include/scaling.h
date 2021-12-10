#ifndef SCALING_H
#define SCALING_H


// Functions to scale problem data
# include "osqp.h"
# include "types.h"
# include "lin_alg.h"


// Enable data scaling if EMBEDDED is disabled or if EMBEDDED == 2
# if EMBEDDED != 1

/**
 * Scale problem matrices
 * @param  solver OSQP solver
 * @return      exitflag
 */
c_int scale_data(OSQPSolver *solver);
# endif // if EMBEDDED != 1


/**
 * Unscale problem matrices
 * @param  solver OSQP solver
 * @return      exitflag
 */
c_int unscale_data(OSQPSolver *solver);


/**
 * Unscale solution
   @param  usolx unscaled x result
   @param  usoly unscaled y result
   @param  solx  x solution to be unscaled
   @param  solx  y solution to be unscaled
 * @param  work Workspace
 * @return      exitflag
 */
  c_int unscale_solution(OSQPVectorf* usolx,
                         OSQPVectorf* usoly,
                         const OSQPVectorf* solx,
                         const OSQPVectorf* soly,
                         OSQPWorkspace *work);


#endif /* ifndef SCALING_H */
