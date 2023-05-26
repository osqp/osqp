/* Solution polishing based on assuming the active set */
#ifndef POLISH_H
#define POLISH_H


#include "osqp.h"
#include "types.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Solution polish: Solve equality constrained QP with assumed active
 *constraints
 * @param  solver OSQP solver
 * @return        Exitflag:  0: Factorization successful
 *                           1: Factorization unsuccessful
 */
OSQPInt polish(OSQPSolver* solver);

#ifdef __cplusplus
}
#endif

#endif /* ifndef POLISH_H */
