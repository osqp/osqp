#ifndef PROJ_H
#define PROJ_H

# ifdef __cplusplus
extern "C" {
# endif // ifdef __cplusplus

# include "osqp.h"
# include "types.h"
# include "lin_alg.h"


/* Define Projections onto set C involved in the ADMM algorithm */

/**
 * Project y onto the polar of the rec cone of \f$C = [l, u]\f$
 * @param y         Vector to project
 * @param l         lower bound vector
 * @param u         upper bound bector
* @param infval     value treated as infinity
 * @param work Workspace
 */
 void project_polar_reccone(OSQPVectorf      *y,
                            OSQPVectorf      *l,
                            OSQPVectorf      *u,
                            c_float      infval);
/**
 * Project z onto \f$C = [l, u]\f$
 * @param y         vector to project
 * @param l         lower bound vector
 * @param u         upper bound bector
* @param infval     positive value treated as infinity
 * @param tol       projection tolerance
 */
c_int test_in_polar_reccone(const OSQPVectorf  *y,
                            const OSQPVectorf  *l,
                            const OSQPVectorf  *u,
                            c_float        infval,
                            c_float          tol);

// Project y onto \f$C = [l, u]\f$
/**
* @param work  Workspace
 * @param y    vector to project
 */

void project(OSQPWorkspace *work, OSQPVectorf *y);


/**
 * Ensure z satisfies box constraints and y is is normal cone of z
 * @param work Workspace
 * @param z    Primal variable z
 * @param y    Dual variable y
 */
 void project_normalcone(OSQPWorkspace *work, OSQPVectorf *z, OSQPVectorf *y);

# ifdef __cplusplus
}
# endif // ifdef __cplusplus

#endif // ifndef PROJ_H
