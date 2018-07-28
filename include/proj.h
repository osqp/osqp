#ifndef PROJ_H
# define PROJ_H

# ifdef __cplusplus
extern "C" {
# endif // ifdef __cplusplus

# include "types.h"


/* Define Projections onto set C involved in the ADMM algorithm */

/**
 * Project z onto \f$C = [l, u]\f$
 * @param z    Vector to project
 * @param work Workspace
 */
void project(OSQPWorkspace *work,
             OSQPVectorf   *z);


/**
 * Ensure z satisfies box constraints and y is is normal cone of z
 * @param work Workspace
 * @param z    Primal variable z
 * @param y    Dual variable y
 */
void project_normalcone(OSQPWorkspace *work,
                        OSQPVectorf   *z,
                        OSQPVectorf   *y);


# ifdef __cplusplus
}
# endif // ifdef __cplusplus

#endif // ifndef PROJ_H
