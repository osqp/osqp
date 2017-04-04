#ifndef PROJ_H
#define PROJ_H

#ifdef __cplusplus
extern "C" {
#endif

#include "types.h"


/* Define Projections onto set C involved in the ADMM algorithm */

/**
 * Project z onto \f$C = [l, u]\f$
 * @param work Workspace
 */
void project_z(OSQPWorkspace *work);


#ifdef __cplusplus
}
#endif

#endif
