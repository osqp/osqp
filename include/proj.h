#ifndef PROJ_H
#define PROJ_H

#ifdef __cplusplus
extern "C" {
#endif

#include "osqp.h"

/* Define Projections onto set C involved in the ADMM algorithm */


/**
 * Project z onto C = [l, u]
 * @param work Workspace
 */
void project_z(Work *work);


#ifdef __cplusplus
}
#endif

#endif
