#ifndef PROJ_H
#define PROJ_H

#include "osqp.h"

/* Define Projections onto set C involved in the ADMM algorithm */


/**
 * Project z onto C = [l, u]
 * @param work Workspace
 */
void project_z(Work *work);


#endif
