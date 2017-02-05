#ifndef SCALING_H
#define SCALING_H

#ifdef __cplusplus
extern "C" {
#endif

// Functions to scale problem data

#include "osqp.h"
#define MAX_SCALING (1e3)
#define MIN_SCALING (1e-3)
#define SCALING_REG (1e-08)

// Scale data stored in workspace
c_int scale_data(Work * work);

// Scale solution
// c_int scale_solution(Work *work);

// Unscale solution
c_int unscale_solution(Work *work);

#ifdef __cplusplus
}
#endif

#endif
