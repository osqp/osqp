#ifndef SCALING_H
#define SCALING_H

#ifdef __cplusplus
extern "C" {
#endif

// Functions to scale problem data
#include "lin_alg.h"
#include "types.h"

#if EMBEDDED != 1
#include "kkt.h"

#define MAX_SCALING (1e3)
#define MIN_SCALING (1e-3)
#define SCALING_REG (1e-08)

// Scale data stored in workspace
c_int scale_data(OSQPWorkspace * work);

#endif  // end EMBEDDED


// Scale solution
// c_int scale_solution(OSQPWorkspace *work);

// Unscale solution
c_int unscale_solution(OSQPWorkspace *work);

#ifdef __cplusplus
}
#endif

#endif
