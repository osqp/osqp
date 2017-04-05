#ifndef SCALING_H
#define SCALING_H

#ifdef __cplusplus
extern "C" {
#endif

// Functions to scale problem data
#include "types.h"
#include "lin_alg.h"

/// Maximum scaling
#define MAX_SCALING (1e3)

/// Minimum scaling
#define MIN_SCALING (1e-3)

/// Regularization in scaling iterations
#define SCALING_REG (1e-08)


// Enable data scaling if EMBEDDED is disabled or if EMBEDDED == 2
#if EMBEDDED != 1
/**
 * Scale problem matrices
 * @param  work Workspace
 * @return      exitflag
 */
c_int scale_data(OSQPWorkspace * work);
#endif


/**
 * Unscale problem matrices
 * @param  work Workspace
 * @return      exitflag
 */
c_int unscale_data(OSQPWorkspace * work);


// Scale solution
// c_int scale_solution(OSQPWorkspace *work);

/**
 * Unscale solution
 * @param  work Workspace
 * @return      exitflag
 */
c_int unscale_solution(OSQPWorkspace *work);

#ifdef __cplusplus
}
#endif

#endif
