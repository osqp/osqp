/* KKT linear system definition and solution */

#ifndef LIN_SYS_H
#define LIN_SYS_H


#ifdef __cplusplus
extern "C" {
#endif

// #include "cs.h"
#include "types.h"

// Include solvers
#include "suitesparse_ldl.h"


#ifndef EMBEDDED
// Initialize linear system solver structure
// NB: Only the upper triangular part of P is stuffed!
/**
 * Initialize linear system solver structure
 * @param 	P		Cost function matrix
 * @param	A		Constraints matrix
 * @param	settings 	Settings structure
 * @param	polish		0/1 depending whether we are allocating for polishing or not
 *
 */
LinSysSolver * init_linsys_solver(const csc * P, const csc * A,
                const OSQPSettings *settings, c_int polish);


#endif


#ifdef __cplusplus
}
#endif

#endif
