#ifndef LIN_SYS_H
#define LIN_SYS_H

/* KKT linear system definition and solution */

#ifdef __cplusplus
extern "C" {
#endif

#include "types.h"

#ifndef EMBEDDED
// Initialize linear system solver structure
// NB: Only the upper triangular part of P is stuffed!
/**
 * Initialize linear system solver structure
 * @param P		    Cost function matrix
 * @param	A		    Constraints matrix
 * @param	sigma   Algorithm parameter
 * @param	rho_vec Algorithm parameter
 * @param	polish  0/1 depending whether we are allocating for polishing or not
 *
 */
LinSysSolver * init_linsys_solver(const csc * P, const csc * A, c_float sigma, c_float * rho_vec, enum linsys_solver_type linsys_solver, c_int polish);


#endif


#ifdef __cplusplus
}
#endif

#endif
