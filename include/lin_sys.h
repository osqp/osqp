/* KKT linear system definition and solution */

#ifndef LIN_SYS_H
#define LIN_SYS_H


#ifdef __cplusplus
extern "C" {
#endif

// #include "cs.h"
#include "types.h"

// #ifdef EMBEDDED
#include "suitesparse_ldl.h"   // Include only this solver in the same directory
// #else
// Include all linear system solvers
// #include "../lin_sys/direct/suitesparse/suitesparse_ldl.h"

#ifdef MKL_FOUND
// #include "../lin_sys/direct/pardiso/pardiso.h"
#include "pardiso.h"
#endif

#endif

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

// #endif
