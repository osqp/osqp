/**
 * @file cuda-cudss_interface.h
 * @author OSQP Team
 * @brief Interface to the NVIDIA cuDSS direct linear system solver.
 */

#ifndef CUDA_CUDSS_INTERFACE_H
#define CUDA_CUDSS_INTERFACE_H

#include "osqp.h"
#include "osqp_api_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * CUDA cuDSS solver structure
 */
typedef struct cudss_solver cudss_solver;

/**
 * Initialize cuDSS solver
 *
 * @param s         Pointer to solver structure
 * @param P         Cost function matrix (upper triangular form)
 * @param A         Constraints matrix
 * @param rho_vec   Diagonal elements of rho_vec parameter
 * @param polish    Flag whether we are initializing for polish or not
 * @param settings  Solver settings
 * @return          Exitflag for errors
 */
OSQPInt init_linsys_solver_cudss(
    cudss_solver**       s,
    const OSQPCscMatrix* P,
    const OSQPCscMatrix* A,
    const OSQPFloat*     rho_vec,
    OSQPInt              polish,
    const OSQPSettings*  settings);

/**
 * Solve linear system and store result in b
 *
 * @param s Solver structure
 * @param b Right-hand side on input, solution on output
 */
void solve_linsys_cudss(
    cudss_solver* s,
    OSQPFloat*    b);

/**
 * Update solver matrices
 *
 * @param s       Solver structure
 * @param P       Cost function matrix (upper triangular form)
 * @param A       Constraints matrix
 * @param rho_vec Diagonal elements of rho_vec parameter
 * @param polish  Flag whether we are updating for polish or not
 * @return        Exitflag for errors
 */
OSQPInt update_linsys_solver_matrices_cudss(
    cudss_solver*        s,
    const OSQPCscMatrix* P,
    const OSQPCscMatrix* A,
    const OSQPFloat*     rho_vec,
    OSQPInt              polish);

/**
 * Update rho parameter
 *
 * @param s       Solver structure
 * @param rho_vec Diagonal elements of rho_vec parameter
 * @param polish  Flag whether we are updating for polish or not
 */
void update_linsys_solver_rho_vec_cudss(
    cudss_solver*    s,
    const OSQPFloat* rho_vec,
    OSQPInt          polish);

/**
 * Free solver
 *
 * @param s Solver structure
 */
void free_linsys_solver_cudss(cudss_solver* s);

/**
 * Get solver name
 *
 * @param s Solver structure
 * @return Solver name
 */
const char* name_cudss(cudss_solver* s);

#ifdef __cplusplus
}
#endif

#endif /* ifndef CUDA_CUDSS_INTERFACE_H */
