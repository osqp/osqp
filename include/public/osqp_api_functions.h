#ifndef  OSQP_API_FUNCTIONS_H
#define  OSQP_API_FUNCTIONS_H

/* Types required by the OSQP API */
# include "osqp_api_types.h"
# include "osqp_export_define.h"

# ifdef __cplusplus
extern "C" {
# endif

/********************
* Main Solver API  *
********************/

/**
 * @name Main solver API
 * @{
 */


/**
 * Return the capabilities of the OSQP solver.
 *
 * @return
 */
OSQP_API OSQPInt osqp_capabilities(void);

/**
 * Return OSQP version
 *
 * @return OSQP version string
 */
OSQP_API const char* osqp_version(void);


/**
 * Return the error string for a given error code.
 *
 * @param error_flag Error code to get description of
 * @return String describing the error code
 */
OSQP_API const char* osqp_error_message(OSQPInt error_flag);


/**
 * Return the number of variables and constraints
 *
 * @param[in]  solver Solver
 * @param[out] m      Pointer to integer that will hold m
 * @param[out] n      Pointer to integer that will hold n
 */
OSQP_API void osqp_get_dimensions(OSQPSolver* solver,
                                  OSQPInt*    m,
                                  OSQPInt*    n);

# ifndef OSQP_EMBEDDED_MODE

/**
 * Initialize OSQP solver allocating memory.
 *
 * It performs:
 * - data and settings validation
 * - problem data scaling
 * - setup linear system solver:
 *      - direct solver: KKT matrix factorization is performed here
 *      - indirect solver: reduced KKT matrix preconditioning is performed here
 *
 * NB: This is the only function that allocates dynamic memory and is not used
 * during code generation
 *
 * @param  solverp   Solver pointer
 * @param  P         Problem data (upper triangular part of quadratic cost term, csc format)
 * @param  q         Problem data (linear cost term)
 * @param  A         Problem data (constraint matrix, csc format)
 * @param  l         Problem data (constraint lower bound)
 * @param  u         Problem data (constraint upper bound)
 * @param  m         Problem data (number of constraints)
 * @param  n         Problem data (number of variables)
 * @param  settings  Solver settings
 * @return           Exitflag for errors (0 if no errors)
 */
OSQP_API OSQPInt osqp_setup(OSQPSolver**         solverp,
                            const OSQPCscMatrix* P,
                            const OSQPFloat*     q,
                            const OSQPCscMatrix* A,
                            const OSQPFloat*     l,
                            const OSQPFloat*     u,
                            OSQPInt              m,
                            OSQPInt              n,
                            const OSQPSettings*  settings);

# endif /* ifndef OSQP_EMBEDDED_MODE */

/**
 * Solve quadratic program
 *
 * The final solver information is stored in the  \a solver->info  structure
 *
 * The solution is stored in the  \a solver->solution  structure
 *
 * If the problem is primal infeasible, the certificate is stored
 * in \a solver->delta_y
 *
 * If the problem is dual infeasible, the certificate is stored in \a
 * solver->delta_x
 *
 * @param  solver Solver
 * @return        Exitflag for errors (0 if no errors)
 */
OSQP_API OSQPInt osqp_solve(OSQPSolver* solver);


# ifndef OSQP_EMBEDDED_MODE

/**
 * Cleanup workspace by deallocating memory
 *
 * This function is not used in code generation
 * @param  solver Solver
 * @return        Exitflag for errors (0 if no errors)
 */
OSQP_API OSQPInt osqp_cleanup(OSQPSolver* solver);

# endif /* ifndef OSQP_EMBEDDED_MODE */


/** @} */


/********************************************
 * Sublevel API                             *
 *                                          *
 * These functions can be called without    *
 * performing setup again.                  *
 ********************************************/

/**
 * @name Sublevel API
 * @{
 */

/**
 * Warm start primal and dual variables
 * @param  solver Solver
 * @param  x      Primal variable, NULL if none
 * @param  y      Dual variable, NULL if none
 * @return        Exitflag for errors (0 if no errors)
 */
OSQP_API OSQPInt osqp_warm_start(OSQPSolver*      solver,
                                 const OSQPFloat* x,
                                 const OSQPFloat* y);

/**
 * Cold start workspace variables xz and y
 * @param solver Solver
 */
OSQP_API void osqp_cold_start(OSQPSolver* solver);

/**
 * Update problem data vectors
 * @param  solver  Solver
 * @param  q_new   New linear cost, NULL if none
 * @param  l_new   New lower bound, NULL if none
 * @param  u_new   New upper bound, NULL if none
 * @return         Exitflag for errors (0 if no errors)
 */
OSQP_API OSQPInt osqp_update_data_vec(OSQPSolver*      solver,
                                      const OSQPFloat* q_new,
                                      const OSQPFloat* l_new,
                                      const OSQPFloat* u_new);

# if OSQP_EMBEDDED_MODE != 1

/**
 * Update elements of matrices P (upper triangular) and A by preserving
 * their sparsity structures.
 *
 * If Px_new_idx (Ax_new_idx) is OSQP_NULL, Px_new (Ax_new) is assumed
 * to be as long as P->x (A->x) and the whole P->x (A->x) is replaced.
 *
 * @param  solver     Solver
 * @param  Px_new     Vector of new elements in P->x (upper triangular), NULL if none
 * @param  Px_new_idx Index mapping new elements to positions in P->x
 * @param  P_new_n    Number of new elements to be changed
 * @param  Ax_new     Vector of new elements in A->x, NULL if none
 * @param  Ax_new_idx Index mapping new elements to positions in A->x
 * @param  A_new_n    Number of new elements to be changed
 * @return            output flag:  0: OK
 *                                  1: P_new_n > nnzP
 *                                  2: A_new_n > nnzA
 *                                 <0: error in the update
 */
OSQP_API OSQPInt osqp_update_data_mat(OSQPSolver*      solver,
                                      const OSQPFloat* Px_new,
                                      const OSQPInt*   Px_new_idx,
                                      OSQPInt          P_new_n,
                                      const OSQPFloat* Ax_new,
                                      const OSQPInt*   Ax_new_idx,
                                      OSQPInt          A_new_n);


# endif /* if OSQP_EMBEDDED_MODE != 1 */

/** @} */


/**
 * @name Settings
 * @{
 */

/**
 * Get the default settings from the osqp_api_constants.h file.
 *
 * @note the @c settings parameter must already be allocated in memory.
 *
 * @param settings Settings structure to populate
 */
OSQP_API void osqp_set_default_settings(OSQPSettings* settings);

/**
 * Update settings in @c solver with the new settings from @c new_settings.
 *
 * The following settings can only be set at problem setup time through @c osqp_setup and are ignored
 * in this function:
 *  - scaling
 *  - rho_is_vec
 *  - sigma
 *  - adaptive_rho
 *  - adaptive_rho_interval
 *  - adaptive_rho_fraction
 *  - adaptive_rho_tolerance
 *
 * The rho setting must be updated using @c osqp_update_rho, and is ignored by this function.
 *
 * @note Every setting from @c new_settings is copied to @c solver.
 *
 * @param  solver       Solver
 * @param  new_settings New solver settings
 * @return              Exitflag for errors (0 if no errors)
 */
OSQP_API OSQPInt osqp_update_settings(OSQPSolver*         solver,
                                      const OSQPSettings* new_settings);

// TODO: Allow for ADAPTIVE_RHO_* settings to be updated.


# if OSQP_EMBEDDED_MODE != 1

/**
 * Update the ADMM parameter rho.
 *
 * Limit it between OSQP_RHO_MIN and OSQP_RHO_MAX.
 *
 * @param  solver  Solver
 * @param  rho_new New rho value
 * @return         Exitflag for errors (0 if no errors)
 */
OSQP_API OSQPInt osqp_update_rho(OSQPSolver* solver,
                                 OSQPFloat   rho_new);

# endif /* if EMBEDDED != 1 */

/** @} */

/* ------------------ Derivative functions ----------------- */

/**
 * @name Solution derivatives
 * @{
 */

/**
 * Compute internal data structures needed for calculation of the adjoint derivatives of P/q/A/l/u.
 *
 * @note An optimal solution must be obtained before calling this function.
 *
 * @param[in] solver Solver
 * @param[in] dx     Vector of dx values (observed - true) of length n
 * @param[in] dy_l   Vector of dy_l values (observed - true) of length m
 * @param[in] dy_u   Vector of dy_u values (observed - true) of length m
 * @return           Exitflag for errors (0 if no errors)
 */
OSQP_API OSQPInt osqp_adjoint_derivative_compute(OSQPSolver*    solver,
                                                 OSQPFloat*     dx,
                                                 OSQPFloat*     dy_l,
                                                 OSQPFloat*     dy_u);

/**
 * Calculate adjoint derivatives of P/A.
 *
 * @note @c osqp_adjoint_derivative_compute must be called first.
 *
 * @param[in]  solver Solver
 * @param[out] dP     Matrix of dP values (n x n)
 * @param[out] dA     Matrix of dA values (m x n)
 * @return            Exitflag for errors (0 if no errors; dP, dA are filled in)
 */
OSQP_API OSQPInt osqp_adjoint_derivative_get_mat(OSQPSolver*    solver,
                                                 OSQPCscMatrix* dP,
                                                 OSQPCscMatrix* dA);

/**
 * Calculate adjoint derivatives of q/l/u.
 *
 * @note @c osqp_adjoint_derivative_compute must be called first.
 *
 * @param[in]  solver Solver
 * @param[out] dq     Vector of dq values of length n
 * @param[out] dl     Matrix of dl values of length m
 * @param[out] du     Matrix of du values of length m
 * @return            Exitflag for errors (0 if no errors; dq, dl, du are filled in)
 */
OSQP_API OSQPInt osqp_adjoint_derivative_get_vec(OSQPSolver*    solver,
                                                 OSQPFloat* dq,
                                                 OSQPFloat* dl,
                                                 OSQPFloat* du);

/** @} */

/* ------------------ Code generation functions ----------------- */

/**
 * @name Code generation
 * @{
 */

/**
 * Set default codegen define values.
 *
 * @note The @c defines structure must already be allocated in memory.
 *
 * @param defines Structure to set to default values.
 */
OSQP_API void osqp_set_default_codegen_defines(OSQPCodegenDefines* defines);

/**
 * Generate source files with a statically allocated OSQPSolver structure.
 *
 * @note @c osqp_setup must be called before a problem can be code generated.
 *
 * @param  solver     Solver
 * @param  output_dir Path to directory to output the files to.
 *                    This string must include the trailing directory separator, and
 *                    an empty string means output to the current directory.
 * @param  prefix     String prefix for the variables and generated files.
 * @param  defines    The defines to use in the generated code.
 * @return            Exitflag for errors (0 if no errors)
 */
OSQP_API OSQPInt osqp_codegen(OSQPSolver*         solver,
                              const char*         output_dir,
                              const char*         prefix,
                              OSQPCodegenDefines* defines);

/** @} */


# ifdef __cplusplus
}
# endif

#endif /* ifndef OSQP_API_FUNCTIONS_H */
