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
 * @return
 */
OSQP_API OSQPInt osqp_capabilities(void);

/**
 * Return OSQP version
 * @return  OSQP version
 */
OSQP_API const char* osqp_version(void);


/**
 * Return the error string for a given error code.
 */
OSQP_API const char* osqp_error_message(OSQPInt error_flag);


/**
 * Return the number of variables and constraints
 * @param  solver Solver
 * @param  m      Pointer to m
 * @param  n      Pointer to n
 */
OSQP_API void osqp_get_dimensions(OSQPSolver* solver,
                                  OSQPInt*      m,
                                  OSQPInt*      n);


/**
 * Set default settings from osqp_api_constants.h file.
 * Assumes settings already allocated in memory.
 * @param settings OSQPSettings structure
 */
OSQP_API void osqp_set_default_settings(OSQPSettings* settings);


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
 * @name Update settings
 * @{
 */

/**
 * Update settings. The following settings can only be set using
 * osqp_setup and are ignored in this function:
 *  - scaling
 *  - rho_is_vec
 *  - sigma
 *  - adaptive_rho
 *  - adaptive_rho_interval
 *  - adaptive_rho_fraction
 *  - adaptive_rho_tolerance
 * Also, rho can be updated using osqp_update_rho and is ignored
 * in this function.
 * @param  solver       Solver
 * @param  new_settings Solver settings
 * @return              Exitflag for errors (0 if no errors)
 */
OSQP_API OSQPInt osqp_update_settings(OSQPSolver*         solver,
                                      const OSQPSettings* new_settings);

// TODO: Allow for ADAPTIVE_RHO_* settings to be updated.


# if OSQP_EMBEDDED_MODE != 1

/**
 * Update the ADMM parameter rho.
 * Limit it between OSQP_RHO_MIN and OSQP_RHO_MAX.
 * @param  solver  Solver
 * @param  rho_new New rho setting
 * @return         Exitflag for errors (0 if no errors)
 */
OSQP_API OSQPInt osqp_update_rho(OSQPSolver* solver,
                                OSQPFloat    rho_new);

// ------------------ Derivative stuff -----------------
OSQP_API OSQPInt osqp_adjoint_derivative(OSQPSolver*    solver,
                                         OSQPFloat*     dx,
                                         OSQPFloat*     dy_l,
                                         OSQPFloat*     dy_u,
                                         OSQPCscMatrix* dP,
                                         OSQPFloat*     dq,
                                         OSQPCscMatrix* dA,
                                         OSQPFloat*     dl,
                                         OSQPFloat*     du);
// ------------------ Derivative stuff -----------------

# endif /* if OSQP_EMBEDDED_MODE != 1 */



# ifdef OSQP_CODEGEN

/**
 * Generate source files with a statically allocated OSQPSolver structure.
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

# endif /* ifdef OSQP_CODEGEN */


/** @} */


# ifdef __cplusplus
}
# endif

#endif /* ifndef OSQP_API_FUNCTIONS_H */
