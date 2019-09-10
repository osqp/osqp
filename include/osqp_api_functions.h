#ifndef  OSQP_API_FUNCTIONS_H
#define  OSQP_API_FUNCTIONS_H

# ifdef __cplusplus
extern "C" {
# endif // ifdef __cplusplus

/* Types required by the OSQP API */
# include "osqp_configure.h"
# include "osqp_api_types.h"



/********************
* Main Solver API  *
********************/

/**
 * @name Main solver API
 * @{
 */


/**
 * Return OSQP version
 * @return  OSQP version
 */
const char* osqp_version(void);


/**
 * Return the number of variables and constraints
 * @param  solver Solver
 * @param  m      Pointer to m
 * @param  n      Pointer to n
 */
void osqp_get_dimensions(OSQPSolver *solver,
                         c_int      *m,
                         c_int      *n);


/**
 * Set default settings from osqp_api_constants.h file
 * assumes settings already allocated in memory
 * @param settings settings structure
 */
void osqp_set_default_settings(OSQPSettings *settings);


# ifndef EMBEDDED

/**
 * Initialize OSQP solver allocating memory.
 *
 * All the inputs must be already allocated in memory before calling.
 *
 * It performs:
 * - data and settings validation
 * - problem data scaling
 * - automatic parameters tuning (if enabled)
 * - setup linear system solver:
 *      - direct solver: KKT matrix factorization is performed here
 *      - indirect solver: KKT matrix preconditioning is performed here
 *
 * NB: This is the only function that allocates dynamic memory and is not used
 *during code generation
 *
 * @param  solverp      Solver pointer
 * @param  P            Problem data (quadratic cost term, csc format)
 * @param  q            Problem data (linear cost term)
 * @param  A            Problem data (constraint matrix, csc format)
 * @param  l            Problem data (constraint lower bound)
 * @param  u            Problem data (constraint upper bound)
 * @param  m            Problem data (number of constraints)
 * @param  n            Problem data (number of variables)
 * @param  settings     Solver settings
 * @return              Exitflag for errors (0 if no errors)
 */
 c_int osqp_setup(OSQPSolver** solverp,
                  const csc*     P,
                  const c_float* q,
                  const csc*     A,
                  const c_float* l,
                  const c_float* u,
                  c_int m,
                  c_int n,
                  const OSQPSettings *settings) ;

# endif // #ifndef EMBEDDED

/**
 * Solve quadratic program
 *
 * The final solver information is stored in the \a solver->info  structure
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
 * @return      Exitflag for errors
 */
c_int osqp_solve(OSQPSolver *solver);


# ifndef EMBEDDED

/**
 * Cleanup workspace by deallocating memory
 *
 * This function is not used in code generation
 * @param  solver Solver
 * @return        Exitflag for errors
 */
c_int osqp_cleanup(OSQPSolver *solver);

# endif // ifndef EMBEDDED

/** @} */


/********************************************
* Sublevel API                             *
*                                          *
* Edit data without performing setup again *
********************************************/

/**
 * @name Sublevel API
 * @{
 */

/**
 * Update linear cost in the problem
 * @param  solver  OSQPSolver
 * @param  q_new   New linear cost
 * @return         Exitflag for errors and warnings
 */
c_int osqp_update_lin_cost(OSQPSolver    *solver,
                           const c_float *q_new);


/**
 * Update lower and upper bounds in the problem constraints
 * @param  solver  OSQPSolver
 * @param  l_new   New lower bound, NULL if none
 * @param  u_new   New upper bound, NULL if none
 * @return         Exitflag: 1 if new lower bound is not <= than new upper bound
 */
c_int osqp_update_bounds(OSQPSolver    *solver,
                         const c_float *l_new,
                         const c_float *u_new);


/**
 * Cold start workspace variables xz and y
 * @param solver Solver
 */
void osqp_cold_start(OSQPSolver *solver);

/**
 * Warm start primal and dual variables
 * @param  solver OSQPSolver structure
 * @param  x      Primal variable, NULL if none
 * @param  y      Dual variable, NULL if none
 * @return        Exitflag
 */
c_int osqp_warm_start(OSQPSolver    *solver,
                      const c_float *x,
                      const c_float *y);


# if EMBEDDED != 1

/**
 * Update elements of matrix P (upper triangular)
 * without changing sparsity structure.
 *
 *
 *  If Px_new_idx is OSQP_NULL, Px_new is assumed to be as long as P->x
 *  and the whole P->x is replaced.
 *
 * @param  solver     OSQPSolver structure
 * @param  Px_new     Vector of new elements in P->x (upper triangular)
 * @param  Px_new_idx Index mapping new elements to positions in P->x
 * @param  P_new_n    Number of new elements to be changed
 * @return            output flag:  0: OK
 *                                  1: P_new_n > nnzP
 *                                 <0: error in the update
 */
c_int osqp_update_P(OSQPSolver    *solver,
                    const c_float *Px_new,
                    const c_int   *Px_new_idx,
                    c_int          P_new_n);


/**
 * Update elements of matrix A without changing sparsity structure.
 *
 *
 *  If Ax_new_idx is OSQP_NULL, Ax_new is assumed to be as long as A->x
 *  and the whole A->x is replaced.
 *
 * @param  solver     OSQPSolver structure
 * @param  Ax_new     Vector of new elements in A->x
 * @param  Ax_new_idx Index mapping new elements to positions in A->x
 * @param  A_new_n    Number of new elements to be changed
 * @return            output flag:  0: OK
 *                                  1: A_new_n > nnzA
 *                                 <0: error in the update
 */
c_int osqp_update_A(OSQPSolver    *solver,
                    const c_float *Ax_new,
                    const c_int   *Ax_new_idx,
                    c_int          A_new_n);


/**
 * Update elements of matrix P (upper triangular) and elements of matrix A
 * without changing sparsity structure.
 *
 *
 *  If Px_new_idx is OSQP_NULL, Px_new is assumed to be as long as P->x
 *  and the whole P->x is replaced.
 *
 *  If Ax_new_idx is OSQP_NULL, Ax_new is assumed to be as long as A->x
 *  and the whole A->x is replaced.
 *
 * @param  solver     OSQPSolver structure
 * @param  Px_new     Vector of new elements in P->x (upper triangular)
 * @param  Px_new_idx Index mapping new elements to positions in P->x
 * @param  P_new_n    Number of new elements to be changed
 * @param  Ax_new     Vector of new elements in A->x
 * @param  Ax_new_idx Index mapping new elements to positions in A->x
 * @param  A_new_n    Number of new elements to be changed
 * @return            output flag:  0: OK
 *                                  1: P_new_n > nnzP
 *                                  2: A_new_n > nnzA
 *                                 <0: error in the update
 */
c_int osqp_update_P_A(OSQPSolver    *solver,
                      const c_float *Px_new,
                      const c_int   *Px_new_idx,
                      c_int          P_new_n,
                      const c_float *Ax_new,
                      const c_int   *Ax_new_idx,
                      c_int          A_new_n);

/**
 * Update rho. Limit it between RHO_MIN and RHO_MAX.
 * @param  work         Workspace
 * @param  rho_new      New rho setting
 * @return              Exitflag
 */
c_int osqp_update_rho(OSQPSolver *solver,
                      c_float     rho_new);

# endif // if EMBEDDED != 1

/** @} */


/**
 * @name Update settings
 * @{
 */


/**
 * Update max_iter setting
 * @param  solver       OSQPSolver
 * @param  max_iter_new New max_iter setting
 * @return              Exitflag
 */
c_int osqp_update_max_iter(OSQPSolver *solver,
                           c_int       max_iter_new);


/**
 * Update absolute tolernace value
 * @param  solver      OSQPSolver
 * @param  eps_abs_new New absolute tolerance value
 * @return             Exitflag
 */
c_int osqp_update_eps_abs(OSQPSolver *solver,
                          c_float     eps_abs_new);


/**
 * Update relative tolernace value
 * @param  solver      OSQPSolver
 * @param  eps_rel_new New relative tolerance value
 * @return             Exitflag
 */
c_int osqp_update_eps_rel(OSQPSolver *solver,
                          c_float     eps_rel_new);


/**
 * Update primal infeasibility tolerance
 * @param  solver            OSQPSolver
 * @param  eps_prim_inf_new  New primal infeasibility tolerance
 * @return                   Exitflag
 */
c_int osqp_update_eps_prim_inf(OSQPSolver *solver,
                               c_float     eps_prim_inf_new);


/**
 * Update dual infeasibility tolerance
 * @param  solver            OSQPSolver
 * @param  eps_dual_inf_new  New dual infeasibility tolerance
 * @return                   Exitflag
 */
c_int osqp_update_eps_dual_inf(OSQPSolver *solver,
                               c_float     eps_dual_inf_new);


/**
 * Update relaxation parameter alpha
 * @param  solver      OSQPSolver
 * @param  alpha_new   New relaxation parameter value
 * @return             Exitflag
 */
c_int osqp_update_alpha(OSQPSolver *solver,
                        c_float     alpha_new);


/**
 * Update warm_start setting
 * @param  solver         OSQPSolver
 * @param  warm_start_new New warm_start setting
 * @return                Exitflag
 */
c_int osqp_update_warm_start(OSQPSolver *solver,
                             c_int       warm_start_new);


/**
 * Update scaled_termination setting
 * @param  solver                  OSQPSolver
 * @param  scaled_termination_new  New scaled_termination setting
 * @return                         Exitflag
 */
c_int osqp_update_scaled_termination(OSQPSolver *solver,
                                     c_int       scaled_termination_new);

/**
 * Update check_termination setting
 * @param  solver                 OSQPSolver
 * @param  check_termination_new  New check_termination setting
 * @return                        Exitflag
 */
c_int osqp_update_check_termination(OSQPSolver *solver,
                                    c_int       check_termination_new);


# ifndef EMBEDDED

/**
 * Update regularization parameter in polish
 * @param  solver    OSQPSolver
 * @param  delta_new New regularization parameter
 * @return           Exitflag
 */
c_int osqp_update_delta(OSQPSolver *solver,
                        c_float     delta_new);


/**
 * Update polish setting
 * @param  solver     OSQPSolver
 * @param  polish_new New polish setting
 * @return            Exitflag
 */
c_int osqp_update_polish(OSQPSolver *solver,
                         c_int       polish_new);


/**
 * Update number of iterative refinement steps in polish
 * @param  solver                 OSQPSolver
 * @param  polish_refine_iter_new New iterative reginement steps
 * @return                        Exitflag
 */
c_int osqp_update_polish_refine_iter(OSQPSolver *solver,
                                     c_int       polish_refine_iter_new);


/**
 * Update verbose setting
 * @param  solver      OSQPSolver
 * @param  verbose_new New verbose setting
 * @return             Exitflag
 */
c_int osqp_update_verbose(OSQPSolver *solver,
                          c_int       verbose_new);


# endif // #ifndef EMBEDDED

# ifdef PROFILING

/**
 * Update time_limit setting
 * @param  solver          OSQPSolver
 * @param  time_limit_new  New time_limit setting
 * @return                 Exitflag
 */
c_int osqp_update_time_limit(OSQPSolver *solver,
                             c_float     time_limit_new);
# endif // ifdef PROFILING

/** @} */


# ifdef __cplusplus
}
# endif // ifdef __cplusplus

#endif // ifndef OSQP_API_FUNCTIONS_H
