#ifndef AUXIL_H
#define AUXIL_H

#include "types.h"

#ifdef __cplusplus
extern "C" {
#endif

/***********************************************************
* Auxiliary functions needed to evaluate ADMM iterations * *
***********************************************************/
# if OSQP_EMBEDDED_MODE != 1

/**
 * Compute rho estimate from residuals
 * @param solver Solver
 * @return       rho estimate
 */
OSQPFloat compute_rho_estimate(const OSQPSolver* solver);

/**
 * Adapt rho value based on current unscaled primal/dual residuals
 * @param solver Solver
 * @return       Exitflag
 */
OSQPInt adapt_rho(OSQPSolver* solver);

/**
 * Set values of rho vector based on constraint types.
 * returns 1 if any constraint types have been updated,
 * and 0 otherwise.
 * @param solver Solver
 */
OSQPInt set_rho_vec(OSQPSolver* solver);

/**
 * Update values of rho vector based on updated constraints.
 * If the constraints change, update the linear systems solver.
 *
 * @param solver Solver
 * @return       Exitflag
 */
OSQPInt update_rho_vec(OSQPSolver *solver);

# endif // OSQP_EMBEDDED_MODE

/**
 * Swap OSQPFloat vector pointers
 * @param a first vector
 * @param b second vector
 */
void swap_vectors(OSQPVectorf** a,
                  OSQPVectorf** b);


/**
 * Update x_tilde and z_tilde variable (first ADMM step)
 * @param solver    Solver
 * @param admm_iter Current ADMM iteration
 */
void update_xz_tilde(OSQPSolver* solver,
                     OSQPInt     admm_iter);


/**
 * Update x (second ADMM step)
 * Update also delta_x (For for dual infeasibility)
 * @param solver Solver
 */
void update_x(OSQPSolver* solver);


/**
 * Update z (third ADMM step)
 * @param solver Solver
 */
void update_z(OSQPSolver* solver);


/**
 * Update y variable (fourth ADMM step)
 * Update also delta_y to check for primal infeasibility
 * @param solver Solver
 */
void update_y(OSQPSolver* solver);


/**
 * Compute objective functions and duality gap from data at (x,y)
 * @param  solver       Solver
 * @param  x            Primal values x
 * @param  y            Dual values y
 * @param  prim_obj_val Primal objective function value
 * @param  dual_obj_val Dual objective function value
 * @param  duality_gap  Duality gap value
 */
void compute_obj_val_dual_gap(const OSQPSolver*  solver,
                              const OSQPVectorf* x,
                              const OSQPVectorf* y,
                                    OSQPFloat*   prim_obj_val,
                                    OSQPFloat*   dual_obj_val,
                                    OSQPFloat*   duality_gap);

/**
 * Check whether QP has solution
 * @param info OSQPInfo
 */
OSQPInt has_solution(const OSQPInfo* info);

/**
 * Store the QP solution
 * @param solver   Solver
 * @param solution Solution object to write to
 */
void store_solution(OSQPSolver* solver, OSQPSolution* solution);


/**
 * Update solver information
 * @param solver             Solver
 * @param iter               Iteration number
 * @param polishing          Boolean (if called from polish)
 */
void update_info(OSQPSolver* solver,
                 OSQPInt     iter,
                 OSQPInt     polishing);


/**
 * Reset solver information (after problem updates)
 * @param info               Information structure
 */
void reset_info(OSQPInfo* info);


/**
 * Update solver status (value and string)
 * @param info OSQPInfo
 * @param status_val new status value
 */
void update_status(OSQPInfo* info,
                   OSQPInt   status_val);


/**
 * Check if termination conditions are satisfied
 * If the boolean flag is ON, it checks for approximate conditions (10 x larger
 * tolerances than the ones set)
 *
 * @param  solver      Solver
 * @param  approximate Boolean
 * @return             Residuals check
 */
OSQPInt check_termination(OSQPSolver* solver,
                          OSQPInt     approximate);


# ifndef OSQP_EMBEDDED_MODE

/**
 * Validate problem data
 * @param  P  Problem data (quadratic cost term, csc format)
 * @param  q  Problem data (linear cost term)
 * @param  A  Problem data (constraint matrix, csc format)
 * @param  l  Problem data (constraint lower bound)
 * @param  u  Problem data (constraint upper bound)
 * @param  m  Problem data (number of constraints)
 * @param  n  Problem data (number of variables)
 * @return    Exitflag to check
 */
OSQPInt validate_data(const OSQPCscMatrix* P,
                      const OSQPFloat*     q,
                      const OSQPCscMatrix* A,
                      const OSQPFloat*     l,
                      const OSQPFloat*     u,
                            OSQPInt        m,
                            OSQPInt        n);

# endif /* ifndef OSQP_EMBEDDED_MODE */


/**
 * Validate problem settings
 * @param  settings   OSQPSettings to be validated
 * @param  from_setup Is the function called from osqp_setup?
 * @return            Exitflag to check
 */
OSQPInt validate_settings(const OSQPSettings* settings,
                          OSQPInt             from_setup);

#ifdef __cplusplus
}
#endif

#endif /* ifndef AUXIL_H */
