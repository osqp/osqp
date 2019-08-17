#ifndef AUXIL_H
# define AUXIL_H

# ifdef __cplusplus
extern "C" {
# endif // ifdef __cplusplus

# include "types.h"


/***********************************************************
* Auxiliary functions needed to compute ADMM iterations * *
***********************************************************/
# if EMBEDDED != 1

/**
 * Compute rho estimate from residuals
 * @param solver Solver
 * @return     rho estimate
 */
c_float compute_rho_estimate(OSQPSolver *solver);

/**
 * Adapt rho value based on current unscaled primal/dual residuals
 * @param solver Solver
 * @return     Exitflag
 */
c_int   adapt_rho(OSQPSolver *solver);

/**
 * Set values of rho vector based on constraint types.
 * returns 1 if any constraint types have been updated,
 * and 0 otherwise.
 * @param solver Solver
 */
c_int    set_rho_vec(OSQPSolver *solver);

/**
 * Update values of rho vector based on updated constraints.
 * If the constraints change, update the linear systems solver.
 *
 * @param solver Solver
 * @return     Exitflag
 */
c_int   update_rho_vec(OSQPSolver *solver);

# endif // EMBEDDED

/**
 * Swap c_float vector pointers
 * @param a first vector
 * @param b second vector
 */
void swap_vectors(OSQPVectorf **a,
                  OSQPVectorf **b);


/**
 * Cold start workspace variables xz and y
 * @param solver Solver
 */
void cold_start(OSQPSolver *solver);


/**
 * Update x_tilde and z_tilde variable (first ADMM step)
 * @param solver Solver
 */
void update_xz_tilde(OSQPSolver *solver);


/**
 * Update x (second ADMM step)
 * Update also delta_x (For for dual infeasibility)
 * @param solver Solver
 */
void update_x(OSQPSolver *solver);


/**
 * Update z (third ADMM step)
 * @param solver Solver
 */
void update_z(OSQPSolver *solver);


/**
 * Update y variable (fourth ADMM step)
 * Update also delta_y to check for primal infeasibility
 * @param solver Solver
 */
void update_y(OSQPSolver *solver);


/**
 * Compute objective function from data at value x
 * @param  solver Solver
 * @param  x    Value x
 * @return      Objective function value
 */
c_float compute_obj_val(OSQPSolver *solver,
                        OSQPVectorf   *x);

/**
 * Check whether QP has solution
 * @param info OSQPInfo
 */
c_int has_solution(OSQPInfo *info);

/**
 * Store the QP solution
 * @param solver Solver
 */
void store_solution(OSQPSolver *solver);


/**
 * Update solver information
 * @param solver             Solver
 * @param iter               Iteration number
 * @param compute_objective  Boolean (if compute the objective or not)
 * @param polish             Boolean (if called from polish)
 */
void update_info(OSQPSolver *solver,
                 c_int       iter,
                 c_int       compute_objective,
                 c_int       polish);


/**
 * Reset solver information (after problem updates)
 * @param info               Information structure
 */
void reset_info(OSQPInfo *info);


/**
 * Update solver status (value and string)
 * @param info OSQPInfo
 * @param status_val new status value
 */
void update_status(OSQPInfo *info,
                   c_int     status_val);


/**
 * Check if termination conditions are satisfied
 * If the boolean flag is ON, it checks for approximate conditions (10 x larger
 * tolerances than the ones set)
 *
 * @param  solver        Solver
 * @param  approximate Boolean
 * @return      Residuals check
 */
c_int check_termination(OSQPSolver *solver,
                        c_int          approximate);


# ifndef EMBEDDED

/**
 * Validate problem data
 * @param  P            Problem data (quadratic cost term, csc format)
 * @param  q            Problem data (linear cost term)
 * @param  A            Problem data (constraint matrix, csc format)
 * @param  l            Problem data (constraint lower bound)
 * @param  u            Problem data (constraint upper bound)
 * @param  m            Problem data (number of constraints)
 * @param  n            Problem data (number of variables)
 * @return      Exitflag to check
 */
c_int validate_data(const csc* P,
                    const c_float* q,
                    const csc* A,
                    const c_float* l,
                    const c_float* u,
                    c_int m,
                    c_int n);


/**
 * Validate problem settings
 * @param  settings OSQPSettings to be validated
 * @return          Exitflag to check
 */
c_int validate_settings(const OSQPSettings *settings);


# endif // #ifndef EMBEDDED

# ifdef __cplusplus
}
# endif // ifdef __cplusplus

#endif // ifndef AUXIL_H
