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
 * @param work Workspace
 * @return     rho estimate
 */
c_float osqp_compute_rho_estimate_(OSQPWorkspace *work);

/**
 * Adapt rho value based on current unscaled primal/dual residuals
 * @param work Workspace
 * @return     Exitflag
 */
c_int   osqp_adapt_rho_(OSQPWorkspace *work);

/**
 * Set values of rho vector based on constraint types
 * @param work Workspace
 */
void    osqp_set_rho_vec_(OSQPWorkspace *work);

/**
 * Update values of rho vector based on updated constraints.
 * If the constraints change, update the linear systems solver.
 *
 * @param work Workspace
 * @return     Exitflag
 */
c_int   osqp_update_rho_vec_(OSQPWorkspace *work);

# endif // EMBEDDED

/**
 * Swap c_float vector pointers
 * @param a first vector
 * @param b second vector
 */
void osqp_swap_vectors_(c_float **a,
                  c_float **b);


/**
 * Cold start workspace variables xz and y
 * @param work Workspace
 */
void osqp_cold_start_(OSQPWorkspace *work);


/**
 * Update x_tilde and z_tilde variable (first ADMM step)
 * @param work [description]
 */
void osqp_update_x_z_tilde_(OSQPWorkspace *work);


/**
 * Update x (second ADMM step)
 * Update also delta_x (For for dual infeasibility)
 * @param work Workspace
 */
void osqp_update_x_(OSQPWorkspace *work);


/**
 * Update z (third ADMM step)
 * @param work Workspace
 */
void osqp_update_z_(OSQPWorkspace *work);


/**
 * Update y variable (fourth ADMM step)
 * Update also delta_y to check for primal infeasibility
 * @param work Workspace
 */
void osqp_update_y_(OSQPWorkspace *work);


/**
 * Compute objective function from data at value x
 * @param  work OSQPWorkspace structure
 * @param  x    Value x
 * @return      Objective function value
 */
c_float osqp_compute_obj_val_(OSQPWorkspace *work,
                        c_float       *x);


/**
 * Store the QP solution
 * @param work Workspace
 */
void osqp_store_solution_(OSQPWorkspace *work);


/**
 * Update solver information
 * @param work               Workspace
 * @param iter               Iteration number
 * @param compute_objective  Boolean (if compute the objective or not)
 * @param polish             Boolean (if called from polish)
 */
void osqp_update_info_(OSQPWorkspace *work,
                 c_int          iter,
                 c_int          compute_objective,
                 c_int          polish);


/**
 * Reset solver information (after problem updates)
 * @param info               Information structure
 */
void osqp_reset_info_(OSQPInfo *info);


/**
 * Update solver status (value and string)
 * @param info OSQPInfo
 * @param status_val new status value
 */
void osqp_update_status_(OSQPInfo *info,
                   c_int     status_val);


/**
 * Check if termination conditions are satisfied
 * If the boolean flag is ON, it checks for approximate conditions (10 x larger
 * tolerances than the ones set)
 *
 * @param  work        Workspace
 * @param  approximate Boolean
 * @return      Redisuals check
 */
c_int osqp_check_termination_(OSQPWorkspace *work,
                        c_int          approximate);


# ifndef EMBEDDED

/**
 * Validate problem data
 * @param  data OSQPData to be validated
 * @return      Exitflag to check
 */
c_int osqp_validate_data_(const OSQPData *data);


/**
 * Validate problem settings
 * @param  settings OSQPSettings to be validated
 * @return      Exitflag to check
 */
c_int osqp_validate_settings_(const OSQPSettings *settings);

# endif // #ifndef EMBEDDED


# ifdef __cplusplus
}
# endif // ifdef __cplusplus

#endif // ifndef AUXIL_H
