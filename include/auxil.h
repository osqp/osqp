#ifndef AUXIL_H
#define AUXIL_H

#ifdef __cplusplus
extern "C" {
#endif

#include "types.h"
#include "proj.h"
#include "lin_alg.h"
#include "constants.h"
#include "scaling.h"
#include "util.h"
#include "lin_sys.h"



/***********************************************************
 * Auxiliary functions needed to compute ADMM iterations * *
 ***********************************************************/

#ifndef EMBEDDED
/**
 * Automatically compute rho
 * @param work Workspace
 */
void compute_rho(OSQPWorkspace * work);

#endif // ifndef EMBEDDED

/**
 * Swap c_float vector pointers
 * @param a first vector
 * @param b second vector
 */
void swap_vectors(c_float ** a, c_float ** b);


/**
* Cold start workspace variables xz and y
* @param work Workspace
*/
void cold_start(OSQPWorkspace *work);



/**
 * Update x_tilde and z_tilde variable (first ADMM step)
 * @param work [description]
 */
void update_xz_tilde(OSQPWorkspace * work);


/**
* Update x (second ADMM step)
* Update also delta_x (For for dual infeasibility)
* @param work Workspace
*/
void update_x(OSQPWorkspace *work);


/**
* Update z (third ADMM step)
* @param work Workspace
*/
void update_z(OSQPWorkspace *work);


/**
* Update y variable (fourth ADMM step)
* Update also delta_y to check for primal infeasibility
* @param work Workspace
*/
void update_y(OSQPWorkspace *work);


/**
* Compute objective function from data at value x
* @param  data OSQPData structure
* @param  x    Value x
* @return      Objective function value
*/
c_float compute_obj_val(OSQPData *data, c_float * x);



/**
* Store the QP solution
* @param work Workspace
*/
void store_solution(OSQPWorkspace *work);


/**
* Update solver information
* @param work               Workspace
* @param iter               Iteration number
* @param compute_objective  Boolean (if compute the objective or not)
* @param polish             Boolean (if called from polish)
*/
void update_info(OSQPWorkspace *work, c_int iter, c_int compute_objective, c_int polish);


/**
* Update solver status (value and string)
* @param info OSQPInfo
* @param status_val new status value
*/
void update_status(OSQPInfo *info, c_int status_val);


/**
* Check if termination conditions are satisfied
* @param  work Workspace
* @return      Redisuals check
*/
c_int check_termination(OSQPWorkspace *work);



#ifndef EMBEDDED

/**
* Validate problem data
* @param  data OSQPData to be validated
* @return      Exitflag to check
*/
c_int validate_data(const OSQPData * data);


/**
* Validate problem settings
* @param  settings OSQPSettings to be validated
* @return      Exitflag to check
*/
c_int validate_settings(const OSQPSettings * settings);

#endif  // #ifndef EMBEDDED



#ifdef __cplusplus
}
#endif

#endif
