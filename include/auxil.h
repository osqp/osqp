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
* Update also delta_x (For unboundedness)
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
* Update also delta_y to check for infeasibility
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
* @param work Workspace
*/
void update_info(OSQPWorkspace *work, c_int iter, c_int polish);


/**
* Update solver status (string)
* @param work Workspace
*/
void update_status_string(OSQPInfo *info);


/**
* Check if termination conditions are satisfied
* @param  work Workspace
* @return      Redisuals check
*/
c_int check_termination(OSQPWorkspace *work);



/**
* Validate problem data
* @param  data OSQPData to be validated
* @return      Exitflag to check
*/
c_int validate_data(const OSQPData * data);


/**
* Validate problem settings
* @param  data OSQPData to be validated
* @return      Exitflag to check
*/
c_int validate_settings(const OSQPSettings * settings);

#ifdef __cplusplus
}
#endif

#endif
