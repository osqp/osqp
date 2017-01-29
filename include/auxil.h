#ifndef AUXIL_H
#define AUXIL_H

#include "osqp.h"

/***********************************************************
 * Auxiliary functions needed to compute ADMM iterations * *
 ***********************************************************/

/**
* Cold start workspace variables xz and y
* @param work Workspace
*/
void cold_start(Work *work);



/**
 * Update x_tilde and z_tilde variable (first ADMM step)
 * @param work [description]
 */
void update_xz_tilde(Work * work);


/**
* Update x (second ADMM step)
* Update also delta_x (For unboundedness)
* @param work Workspace
*/
void update_x(Work *work);


/**
* Update z (third ADMM step)
* @param work Workspace
*/
void update_z(Work *work);


/**
* Update y variable (fourth ADMM step)
* Update also delta_y to check for infeasibility
* @param work Workspace
*/
void update_y(Work *work);


/**
* Compute objective function from data at value x
* @param  data Data structure
* @param  x    Value x
* @return      Objective function value
*/
c_float compute_obj_val(Data *data, c_float * x);



/**
* Store the QP solution
* @param work Workspace
*/
void store_solution(Work *work);


/**
* Update solver information
* @param work Workspace
*/
void update_info(Work *work, c_int iter, c_int polish);


/**
* Update solver status (string)
* @param work Workspace
*/
void update_status_string(Info *info);


/**
* Check if termination conditions are satisfied
* @param  work Workspace
* @return      Redisuals check
*/
c_int check_termination(Work *work);



/**
* Validate problem data
* @param  data Data to be validated
* @return      Exitflag to check
*/
c_int validate_data(const Data * data);


/**
* Validate problem settings
* @param  data Data to be validated
* @return      Exitflag to check
*/
c_int validate_settings(const Settings * settings);

#endif
