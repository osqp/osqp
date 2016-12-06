#ifndef AUXIL_H
#define AUXIL_H

#include "osqp.h"

/***********************************************************
 * Auxiliary functions needed to compute ADMM iterations * *
 ***********************************************************/

/**
* Cold start workspace variables
* @param work Workspace
*/
void cold_start(Work *work);


/**
* Update RHS during first tep of ADMM iteration (store it into x)
* @param  work Workspace
*/
void compute_rhs(Work *work);


/**
* Update x variable (slacks s related part)
* after solving linear system (first ADMM step)
*
* @param work Workspace
*/
void update_x(Work *work);


/**
* Project x (second ADMM step)
* @param work Workspace
*/
void project_x(Work *work);


/**
* Update u variable (third ADMM step)
* @param work Workspace
*/
void update_u(Work *work);


/**
* Compute objective function from data at value x
* @param  data Data structure
* @param  x    Value x
* @return      Objective function value
*/
c_float compute_obj_val(Work *work, c_int polish);



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
* Check if residuals norm meet the required tolerance
* @param  work Workspace
* @return      Redisuals check
*/
c_int residuals_check(Work *work);



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
