#ifndef AUX_H
#define AUX_H

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
 c_float compute_obj_val(Data * data, c_float * x);


 /**
  * Update solver information
  * @param work Workspace
  */
 void update_info(Work *work, c_int iter);



 /**
  * Check if residuals norm meet the required tolerance
  * @param  work Workspace
  * @return      Redisuals check
  */
 c_int residuals_check(Work *work);

 #endif
