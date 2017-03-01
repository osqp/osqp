#ifndef OSQP_H
#define OSQP_H

#ifdef __cplusplus
extern "C" {
#endif


/* Includes */
#include "types.h"
#include "auxil.h"
#include "util.h"
#include "scaling.h"
#ifndef EMBEDDED
#include "polish.h"
#endif

/********************
 * Main Solver API  *
 ********************/

#ifndef EMBEDDED

/**
 * Initialize OSQP solver allocating memory.
 *
 * It also sets the linear system solver:
 * - direct solver: KKT matrix factorization is performed here
 *
 *
 * N.B. This is the only function that allocates dynamic memory. During code
 * generation it is going to be removed.
 *
 * @param  data         Problem data
 * @param  settings     Solver settings
 * @return              Solver environment
 */
OSQPWorkspace * osqp_setup(const OSQPData * data, OSQPSettings *settings);

#endif  // #ifndef EMBEDDED

/**
 * Solve Quadratic Program
 * @param  work Workspace allocated
 * @return      Exitflag for errors
 */
c_int osqp_solve(OSQPWorkspace * work);


#ifndef EMBEDDED

/**
 * Cleanup workspace
 * @param  work Workspace
 * @return      Exitflag for errors
 */
c_int osqp_cleanup(OSQPWorkspace * work);

#endif


//TODO: Add sublevel API functions
/********************************************
 * Sublevel API                             *
 *                                          *
 * Edit data without performing setup again *
 ********************************************/

/**
 * Update linear cost in the problem
 * @param  work  Workspace
 * @param  q_new New linear cost
 * @return       Exitflag for errors and warnings
 */
c_int osqp_update_lin_cost(OSQPWorkspace * work, c_float * q_new);


/**
 * Update lower and upper bounds in the problem constraints
 * @param  work   Workspace
 * @param  l_new New lower bound
 * @param  u_new New upper bound
 * @return        Exitflag: 1 if new lower bound is not <= than new upper bound
 */
c_int osqp_update_bounds(OSQPWorkspace * work, c_float * l_new, c_float * u_new);


/**
 * Update lower bound in the problem constraints
 * @param  work   Workspace
 * @param  l_new New lower bound
 * @return        Exitflag: 1 if new lower bound is not <= than upper bound
 */
c_int osqp_update_lower_bound(OSQPWorkspace * work, c_float * l_new);


/**
 * Update upper bound in the problem constraints
 * @param  work   Workspace
 * @param  u_new New upper bound
 * @return        Exitflag: 1 if new upper bound is not >= than lower bound
 */
c_int osqp_update_upper_bound(OSQPWorkspace * work, c_float * u_new);


/**
 * Warm start primal and dual variables
 * @param  work Workspace structure
 * @param  x    Primal variable
 * @param  y    Dual variable
 * @return      Exitflag
 */
c_int osqp_warm_start(OSQPWorkspace * work, c_float * x, c_float * y);


/**
 * Warm start primal variable
 * @param  work Workspace structure
 * @param  x    Primal variable
 * @return      Exitflag
 */
c_int osqp_warm_start_x(OSQPWorkspace * work, c_float * x);


/**
 * Warm start dual variable
 * @param  work Workspace structure
 * @param  y    Dual variable
 * @return      Exitflag
 */
c_int osqp_warm_start_y(OSQPWorkspace * work, c_float * y);



/************************************************
 * Edit settings without performing setup again *
 ************************************************/

/**
* Update max_iter setting
* @param  work         Workspace
* @param  max_iter_new New max_iter setting
* @return              Exitflag
*/
c_int osqp_update_max_iter(OSQPWorkspace * work, c_int max_iter_new);


/**
 * Update absolute tolernace value
 * @param  work        Workspace
 * @param  eps_abs_new New absolute tolerance value
 * @return             Exitflag
 */
c_int osqp_update_eps_abs(OSQPWorkspace * work, c_float eps_abs_new);


/**
 * Update relative tolernace value
 * @param  work        Workspace
 * @param  eps_rel_new New relative tolerance value
 * @return             Exitflag
 */
c_int osqp_update_eps_rel(OSQPWorkspace * work, c_float eps_rel_new);


/**
 * Update relaxation parameter alpha
 * @param  work  Workspace
 * @param  alpha New relaxation parameter value
 * @return       Exitflag
 */
c_int osqp_update_alpha(OSQPWorkspace * work, c_float alpha_new);


/**
 * Update warm_start setting
 * @param  work           Workspace
 * @param  warm_start_new New warm_start setting
 * @return                Exitflag
 */
c_int osqp_update_warm_start(OSQPWorkspace * work, c_int warm_start_new);


#ifndef EMBEDDED

/**
 * Update regularization parameter in polishing
 * @param  work      Workspace
 * @param  delta_new New regularization parameter
 * @return           Exitflag
 */
c_int osqp_update_delta(OSQPWorkspace * work, c_float delta_new);


/**
 * Update polishing setting
 * @param  work          Workspace
 * @param  polishing_new New polishing setting
 * @return               Exitflag
 */
c_int osqp_update_polishing(OSQPWorkspace * work, c_int polishing_new);


/**
 * Update number of iterative refinement steps in polishing
 * @param  work                Workspace
 * @param  pol_refine_iter_new New iterative reginement steps
 * @return                     Exitflag
 */
c_int osqp_update_pol_refine_iter(OSQPWorkspace * work, c_int pol_refine_iter_new);


/**
 * Update verbose setting
 * @param  work        Workspace
 * @param  verbose_new New verbose setting
 * @return             Exitflag
 */
c_int osqp_update_verbose(OSQPWorkspace * work, c_int verbose_new);

#endif  // #ifndef EMBEDDED



#ifdef __cplusplus
}
#endif

#endif
