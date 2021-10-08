#include "osqp.h"
#include "auxil.h"
#include "proj.h"
#include "lin_alg.h"
#include "scaling.h"
#include "util.h"

/***********************************************************
* Auxiliary functions needed to compute ADMM iterations * *
***********************************************************/
#if EMBEDDED != 1

c_float compute_rho_estimate(const OSQPSolver *solver) {

  c_int   n, m;                         // Dimensions
  c_float prim_res, dual_res;           // Primal and dual residuals
  c_float prim_res_norm, dual_res_norm; // Normalization for the residuals
  c_float temp_res_norm;                // Temporary residual norm
  c_float rho_estimate;                 // Rho estimate value

  OSQPSettings*  settings = solver->settings;
  OSQPWorkspace* work     = solver->work;

  // Get problem dimensions
  n = work->data->n;
  m = work->data->m;

  // Get primal and dual residuals
  prim_res = work->scaled_prim_res;
  dual_res = work->scaled_dual_res;

  // Normalize primal residual
  prim_res_norm = OSQPVectorf_norm_inf(work->z);        // ||z||
  temp_res_norm = OSQPVectorf_norm_inf(work->Ax);       // ||Ax||
  prim_res_norm = c_max(prim_res_norm, temp_res_norm);  // max (||z||,||Ax||)
  prim_res     /= (prim_res_norm + OSQP_DIVISION_TOL);

  // Normalize dual residual
  dual_res_norm = OSQPVectorf_norm_inf(work->data->q);  // ||q||
  temp_res_norm = OSQPVectorf_norm_inf(work->Aty);      // ||A' y||
  dual_res_norm = c_max(dual_res_norm, temp_res_norm);
  temp_res_norm = OSQPVectorf_norm_inf(work->Px);       //  ||P x||
  dual_res_norm = c_max(dual_res_norm, temp_res_norm);  // max(||q||,||A' y||,||P x||)
  dual_res     /= (dual_res_norm + OSQP_DIVISION_TOL);

  // Return rho estimate
  rho_estimate = settings->rho * c_sqrt(prim_res / dual_res);
  rho_estimate = c_min(c_max(rho_estimate, RHO_MIN), RHO_MAX);

  return rho_estimate;
}

c_int adapt_rho(OSQPSolver *solver) {

  c_int   exitflag; // Exitflag
  c_float rho_new;  // New rho value

  OSQPInfo*      info     = solver->info;
  OSQPSettings*  settings = solver->settings;

  exitflag = 0;     // Initialize exitflag to 0

  // Compute new rho
  rho_new = compute_rho_estimate(solver);

  // Set rho estimate in info
  info->rho_estimate = rho_new;

  // Check if the new rho is large or small enough and update it in case
  if ((rho_new > settings->rho * settings->adaptive_rho_tolerance) ||
      (rho_new < settings->rho / settings->adaptive_rho_tolerance)) {
    exitflag                 = osqp_update_rho(solver, rho_new);
    info->rho_updates += 1;
  }

  return exitflag;
}

c_int set_rho_vec(OSQPSolver *solver) {

  c_int constr_types_changed = 0;

  OSQPSettings  *settings = solver->settings;
  OSQPWorkspace *work     = solver->work;

  settings->rho = c_min(c_max(settings->rho, RHO_MIN), RHO_MAX);

  constr_types_changed = OSQPVectorf_ew_bounds_type(work->constr_type,
                                                    work->data->l,
                                                    work->data->u,
                                                    RHO_TOL,
                                                    OSQP_INFTY * MIN_SCALING);


  //NB: Always refresh the complete rho vector, since the rho_vals
  //might be garbage if they have not been initialised yet.  This means
  //that there is some wasted effort in the case that the constraint types
  //haven't changed and the rho values are already correct, but such is life.
  OSQPVectorf_set_scalar_conditional(work->rho_vec,
                                     work->constr_type,
                                     RHO_MIN,                             //const  == -1
                                     settings->rho,                       //constr == 0
                                     RHO_EQ_OVER_RHO_INEQ*settings->rho); //constr == 1

  OSQPVectorf_ew_reciprocal(work->rho_inv_vec, work->rho_vec);

  return constr_types_changed;
}

c_int update_rho_vec(OSQPSolver *solver) {

  c_int constr_type_changed;
  c_int exitflag = 0;
  OSQPWorkspace* work = solver->work;

  //update rho_vec and see if anything changed
  constr_type_changed = set_rho_vec(solver);

  // Update rho_vec in KKT matrix if constraints type has changed
  if (constr_type_changed == 1) {
    exitflag = work->linsys_solver->update_rho_vec(work->linsys_solver, work->rho_vec, solver->settings->rho);
  }

  return exitflag;
}

#endif // EMBEDDED != 1


void swap_vectors(OSQPVectorf **a,
                  OSQPVectorf **b) {
  OSQPVectorf *temp;

  temp = *b;
  *b   = *a;
  *a   = temp;
}

static void compute_rhs(OSQPSolver *solver) {

  OSQPWorkspace* work     = solver->work;
  OSQPSettings*  settings = solver->settings;

  //part related to x variables
  OSQPVectorf_add_scaled(work->xtilde_view,
                         settings->sigma,work->x_prev,
                         -1., work->data->q);

  //part related to dual variable in the equality constrained QP (nu)
  if (settings->rho_is_vec) {
    OSQPVectorf_ew_prod(work->ztilde_view, work->rho_inv_vec, work->y);
    OSQPVectorf_add_scaled(work->ztilde_view,
                           -1.0, work->ztilde_view,
                           1.0, work->z_prev);
  }
  else {
    OSQPVectorf_add_scaled(work->ztilde_view,
                           1.0, work->z_prev,
                           -work->rho_inv, work->y);
  }
}

void update_xz_tilde(OSQPSolver *solver,
                     c_int       admm_iter) {

  OSQPWorkspace* work     = solver->work;

  // Compute right-hand side
  compute_rhs(solver);

  // Solve linear system
  work->linsys_solver->solve(work->linsys_solver, work->xz_tilde, admm_iter);
}

void update_x(OSQPSolver *solver) {

  OSQPSettings*  settings = solver->settings;
  OSQPWorkspace* work     = solver->work;

  // update x
  OSQPVectorf_add_scaled(work->x,
                         settings->alpha,work->xtilde_view,
                         (1.0 - settings->alpha),work->x_prev);

  // update delta_x
  OSQPVectorf_minus(work->delta_x,work->x,work->x_prev);
}

void update_z(OSQPSolver *solver) {

  OSQPSettings*  settings = solver->settings;
  OSQPWorkspace* work     = solver->work;

  // update z
  if (settings->rho_is_vec) {
    OSQPVectorf_ew_prod(work->z, work->rho_inv_vec,work->y);
    OSQPVectorf_add_scaled3(work->z,
                            1., work->z,
                            settings->alpha, work->ztilde_view,
                            (1.0 - settings->alpha), work->z_prev);
  }
  else {
    OSQPVectorf_add_scaled3(work->z,
                            settings->alpha, work->ztilde_view,
                            (1.0 - settings->alpha), work->z_prev,
                            work->rho_inv, work->y);
  }

  // project z
  project(work, work->z);

}

void update_y(OSQPSolver *solver) {

  OSQPSettings*  settings = solver->settings;
  OSQPWorkspace* work     = solver->work;

  OSQPVectorf_add_scaled3(work->delta_y,
                          settings->alpha, work->ztilde_view,
                          (1.0 - settings->alpha), work->z_prev,
                          -1.0, work->z);

  if (settings->rho_is_vec) {
    OSQPVectorf_ew_prod(work->delta_y, work->delta_y, work->rho_vec);
  }
  else {
    OSQPVectorf_mult_scalar(work->delta_y, settings->rho);
  }

  OSQPVectorf_plus(work->y, work->y, work->delta_y);

}

c_float compute_obj_val(const OSQPSolver  *solver,
                        const OSQPVectorf *x) {

  c_float obj_val;
  OSQPSettings*  settings = solver->settings;
  OSQPWorkspace* work     = solver->work;

  obj_val = OSQPMatrix_quad_form(work->data->P, x) +
            OSQPVectorf_dot_prod(work->data->q, x);

  if (settings->scaling) {
    obj_val *= work->scaling->cinv;
  }

  return obj_val;
}

c_float compute_prim_res(OSQPSolver        *solver,
                         const OSQPVectorf *x,
                         const OSQPVectorf *z) {

  // NB: Use z_prev as working vector
  // pr = Ax - z

  OSQPSettings*  settings = solver->settings;
  OSQPWorkspace* work     = solver->work;
  c_float prim_res;

  OSQPMatrix_Axpy(work->data->A,x,work->Ax, 1.0, 0.0); //Ax = A*x
  OSQPVectorf_minus(work->z_prev, work->Ax, z);

  work->scaled_prim_res = OSQPVectorf_norm_inf(work->z_prev);

  // If scaling active -> rescale residual
  if (settings->scaling && !settings->scaled_termination) {
    prim_res =  OSQPVectorf_scaled_norm_inf(work->scaling->Einv, work->z_prev);
  }
  else{
    prim_res  = work->scaled_prim_res;
  }
  return prim_res;
}

c_float compute_pri_tol(const OSQPSolver *solver,
                        c_float           eps_abs,
                        c_float           eps_rel) {

  c_float max_rel_eps, temp_rel_eps;
  OSQPSettings*  settings = solver->settings;
  OSQPWorkspace* work     = solver->work;

  // max_rel_eps = max(||z||, ||A x||)
  if (settings->scaling && !settings->scaled_termination) {
    // ||Einv * z||
    max_rel_eps =
    OSQPVectorf_scaled_norm_inf(work->scaling->Einv, work->z);

    // ||Einv * A * x||
    temp_rel_eps =
    OSQPVectorf_scaled_norm_inf(work->scaling->Einv, work->Ax);

    // Choose maximum
    max_rel_eps = c_max(max_rel_eps, temp_rel_eps);
  }

  else { // No unscaling required
    // ||z||
    max_rel_eps = OSQPVectorf_norm_inf(work->z);

    // ||A * x||
    temp_rel_eps = OSQPVectorf_norm_inf(work->Ax);

    // Choose maximum
    max_rel_eps = c_max(max_rel_eps, temp_rel_eps);
  }

  // eps_prim
  return eps_abs + eps_rel * max_rel_eps;
}

c_float compute_dual_res(OSQPSolver        *solver,
                         const OSQPVectorf *x,
                         const OSQPVectorf *y) {

  // NB: Use x_prev as temporary vector
  // NB: Only upper triangular part of P is stored.
  // dr = q + A'*y + P*x

  OSQPSettings*  settings = solver->settings;
  OSQPWorkspace* work     = solver->work;
  c_float dual_res;

  // dr = q
  OSQPVectorf_copy(work->x_prev,work->data->q);

  // P * x (upper triangular part)
  OSQPMatrix_Axpy(work->data->P, x, work->Px, 1.0, 0.0); //Px = P*x

  // dr += P * x (full P matrix)
  OSQPVectorf_plus(work->x_prev, work->x_prev, work->Px);

  // dr += A' * y
  if (work->data->m > 0) {
    OSQPMatrix_Atxpy(work->data->A, y, work->Aty, 1.0, 0.0); //Ax = A*x
    OSQPVectorf_plus(work->x_prev, work->x_prev, work->Aty);
  }

  work->scaled_dual_res = OSQPVectorf_norm_inf(work->x_prev);

  // If scaling active -> rescale residual
  if (settings->scaling && !settings->scaled_termination) {
    dual_res =  work->scaling->cinv * OSQPVectorf_scaled_norm_inf(work->scaling->Dinv,
                                                                  work->x_prev);
  }
  else {
    dual_res = work->scaled_dual_res;
  }

  return dual_res;
}

c_float compute_dua_tol(const OSQPSolver *solver,
                        c_float           eps_abs,
                        c_float           eps_rel) {

  c_float max_rel_eps, temp_rel_eps;
  OSQPSettings*  settings = solver->settings;
  OSQPWorkspace* work     = solver->work;

  // max_rel_eps = max(||q||, ||A' y|, ||P x||)
  if (settings->scaling && !settings->scaled_termination) {
    // || Dinv q||
    max_rel_eps =
    OSQPVectorf_scaled_norm_inf(work->scaling->Dinv,
                                work->data->q);

    // || Dinv A' y ||
    temp_rel_eps =
    OSQPVectorf_scaled_norm_inf(work->scaling->Dinv,
                                work->Aty);

    max_rel_eps = c_max(max_rel_eps, temp_rel_eps);

    // || Dinv P x||
    temp_rel_eps =
    OSQPVectorf_scaled_norm_inf(work->scaling->Dinv,
                                work->Px);

    max_rel_eps = c_max(max_rel_eps, temp_rel_eps);

    // Multiply by cinv
    max_rel_eps *= work->scaling->cinv;
  } else { // No scaling required
    // ||q||
    max_rel_eps = OSQPVectorf_norm_inf(work->data->q);

    // ||A'*y||
    temp_rel_eps = OSQPVectorf_norm_inf(work->Aty);
    max_rel_eps  = c_max(max_rel_eps, temp_rel_eps);

    // ||P*x||
    temp_rel_eps = OSQPVectorf_norm_inf(work->Px);
    max_rel_eps  = c_max(max_rel_eps, temp_rel_eps);
  }

  // eps_dual
  return eps_abs + eps_rel * max_rel_eps;
}

c_int is_primal_infeasible(OSQPSolver *solver,
                           c_float     eps_prim_inf) {

  /* This function checks the primal infeasibility criteria
     1) || A' * delta_y || < eps * ||delta_y||
     2) u'*max(delta_y, 0) + l'*min(delta_y, 0) < 0
   */

  c_float norm_delta_y;
  c_float ineq_lhs = 0.0;
  OSQPSettings*  settings = solver->settings;
  OSQPWorkspace* work     = solver->work;

  // Project delta_y onto the polar of the recession cone of [l,u]
  project_polar_reccone(work->delta_y,
                        work->data->l,
                        work->data->u,
                        OSQP_INFTY * MIN_SCALING);

  // Compute infinity norm of delta_y (unscale if necessary)
  if (settings->scaling && !settings->scaled_termination) {
    // Use work->Adelta_x as temporary vector
    OSQPVectorf_ew_prod(work->Adelta_x,
                        work->scaling->E,
                        work->delta_y);
    norm_delta_y = OSQPVectorf_norm_inf(work->Adelta_x);
  }
  else
    norm_delta_y = OSQPVectorf_norm_inf(work->delta_y);

  if (norm_delta_y > OSQP_DIVISION_TOL) {

    ineq_lhs  = OSQPVectorf_dot_prod_signed(work->data->u, work->delta_y, +1);
    ineq_lhs += OSQPVectorf_dot_prod_signed(work->data->l, work->delta_y, -1);

    /* Check if the condition is satisfied */
    if (ineq_lhs < 0.0) {
      OSQPMatrix_Atxpy(work->data->A, work->delta_y, work->Atdelta_y, 1.0, 0.0);

      // Unscale if necessary
      if (settings->scaling && !settings->scaled_termination) {
        OSQPVectorf_ew_prod(work->Atdelta_y,
                            work->Atdelta_y,
                            work->scaling->Dinv);
      }

      return OSQPVectorf_norm_inf(work->Atdelta_y) < eps_prim_inf * norm_delta_y;
    }
  }

  // Conditions not satisfied -> not primal infeasible
  return 0;
}

c_int is_dual_infeasible(OSQPSolver *solver,
                         c_float     eps_dual_inf) {

  /* This function checks the dual infeasibility criteria
     1) q * delta_x < eps * || delta_x ||
     2) ||P * delta_x || < eps * || delta_x ||
     3) (A * delta_x)_i > -eps * || delta_x ||,    l_i != -inf
        (A * delta_x)_i <  eps * || delta_x ||,    u_i != +inf
   */

  c_float norm_delta_x;
  c_float cost_scaling;
  OSQPSettings*  settings = solver->settings;
  OSQPWorkspace* work     = solver->work;

  // Compute norm of delta_x
  if (settings->scaling && !settings->scaled_termination) { // Unscale if needed

    norm_delta_x =
    OSQPVectorf_scaled_norm_inf(work->scaling->D,
                                work->delta_x);
    cost_scaling = work->scaling->c;
  }
  else {
    norm_delta_x = OSQPVectorf_norm_inf(work->delta_x);
    cost_scaling = 1.0;
  }

  // Prevent 0 division || delta_x || > 0
  if (norm_delta_x > OSQP_DIVISION_TOL) {
    // Normalize delta_x by its norm

    /* vec_mult_scalar(work->delta_x, 1./norm_delta_x, work->data->n); */

    // Check first if q'*delta_x < 0
    if (OSQPVectorf_dot_prod(work->data->q, work->delta_x) < 0.0) {
      // Compute product P * delta_x
      OSQPMatrix_Axpy(work->data->P, work->delta_x, work->Pdelta_x, 1.0, 0.0);

      // Scale if necessary
      if (settings->scaling && !settings->scaled_termination) {
        OSQPVectorf_ew_prod(work->Pdelta_x,
                            work->Pdelta_x,
                            work->scaling->Dinv);
      }

      // Check if || P * delta_x || = 0
      if (OSQPVectorf_norm_inf(work->Pdelta_x) <
          cost_scaling * eps_dual_inf * norm_delta_x) {
        // Compute A * delta_x
        OSQPMatrix_Axpy(work->data->A, work->delta_x, work->Adelta_x,1.0,0.0);

        // Scale if necessary
        if (settings->scaling && !settings->scaled_termination) {
          OSQPVectorf_ew_prod(work->Adelta_x, work->Adelta_x, work->scaling->Einv);
        }

        // De Morgan Law Applied to dual infeasibility conditions for A * x
        // NB: Note that MIN_SCALING is used to adjust the infinity value
        // in case the problem is scaled.

        //if you get this far, then all tests passed,
        //so return results from final test
        return test_in_reccone(work->Adelta_x,
                               work->data->l,
                               work->data->u,
                               OSQP_INFTY * MIN_SCALING,
                               eps_dual_inf * norm_delta_x);
      }
    }
  }

  // Conditions not satisfied -> not dual infeasible
  return 0;
}

c_int has_solution(const OSQPInfo * info){

  return ((info->status_val != OSQP_PRIMAL_INFEASIBLE) &&
      (info->status_val != OSQP_PRIMAL_INFEASIBLE_INACCURATE) &&
      (info->status_val != OSQP_DUAL_INFEASIBLE) &&
      (info->status_val != OSQP_DUAL_INFEASIBLE_INACCURATE) &&
      (info->status_val != OSQP_NON_CVX));

}

void store_solution(OSQPSolver *solver) {

#ifndef EMBEDDED
  c_float norm_vec;
#endif /* ifndef EMBEDDED */

  OSQPInfo*      info     = solver->info;
  OSQPSolution*  solution = solver->solution;
  OSQPSettings*  settings = solver->settings;
  OSQPWorkspace* work     = solver->work;


  if (has_solution(info)) {
    // Unscale solution if scaling has been performed
      if (settings->scaling){
          //use x_prev and z_prev as scratch space
          unscale_solution(work->x_prev,work->z_prev, //unscaled solution
                           work->x,work->y,           //scaled solution
                           work);
          OSQPVectorf_to_raw(solution->x, work->x_prev); // primal
          OSQPVectorf_to_raw(solution->y, work->z_prev); // dual
      }
      else{
          OSQPVectorf_to_raw(solution->x, work->x); // primal
          OSQPVectorf_to_raw(solution->y, work->y); // dual
      }
      /* Set infeasibility certificates to NaN */
      OSQPVectorf_set_scalar(work->delta_y, OSQP_NAN);
      OSQPVectorf_set_scalar(work->delta_x, OSQP_NAN);
      OSQPVectorf_to_raw(solution->prim_inf_cert, work->delta_y);
      OSQPVectorf_to_raw(solution->dual_inf_cert, work->delta_x);
  }

  else {

    // No solution present. Solution is NaN
    OSQPVectorf_set_scalar(work->x, OSQP_NAN);
    OSQPVectorf_set_scalar(work->y, OSQP_NAN);
    OSQPVectorf_to_raw(solution->x, work->x); // primal
    OSQPVectorf_to_raw(solution->y, work->y); // dual

    // reset iterates to 0 for next run (they cannot start from NaN)
    osqp_cold_start(solver);


#ifndef EMBEDDED

    // Normalize infeasibility certificates if embedded is off
    // NB: It requires a division
    if ((info->status_val == OSQP_PRIMAL_INFEASIBLE) ||
        ((info->status_val == OSQP_PRIMAL_INFEASIBLE_INACCURATE))) {
      norm_vec = OSQPVectorf_norm_inf(work->delta_y);
      OSQPVectorf_mult_scalar(work->delta_y, 1. / norm_vec);
      OSQPVectorf_to_raw(solution->prim_inf_cert, work->delta_y);

      /* Set dual infeasibility certificate to NaN */
      OSQPVectorf_set_scalar(work->delta_x, OSQP_NAN);
      OSQPVectorf_to_raw(solution->dual_inf_cert, work->delta_x);
    }

    if ((info->status_val == OSQP_DUAL_INFEASIBLE) ||
        ((info->status_val == OSQP_DUAL_INFEASIBLE_INACCURATE))) {
      norm_vec = OSQPVectorf_norm_inf(work->delta_x);
      OSQPVectorf_mult_scalar(work->delta_x, 1. / norm_vec);
      OSQPVectorf_to_raw(solution->dual_inf_cert, work->delta_x);

      /* Set primal infeasibility certificate to NaN */
      OSQPVectorf_set_scalar(work->delta_y, OSQP_NAN);
      OSQPVectorf_to_raw(solution->prim_inf_cert, work->delta_y);
    }

#endif /* ifndef EMBEDDED */
  }
}

void update_info(OSQPSolver *solver,
                 c_int       iter,
                 c_int       compute_objective,
                 c_int       polishing) {

  OSQPVectorf *x, *z, *y;                   // Allocate pointers to vectors
  c_float *obj_val, *prim_res, *dual_res;   // objective value, residuals

  OSQPInfo*      info     = solver->info;
  OSQPWorkspace* work     = solver->work;

#ifdef PROFILING
  c_float *run_time;                    // Execution time
#endif /* ifdef PROFILING */

#ifndef EMBEDDED

  if (polishing) {
    x        = work->pol->x;
    y        = work->pol->y;
    z        = work->pol->z;
    obj_val  = &work->pol->obj_val;
    prim_res = &work->pol->prim_res;
    dual_res = &work->pol->dual_res;
# ifdef PROFILING
    run_time = &info->polish_time;
# endif /* ifdef PROFILING */
  }
  else {
#endif // EMBEDDED
    x          = work->x;
    y          = work->y;
    z          = work->z;
    obj_val    = &info->obj_val;
    prim_res   = &info->prim_res;
    dual_res   = &info->dual_res;
    info->iter = iter;
#ifdef PROFILING
    run_time   = &info->solve_time;
#endif /* ifdef PROFILING */
#ifndef EMBEDDED
}

#endif /* ifndef EMBEDDED */


  // Compute the objective if needed
  if (compute_objective) {
    *obj_val = compute_obj_val(solver, x);
  }

  // Compute primal residual
  if (work->data->m == 0) {
    // No constraints -> Always primal feasible
    *prim_res = 0.;
  } else {
    *prim_res = compute_prim_res(solver, x, z);
  }

  // Compute dual residual
  *dual_res = compute_dual_res(solver, x, y);

  // Update timing
#ifdef PROFILING
  *run_time = osqp_toc(work->timer);
#endif /* ifdef PROFILING */

#ifdef PRINTING
  work->summary_printed = 0; // The just updated info have not been printed
#endif /* ifdef PRINTING */
}


void reset_info(OSQPInfo *info) {
#ifdef PROFILING

  // Initialize info values.
  info->solve_time = 0.0;  // Solve time to zero
# ifndef EMBEDDED
  info->polish_time = 0.0; // Polish time to zero
# endif /* ifndef EMBEDDED */

  // NB: We do not reset the setup_time because it is performed only once
#endif /* ifdef PROFILING */

  update_status(info, OSQP_UNSOLVED); // Problem is unsolved

#if EMBEDDED != 1
  info->rho_updates = 0;              // Rho updates are now 0
#endif /* if EMBEDDED != 1 */
}

void update_status(OSQPInfo *info,
                   c_int     status_val) {

  // Update status value
  info->status_val = status_val;

  // Update status string depending on status val
  if (status_val == OSQP_SOLVED) c_strcpy(info->status, "solved");

  if (status_val == OSQP_SOLVED_INACCURATE) c_strcpy(info->status,
                                                     "solved inaccurate");
  else if (status_val == OSQP_PRIMAL_INFEASIBLE) c_strcpy(info->status,
                                                          "primal infeasible");
  else if (status_val == OSQP_PRIMAL_INFEASIBLE_INACCURATE) c_strcpy(info->status,
                                                                     "primal infeasible inaccurate");
  else if (status_val == OSQP_UNSOLVED) c_strcpy(info->status, "unsolved");
  else if (status_val == OSQP_DUAL_INFEASIBLE) c_strcpy(info->status,
                                                        "dual infeasible");
  else if (status_val == OSQP_DUAL_INFEASIBLE_INACCURATE) c_strcpy(info->status,
                                                                   "dual infeasible inaccurate");
  else if (status_val == OSQP_MAX_ITER_REACHED) c_strcpy(info->status,
                                                         "maximum iterations reached");
#ifdef PROFILING
  else if (status_val == OSQP_TIME_LIMIT_REACHED) c_strcpy(info->status,
                                                           "run time limit reached");
#endif /* ifdef PROFILING */
  else if (status_val == OSQP_SIGINT) c_strcpy(info->status, "interrupted");

  else if (status_val == OSQP_NON_CVX) c_strcpy(info->status, "problem non convex");

}

c_int check_termination(OSQPSolver *solver,
                        c_int       approximate) {

  c_float eps_prim, eps_dual, eps_prim_inf, eps_dual_inf;
  c_int   exitflag;
  c_int   prim_res_check, dual_res_check, prim_inf_check, dual_inf_check;
  c_float eps_abs, eps_rel;

  OSQPInfo*      info     = solver->info;
  OSQPSettings*  settings = solver->settings;
  OSQPWorkspace* work     = solver->work;

  // Initialize variables to 0
  exitflag       = 0;
  prim_res_check = 0; dual_res_check = 0;
  prim_inf_check = 0; dual_inf_check = 0;

  // Initialize tolerances
  eps_abs      = settings->eps_abs;
  eps_rel      = settings->eps_rel;
  eps_prim_inf = settings->eps_prim_inf;
  eps_dual_inf = settings->eps_dual_inf;

  // If residuals are too large, the problem is probably non convex
  if ((info->prim_res > OSQP_INFTY) ||
      (info->dual_res > OSQP_INFTY)){
    // Looks like residuals are diverging. Probably the problem is non convex!
    // Terminate and report it
    update_status(info, OSQP_NON_CVX);
    info->obj_val = OSQP_NAN;
    return 1;
  }

  // If approximate solution required, increase tolerances by 10
  if (approximate) {
    eps_abs      *= 10;
    eps_rel      *= 10;
    eps_prim_inf *= 10;
    eps_dual_inf *= 10;
  }

  // Check residuals
  if (work->data->m == 0) {
    prim_res_check = 1; // No constraints -> Primal feasibility always satisfied
  }
  else {
    // Compute primal tolerance
    eps_prim = compute_pri_tol(solver, eps_abs, eps_rel);

    // Primal feasibility check
    if (info->prim_res < eps_prim) {
      prim_res_check = 1;
    } else {
      // Primal infeasibility check
      prim_inf_check = is_primal_infeasible(solver, eps_prim_inf);
    }
  } // End check if m == 0

  // Compute dual tolerance
  eps_dual = compute_dua_tol(solver, eps_abs, eps_rel);

  // Dual feasibility check
  if (info->dual_res < eps_dual) {
    dual_res_check = 1;
  } else {
    // Check dual infeasibility
    dual_inf_check = is_dual_infeasible(solver, eps_dual_inf);
  }

  // Compare checks to determine solver status
  if (prim_res_check && dual_res_check) {
    // Update final information
    if (approximate) {
      update_status(info, OSQP_SOLVED_INACCURATE);
    } else {
      update_status(info, OSQP_SOLVED);
    }
    exitflag = 1;
  }
  else if (prim_inf_check) {
    // Update final information
    if (approximate) {
      update_status(info, OSQP_PRIMAL_INFEASIBLE_INACCURATE);
    } else {
      update_status(info, OSQP_PRIMAL_INFEASIBLE);
    }

    if (settings->scaling && !settings->scaled_termination) {
      // Update infeasibility certificate
      OSQPVectorf_ew_prod(work->delta_y,
                          work->delta_y,
                          work->scaling->E);
    }
    info->obj_val = OSQP_INFTY;
    exitflag            = 1;
  }
  else if (dual_inf_check) {
    // Update final information
    if (approximate) {
      update_status(info, OSQP_DUAL_INFEASIBLE_INACCURATE);
    } else {
      update_status(info, OSQP_DUAL_INFEASIBLE);
    }

    if (settings->scaling && !settings->scaled_termination) {
      // Update infeasibility certificate
      OSQPVectorf_ew_prod(work->delta_x,
                          work->delta_x,
                          work->scaling->D);
    }
    info->obj_val = -OSQP_INFTY;
    exitflag            = 1;
  }

  return exitflag;
}


#ifndef EMBEDDED

c_int validate_data(const csc     *P,
                    const c_float *q,
                    const csc     *A,
                    const c_float *l,
                    const c_float *u,
                    c_int          m,
                    c_int          n) {
  c_int j, ptr;

  if (!P) {
# ifdef PRINTING
    c_eprint("Missing quadratic cost matrix P");
# endif
    return 1;
  }

  if (!A) {
# ifdef PRINTING
    c_eprint("Missing constraint matrix A");
# endif
    return 1;
  }

  if (!q) {
# ifdef PRINTING
    c_eprint("Missing linear cost vector q");
# endif
    return 1;
  }

  // General dimensions Tests
  if ((n <= 0) || (m < 0)) {
# ifdef PRINTING
    c_eprint("n must be positive and m nonnegative; n = %i, m = %i",
             (int)n, (int)m);
# endif /* ifdef PRINTING */
    return 1;
  }

  // Matrix P
  if (P->m != n) {
# ifdef PRINTING
    c_eprint("P does not have dimension n x n with n = %i", (int)n);
# endif /* ifdef PRINTING */
    return 1;
  }

  if (P->m != P->n) {
# ifdef PRINTING
    c_eprint("P is not square");
# endif /* ifdef PRINTING */
    return 1;
  }

  for (j = 0; j < n; j++) { // COLUMN
    for (ptr = P->p[j]; ptr < P->p[j + 1]; ptr++) {
      if (P->i[ptr] > j) {  // if ROW > COLUMN
# ifdef PRINTING
        c_eprint("P is not upper triangular");
# endif /* ifdef PRINTING */
        return 1;
      }
    }
  }

  // Matrix A
  if ((A->m != m) || (A->n != n)) {
# ifdef PRINTING
    c_eprint("A does not have dimension %i x %i", (int)m, (int)n);
# endif /* ifdef PRINTING */
    return 1;
  }

  // Lower and upper bounds
  for (j = 0; j < m; j++) {
    if (l[j] > u[j]) {
# ifdef PRINTING
      c_eprint("Lower bound at index %d is greater than upper bound: %.4e > %.4e",
               (int)j, l[j], u[j]);
# endif /* ifdef PRINTING */
      return 1;
    }
  }

  return 0;
}

#endif // #ifndef EMBEDDED


c_int validate_linsys_solver(c_int linsys_solver) {

#ifdef CUDA_SUPPORT

  if (linsys_solver != CUDA_PCG_SOLVER) {
    return 1;
  }

#else /* ifdef CUDA_SUPPORT */

  if ((linsys_solver != QDLDL_SOLVER) &&
      (linsys_solver != MKL_PARDISO_SOLVER)) {
    return 1;
  }

#endif /* ifdef CUDA_SUPPORT */

  // TODO: Add more solvers in case

  // Valid solver
  return 0;
}


c_int validate_settings(const OSQPSettings *settings,
                        c_int               from_setup) {

  if (!settings) {
# ifdef PRINTING
    c_eprint("Missing settings!");
# endif /* ifdef PRINTING */
    return 1;
  }

  if (from_setup &&
      validate_linsys_solver(settings->linsys_solver)) {
# ifdef PRINTING
    c_eprint("linsys_solver not recognized");
# endif /* ifdef PRINTING */
    return 1;
  }

  if ((settings->verbose != 0) &&
      (settings->verbose != 1)) {
# ifdef PRINTING
    c_eprint("verbose must be either 0 or 1");
# endif /* ifdef PRINTING */
    return 1;
  }

  if ((settings->warm_starting != 0) &&
      (settings->warm_starting != 1)) {
# ifdef PRINTING
    c_eprint("warm_start must be either 0 or 1");
# endif /* ifdef PRINTING */
    return 1;
  }

  if (from_setup && settings->scaling < 0) {
# ifdef PRINTING
    c_eprint("scaling must be nonnegative");
# endif /* ifdef PRINTING */
    return 1;
  }

  if ((settings->polishing != 0) &&
      (settings->polishing != 1)) {
# ifdef PRINTING
    c_eprint("polishing must be either 0 or 1");
# endif /* ifdef PRINTING */
    return 1;
  }

  if (from_setup && settings->rho <= 0.0) {
# ifdef PRINTING
    c_eprint("rho must be positive");
# endif /* ifdef PRINTING */
    return 1;
  }

  if (from_setup &&
      (settings->rho_is_vec != 0) &&
      (settings->rho_is_vec != 1)) {
# ifdef PRINTING
    c_eprint("rho_is_vec must be either 0 or 1");
# endif /* ifdef PRINTING */
    return 1;
  }

  if (from_setup && settings->sigma <= 0.0) {
# ifdef PRINTING
    c_eprint("sigma must be positive");
# endif /* ifdef PRINTING */
    return 1;
  }

  if ((settings->alpha <= 0.0) ||
      (settings->alpha >= 2.0)) {
# ifdef PRINTING
    c_eprint("alpha must be strictly between 0 and 2");
# endif /* ifdef PRINTING */
    return 1;
  }

  if (from_setup &&
      (settings->adaptive_rho != 0) &&
      (settings->adaptive_rho != 1)) {
# ifdef PRINTING
    c_eprint("adaptive_rho must be either 0 or 1");
# endif /* ifdef PRINTING */
    return 1;
  }

  if (from_setup && settings->adaptive_rho_interval < 0) {
# ifdef PRINTING
    c_eprint("adaptive_rho_interval must be nonnegative");
# endif /* ifdef PRINTING */
    return 1;
  }

  if (from_setup && settings->adaptive_rho_fraction <= 0) {
#  ifdef PRINTING
    c_eprint("adaptive_rho_fraction must be positive");
#  endif /* ifdef PRINTING */
    return 1;
  }

  if (from_setup && settings->adaptive_rho_tolerance < 1.0) {
# ifdef PRINTING
    c_eprint("adaptive_rho_tolerance must be >= 1");
# endif /* ifdef PRINTING */
    return 1;
  }

  if (settings->max_iter <= 0) {
# ifdef PRINTING
    c_eprint("max_iter must be positive");
# endif /* ifdef PRINTING */
    return 1;
  }

  if (settings->eps_abs < 0.0) {
# ifdef PRINTING
    c_eprint("eps_abs must be nonnegative");
# endif /* ifdef PRINTING */
    return 1;
  }

  if (settings->eps_rel < 0.0) {
# ifdef PRINTING
    c_eprint("eps_rel must be nonnegative");
# endif /* ifdef PRINTING */
    return 1;
  }

  if ((settings->eps_rel == 0.0) &&
      (settings->eps_abs == 0.0)) {
# ifdef PRINTING
    c_eprint("at least one of eps_abs and eps_rel must be positive");
# endif /* ifdef PRINTING */
    return 1;
  }

  if (settings->eps_prim_inf <= 0.0) {
# ifdef PRINTING
    c_eprint("eps_prim_inf must be positive");
# endif /* ifdef PRINTING */
    return 1;
  }

  if (settings->eps_dual_inf <= 0.0) {
# ifdef PRINTING
    c_eprint("eps_dual_inf must be positive");
# endif /* ifdef PRINTING */
    return 1;
  }

  if ((settings->scaled_termination != 0) &&
      (settings->scaled_termination != 1)) {
# ifdef PRINTING
    c_eprint("scaled_termination must be either 0 or 1");
# endif /* ifdef PRINTING */
    return 1;
  }

  if (settings->check_termination < 0) {
# ifdef PRINTING
    c_eprint("check_termination must be nonnegative");
# endif /* ifdef PRINTING */
    return 1;
  }

  if (settings->time_limit <= 0.0) {
#  ifdef PRINTING
    c_eprint("time_limit must be positive\n");
#  endif /* ifdef PRINTING */
    return 1;
  }

  if (settings->delta <= 0.0) {
# ifdef PRINTING
    c_eprint("delta must be positive");
# endif /* ifdef PRINTING */
    return 1;
  }
  
  if (settings->polish_refine_iter < 0) {
# ifdef PRINTING
    c_eprint("polish_refine_iter must be nonnegative");
# endif /* ifdef PRINTING */
    return 1;
  }

  return 0;
}
