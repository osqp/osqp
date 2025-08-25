#include "osqp.h"
#include "auxil.h"
#include "lin_alg.h"
#include "scaling.h"
#include "util.h"
#include "printing.h"
#include "timing.h"
#include <math.h>

/***********************************************************
* Auxiliary functions needed to compute ADMM iterations * *
***********************************************************/
#if OSQP_EMBEDDED_MODE != 1

OSQPFloat compute_rho_estimate(const OSQPSolver* solver) {

  OSQPFloat prim_res, dual_res;           // Primal and dual residuals
  OSQPFloat prim_res_norm, dual_res_norm; // Normalization for the residuals
  OSQPFloat temp_res_norm;                // Temporary residual norm
  OSQPFloat rho_estimate;                 // Rho estimate value

  OSQPSettings*  settings = solver->settings;
  OSQPWorkspace* work     = solver->work;

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
  if (settings->pid_controller) {
    if (settings->pid_controller_sqrt){
      work->rho_ratio = sqrt((prim_res / dual_res) - 1.);
      work->rho_delta += work->rho_ratio;
      work->rho_sum += work->rho_ratio;
      rho_estimate = log(settings->rho) - (
        settings->KP * work->rho_ratio + settings->KI * work->rho_sum + settings->KD * (work->rho_delta)
      );
      rho_estimate = exp(rho_estimate);
    }
    else if (settings->pid_controller_sqrt_mult) {
      work->rho_ratio = sqrt((prim_res / dual_res) - 1.);
      work->rho_delta += work->rho_ratio;
      work->rho_sum += work->rho_ratio;
      rho_estimate = settings->rho * (
        settings->KP * work->rho_ratio + settings->KI * work->rho_sum + settings->KD * (work->rho_delta)
      );
    }
    else if (settings->pid_controller_sqrt_mult_2) {
      work->rho_ratio = sqrt((prim_res / dual_res));
      work->rho_delta += work->rho_ratio;
      work->rho_sum += work->rho_ratio;
      rho_estimate = settings->rho * (
        settings->KP * work->rho_ratio + settings->KI * work->rho_sum + settings->KD * (work->rho_delta)
      );
    }
    else if (settings->pid_controller_log) {
      work->rho_ratio = log((prim_res / dual_res) - 1.);
      work->rho_delta += work->rho_ratio;
      work->rho_sum += work->rho_ratio;
      rho_estimate = log(settings->rho) - (
        settings->KP * work->rho_ratio + settings->KI * work->rho_sum + settings->KD * (work->rho_delta)
      );
      rho_estimate = exp(rho_estimate);
    }
    else {
      work->rho_ratio = (prim_res / dual_res) - 1.;
      work->rho_delta += work->rho_ratio;
      work->rho_sum += work->rho_ratio;
      rho_estimate = log(settings->rho) - (
        settings->KP * work->rho_ratio + settings->KI * work->rho_sum + settings->KD * (work->rho_delta)
      );
      rho_estimate = exp(rho_estimate);
    }
  }
  else {
    rho_estimate = settings->rho * c_sqrt(prim_res / dual_res);
  }
  rho_estimate = c_min(c_max(rho_estimate, OSQP_RHO_MIN), OSQP_RHO_MAX);

  return rho_estimate;
}

OSQPInt adapt_rho(OSQPSolver* solver) {

  OSQPInt   exitflag; // Exitflag
  OSQPFloat rho_new;  // New rho value

  OSQPSettings*  settings = solver->settings;
  OSQPWorkspace* work     = solver->work;
  OSQPInfo*      info     = solver->info;

  exitflag = 0;     // Initialize exitflag to 0
  // c_print("adaptive_rho_tolerance_greater %f\n", settings->adaptive_rho_tolerance_greater);
  // c_print("adaptive_rho_tolerance_less %f\n", settings->adaptive_rho_tolerance_less);

  // Compute new rho
  rho_new = compute_rho_estimate(solver);

  // Set rho estimate in info
  info->rho_estimate = rho_new;

  // Check if the new rho is large or small enough and update it in case
  if (settings->rho_custom_condition) {
    // c_print("rho_custom_condition %d\n", settings->rho_custom_condition);
    // c_print("rho_custom_tolerance %f\n", settings->rho_custom_tolerance);
    OSQPFloat first_part_norm = 0.0;
    OSQPFloat second_part_norm = 0.0;

    OSQPVectorf_minus(work->delta_x, work->x, work->x_prev);
    OSQPVectorf_minus(work->delta_v, work->v, work->v_prev);

    first_part_norm = OSQPVectorf_norm_2(work->delta_x);
    second_part_norm = OSQPVectorf_norm_2(work->delta_v);

    if ((rho_new > settings->rho * settings->adaptive_rho_tolerance_greater) ||
        (rho_new < settings->rho / settings->adaptive_rho_tolerance_less) ||
        (settings->rho_custom_tolerance < first_part_norm / second_part_norm)) {
      exitflag                 = osqp_update_rho(solver, rho_new);
      info->rho_updates += 1;
      solver->work->rho_updated = 1;

      if (settings->pid_controller) {
        work->rho_delta = -work->rho_ratio;
      }
    }
    else if (settings->pid_controller) {
      work->rho_sum -= work->rho_ratio;
      work->rho_delta -= work->rho_ratio;
    }
  }
  if ((rho_new > settings->rho * settings->adaptive_rho_tolerance_greater) ||
      (rho_new < settings->rho / settings->adaptive_rho_tolerance_less)) {
    exitflag                 = osqp_update_rho(solver, rho_new);
    info->rho_updates += 1;
    solver->work->rho_updated = 1;

    if (settings->pid_controller) {
      work->rho_delta = -work->rho_ratio;
    }
  }
  else if (settings->pid_controller) {
    work->rho_sum -= work->rho_ratio;
    work->rho_delta -= work->rho_ratio;
  }

  return exitflag;
}

OSQPInt set_rho_vec(OSQPSolver* solver) {

  OSQPInt constr_types_changed = 0;

  OSQPSettings*  settings = solver->settings;
  OSQPWorkspace* work     = solver->work;

  settings->rho = c_min(c_max(settings->rho, OSQP_RHO_MIN), OSQP_RHO_MAX);

  constr_types_changed = OSQPVectorf_ew_bounds_type(work->constr_type,
                                                    work->data->l,
                                                    work->data->u,
                                                    OSQP_RHO_TOL,
                                                    OSQP_INFTY * OSQP_MIN_SCALING);


  //NB: Always refresh the complete rho vector, since the rho_vals
  //might be garbage if they have not been initialised yet.  This means
  //that there is some wasted effort in the case that the constraint types
  //haven't changed and the rho values are already correct, but such is life.
  OSQPVectorf_set_scalar_conditional(work->rho_vec,
                                     work->constr_type,
                                     OSQP_RHO_MIN,                             //const  == -1
                                     settings->rho,                       //constr == 0
                                     OSQP_RHO_EQ_OVER_RHO_INEQ*settings->rho); //constr == 1

  OSQPVectorf_ew_reciprocal(work->rho_inv_vec, work->rho_vec);

  return constr_types_changed;
}

OSQPInt update_rho_vec(OSQPSolver* solver) {

  OSQPInt constr_type_changed;
  OSQPInt exitflag = 0;
  OSQPWorkspace* work = solver->work;

  //update rho_vec and see if anything changed
  constr_type_changed = set_rho_vec(solver);

  // Update rho_vec in KKT matrix if constraints type has changed
  if (constr_type_changed == 1) {
    exitflag = work->linsys_solver->update_rho_vec(work->linsys_solver, work->rho_vec, solver->settings->rho);
  }

  return exitflag;
}

#endif // OSQP_EMBEDDED_MODE != 1


void swap_vectors(OSQPVectorf** a,
                  OSQPVectorf** b) {
  OSQPVectorf* temp;

  temp = *b;
  *b   = *a;
  *a   = temp;
}

static void compute_rhs(OSQPSolver* solver) {

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

void update_xz_tilde(OSQPSolver* solver,
                     OSQPInt     admm_iter) {

  OSQPWorkspace* work = solver->work;

  // Compute right-hand side
  compute_rhs(solver);

  // Solve linear system
  work->linsys_solver->solve(work->linsys_solver, work->xz_tilde, admm_iter);
}

void update_x(OSQPSolver* solver) {

  OSQPSettings*  settings = solver->settings;
  OSQPWorkspace* work     = solver->work;

  // update x
  OSQPVectorf_add_scaled(work->x,
                         settings->alpha,work->xtilde_view,
                         (1.0 - settings->alpha),work->x_prev);

  // update delta_x
  OSQPVectorf_minus(work->delta_x,work->x,work->x_prev);
}

void update_z(OSQPSolver* solver) {

  OSQPSettings*  settings = solver->settings;
  OSQPWorkspace* work     = solver->work;

  // update z
  if (settings->rho_is_vec) {
    OSQPVectorf_ew_prod(work->z, work->rho_inv_vec, work->y);
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

  // Update v
  OSQPVectorf_copy(work->v, work->z);

  // project z onto C = [l,u]
  OSQPVectorf_ew_bound_vec(work->z, work->z, work->data->l, work->data->u);
}

void update_y(OSQPSolver* solver) {

  OSQPSettings*  settings = solver->settings;
  OSQPWorkspace* work     = solver->work;

  // OSQPFloat* y_data = OSQPVectorf_data(work->y);
  // c_print("Update_y First element of y: %f\n", y_data[0]);

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

  // y_data = OSQPVectorf_data(work->y);
  // c_print("Update_y First element of y (2): %f\n", y_data[0]);
  // c_print("settings->rho: %f\n", settings->rho);
  // OSQPFloat* rho_vec_data = OSQPVectorf_data(work->rho_vec);
  // c_print("First element of rho_vec: %f\n", rho_vec_data[0]); 
  // OSQPFloat* rho_vec_data = OSQPVectorf_data(work->rho_vec);
  // OSQPInt rho_vec_length = OSQPVectorf_length(work->rho_vec);
  // c_print("rho_vec (length %d): ", rho_vec_length);
  // for (OSQPInt i = 0; i < rho_vec_length; i++) {
  //     c_print("  [%d]: %f\n", i, rho_vec_data[i]);
  // }

  if (strcmp(settings->restart_type, "averaged") == 0) {
    // y_pred^{k+1} = y^{k+1} + \rho (z^{k+1} - z^{k}) + \rho (1 - \alpha) (\tilde{z}^{k+1} - z^{k})
    // I don't know of a better way to do this other than the following three updates
    OSQPVectorf_add_scaled3(
      work->y_pred, 1., work->z, (settings->alpha - 2.), work->z_prev,
      (1. - settings->alpha), work->ztilde_view
    );
    if (settings->rho_is_vec) {
      OSQPVectorf_ew_prod(work->y_pred, work->y_pred, work->rho_vec);
    }
    else {
      OSQPVectorf_mult_scalar(work->y_pred, settings->rho);
    }
    OSQPVectorf_plus(work->y_pred, work->y_pred, work->y);
  }
}

void update_w(OSQPSolver* solver) {
  
  OSQPSettings*  settings = solver->settings;
  OSQPWorkspace* work     = solver->work;

  OSQPVectorf_add_scaled3(work->delta_w,
                          settings->alpha, work->xtilde_view,
                          (1.0 - settings->alpha), work->x_prev,
                          -1.0, work->x);

  OSQPVectorf_mult_scalar(work->delta_w, settings->sigma);

  OSQPVectorf_plus(work->w, work->w, work->delta_w);

  // w_pred^{k+1} = w^{k+1} + \sigma (x^{k+1} - x^{k}) + \sigma (1 - \alpha) (\tilde{x}^{k+1} - x^{k})
  // I don't know of a better way to do this other than the following three updates
  OSQPVectorf_add_scaled3(
    work->w_pred, 1., work->x, (settings->alpha - 2.), work->x_prev,
    (1. - settings->alpha), work->xtilde_view
  );
  OSQPVectorf_mult_scalar(work->w_pred, settings->sigma);
  OSQPVectorf_plus(work->w_pred, work->w_pred, work->w);
}

void reset_sum(OSQPSolver* solver) {
  OSQPWorkspace* work     = solver->work;

  OSQPVectorf_set_scalar(work->sum_x, 0.0);
  OSQPVectorf_set_scalar(work->sum_y, 0.0);
  OSQPVectorf_set_scalar(work->sum_z, 0.0);
  OSQPVectorf_set_scalar(work->sum_xz_tilde, 0.0);
  OSQPVectorf_set_scalar(work->sum_y_pred, 0.0);
  OSQPVectorf_set_scalar(work->sum_w_pred, 0.0);
  OSQPVectorf_set_scalar(work->sum_w, 0.0);
  OSQPVectorf_set_scalar(work->sum_v, 0.0);
}

void reset_avg(OSQPSolver* solver) {
  OSQPWorkspace* work     = solver->work;

  OSQPVectorf_set_scalar(work->avg_x, 0.0);
  OSQPVectorf_set_scalar(work->avg_y, 0.0);
  OSQPVectorf_set_scalar(work->avg_z, 0.0);
  OSQPVectorf_set_scalar(work->avg_xz_tilde, 0.0);
  OSQPVectorf_set_scalar(work->avg_y_pred, 0.0);
  OSQPVectorf_set_scalar(work->avg_w_pred, 0.0);
  OSQPVectorf_set_scalar(work->avg_w, 0.0);
  OSQPVectorf_set_scalar(work->avg_v, 0.0);
}

void sum_add(OSQPSolver* solver, OSQPFloat weight) {
  OSQPWorkspace* work     = solver->work;

  OSQPVectorf_add_scaled(work->sum_x, 1., work->sum_x,  weight, work->x);
  OSQPVectorf_add_scaled(work->sum_z, 1., work->sum_z, weight, work->z);
  OSQPVectorf_add_scaled(work->sum_xz_tilde, 1., work->sum_xz_tilde, weight, work->xz_tilde);
  OSQPVectorf_add_scaled(work->sum_y, 1., work->sum_y, weight, work->y);
  OSQPVectorf_add_scaled(work->sum_w, 1., work->sum_w, weight, work->w);
  OSQPVectorf_add_scaled(work->sum_v, 1., work->sum_v, weight, work->v);
  OSQPVectorf_add_scaled(work->sum_y_pred, 1., work->sum_y_pred, weight, work->y_pred);
  OSQPVectorf_add_scaled(work->sum_w_pred, 1., work->sum_w_pred, weight, work->w_pred);
}

void update_average(OSQPSolver* solver) {
  OSQPWorkspace* work = solver->work;
  OSQPInfo* info      = solver->info;

  OSQPFloat scalling          = (info->inner_loop_iter - 1.) / (info->inner_loop_iter);
  OSQPFloat reverse_scalling  = (1. - scalling);

  OSQPVectorf_add_scaled(work->avg_x, scalling, work->avg_x, reverse_scalling, work->x);
  OSQPVectorf_add_scaled(work->avg_z, scalling, work->avg_z, reverse_scalling, work->z);
  OSQPVectorf_add_scaled(work->avg_xz_tilde, scalling, work->avg_xz_tilde, reverse_scalling, work->xz_tilde);
  OSQPVectorf_add_scaled(work->avg_y, scalling, work->avg_y, reverse_scalling, work->y);
  OSQPVectorf_add_scaled(work->avg_w, scalling, work->avg_w, reverse_scalling, work->w);
  OSQPVectorf_add_scaled(work->avg_v, scalling, work->avg_v, reverse_scalling, work->v);
  OSQPVectorf_add_scaled(work->avg_y_pred, scalling, work->avg_y_pred, reverse_scalling, work->y_pred);
  OSQPVectorf_add_scaled(work->avg_w_pred, scalling, work->avg_w_pred, reverse_scalling, work->w_pred);
}

void restart_to_average(OSQPSolver* solver) {
  OSQPSettings*  settings = solver->settings;
  OSQPWorkspace* work     = solver->work;

  if (settings->custom_average_rest) {
    // c_print("custom_average_rest %d\n", settings->custom_average_rest);
    // I temporarly use OSQPVectorf_copy() instead of swap_vectors(), to prevent errors
    OSQPVectorf_ew_prod(work->v, work->rho_inv_vec, work->avg_y_pred);
    OSQPVectorf_plus(work->v, work->v, work->avg_z);

    OSQPVectorf_ew_bound_vec(work->z, work->v, work->data->l, work->data->u);

    OSQPVectorf_minus(work->y, work->v, work->z);
    OSQPVectorf_ew_prod(work->y, work->rho_vec, work->y);

    OSQPVectorf_add_scaled(work->x, (1. / settings->sigma), work->avg_w_pred, 1., work->avg_x);
    
    OSQPVectorf_mult_scalar(work->w, 0.0);

    OSQPVectorf_copy(work->xz_tilde, work->avg_xz_tilde);
  }
  else {
    // I temporarly use OSQPVectorf_copy() instead of swap_vectors(), to prevent errors
    OSQPVectorf_copy(work->x, work->avg_x);
    OSQPVectorf_copy(work->z, work->avg_z);
    OSQPVectorf_copy(work->xz_tilde, work->avg_xz_tilde);
    OSQPVectorf_copy(work->y, work->avg_y_pred);
    OSQPVectorf_copy(work->w, work->avg_w_pred);

    OSQPVectorf_ew_prod(work->v, work->rho_inv_vec, work->y);
    OSQPVectorf_plus(work->v, work->v, work->z);
  }
}

void reflected_halpern_step(OSQPSolver* solver, OSQPFloat scalling) {
  OSQPSettings*  settings = solver->settings;
  OSQPWorkspace* work     = solver->work;

  OSQPFloat lambd_plus_one  = 1. + settings->lambd;

  // c_print("lambd %f\n", settings->lambd);

  if (settings->alpha_adjustment_reflected_halpern) {
    // c_print("alpha_adjustment_reflected_halpern %d\n", settings->alpha_adjustment_reflected_halpern);
    // 2 / alpha
    // OSQPFloat alpha_adjustment = 2. / settings->alpha;
    OSQPFloat alpha_adjustment = lambd_plus_one / settings->alpha;
    // ((2 / alpha) - 1) * [(k + 1) / (k + 2)]
    OSQPFloat alpha_minus_one_scalling  = (alpha_adjustment - 1.)  * scalling;
    OSQPVectorf_add_scaled(work->v, alpha_adjustment, work->v, -alpha_minus_one_scalling, work->v_prev);
    OSQPVectorf_add_scaled(work->x, alpha_adjustment, work->x, -alpha_minus_one_scalling, work->x_prev);
  }
  else {
    OSQPFloat lambd_scalling  = settings->lambd * scalling;
    OSQPVectorf_add_scaled(work->v, lambd_plus_one, work->v, -lambd_scalling, work->v_prev);
    OSQPVectorf_add_scaled(work->x, lambd_plus_one, work->x, -lambd_scalling, work->x_prev);
  }
}

void update_halpern(OSQPSolver* solver) {
  OSQPSettings*  settings = solver->settings;
  OSQPWorkspace* work     = solver->work;
  OSQPInfo* info          = solver->info;

  OSQPFloat scalling;

  // OSQPFloat* x_data = OSQPVectorf_data(work->x);
  // OSQPFloat* z_data = OSQPVectorf_data(work->z);
  // OSQPFloat* y_data = OSQPVectorf_data(work->y);
  // OSQPFloat* v_data = OSQPVectorf_data(work->v);
  // OSQPFloat* x_prev_data = OSQPVectorf_data(work->x_prev);
  // OSQPFloat* v_prev_data = OSQPVectorf_data(work->v_prev);
  // OSQPFloat* x_outer_data = OSQPVectorf_data(work->x_outer);
  // OSQPFloat* v_outer_data = OSQPVectorf_data(work->v_outer);
  // c_print("Sent into func First element of x: %f\n", x_data[0]);
  // c_print("Sent into func First element of z: %f\n", z_data[0]);
  // c_print("Sent into func First element of y: %f\n", y_data[0]);
  // c_print("Sent into func First element of v: %f\n", v_data[0]);
  // c_print("Sent into func First element of x_outer: %f\n", x_outer_data[0]);
  // c_print("Sent into func First element of v_outer: %f\n", v_outer_data[0]);
  // c_print("Sent into func First element of x_prev: %f\n", x_prev_data[0]);
  // c_print("Sent into func First element of v_prev: %f\n", v_prev_data[0]);
  // c_print("sigma = %f\n", settings->sigma);
  // c_print("rho = %f\n", settings->rho);



  if ((strcmp(settings->halpern_scheme, "adaptive") == 0) ||
      (strcmp(settings->halpern_scheme, "adaptive only before init_rest_len") == 0)) {
    // c_print("halpern_scheme %s\n", settings->halpern_scheme);
    OSQPVectorf_minus(work->delta_x, work->x_prev, work->x);
    OSQPVectorf_minus(work->delta_x_loop, work->x_outer, work->x_prev);
    OSQPVectorf_minus(work->delta_v, work->v_prev, work->v);
    OSQPVectorf_minus(work->delta_v_loop, work->v_outer, work->v_prev);

    OSQPFloat delta_x_norm = OSQPVectorf_norm_2(work->delta_x);
    OSQPFloat delta_v_norm = OSQPVectorf_norm_2(work->delta_v);

    // We add 1e-10 for numerical stability
    OSQPFloat phi = 2. * (
      (
        settings->sigma * OSQPVectorf_dot_prod_signed(work->delta_x, work->delta_x_loop, 0) +
        settings->rho * OSQPVectorf_dot_prod_signed(work->delta_v, work->delta_v_loop, 0)
      ) / (
        settings->sigma * delta_x_norm * delta_x_norm +
        settings->rho * delta_v_norm * delta_v_norm + 1e-10
      )
    ) + 1.;
    // c_print("phi: %f\n", phi);
    // c_print("np.dot(self.x_prev - self.x, self.last_restart_x - self.x_prev) %f\n",OSQPVectorf_dot_prod_signed(work->delta_x, work->delta_x_loop, 0));
    // c_print("np.dot(self.v_prev - self.v, self.last_restart_v - self.v_prev) %f\n",OSQPVectorf_dot_prod_signed(work->delta_v, work->delta_v_loop, 0));
    // c_print("la.norm(self.x_prev - self.x, 2) %f\n",delta_x_norm);
    // c_print("la.norm(self.v_prev - self.v, 2) %f\n",delta_v_norm);

    scalling = (phi) / (phi + 1.);
  }
  else {
    scalling = (info->inner_loop_iter + 1.) / (info->inner_loop_iter + 2.);
  }
  // We compute [1 / (k + 2)] using [1.0 - scalling] as division is expensive/inaccurate (I think?)
  OSQPFloat reverse_scalling  = (1. - scalling);

  // (x^k, v^k) = [(k+1) / (k+2)] * T((x^k, v^k))
  OSQPVectorf_mult_scalar(work->v, scalling);
  OSQPVectorf_mult_scalar(work->x, scalling);

  if (strcmp(settings->restart_type, "reflected halpern") == 0) {
    // (x^k, v^k) = (lambd + 1) (x^k, v^k) - [(k+1) / (k+2)] * (x^k, v^k)
    // or (x^k, v^k) = [(lambd + 1) / alpha] (x^k, v^k) - [(lambd + 1) / alpha] * [(k+1) / (k+2)] * (x^k, v^k)
    // c_print("restart_type %s\n", settings->restart_type);
    reflected_halpern_step(solver, scalling);
  }

  // (x^k, v^k) = (x^k, v^k) + [1 / (k+2)] * (x^0, v^0)
  OSQPVectorf_add_scaled(work->v, 1., work->v, reverse_scalling, work->v_outer);
  OSQPVectorf_add_scaled(work->x, 1., work->x, reverse_scalling, work->x_outer);

  // z^k = Proj_C (v^k)
  OSQPVectorf_ew_bound_vec(work->z, work->v, work->data->l, work->data->u);

  // y^k = rho * (Id - Proj_C) (v^k)
  OSQPVectorf_minus(work->y, work->v, work->z);
  if (settings->rho_is_vec) {
    OSQPVectorf_ew_prod(work->y, work->y, work->rho_vec);
  }
  else {
    OSQPVectorf_mult_scalar(work->y, settings->rho);
  }

  // x_data = OSQPVectorf_data(work->x);
  // z_data = OSQPVectorf_data(work->z);
  // y_data = OSQPVectorf_data(work->y);
  // v_data = OSQPVectorf_data(work->v);
  // x_prev_data = OSQPVectorf_data(work->x_prev);
  // v_prev_data = OSQPVectorf_data(work->v_prev);
  // x_outer_data = OSQPVectorf_data(work->x_outer);
  // v_outer_data = OSQPVectorf_data(work->v_outer);
  // c_print("New First element of x: %f\n", x_data[0]);
  // c_print("New First element of z: %f\n", z_data[0]);
  // c_print("New First element of y: %f\n", y_data[0]);
  // c_print("New First element of v: %f\n", v_data[0]);
  // c_print("New First element of x_outer: %f\n", x_outer_data[0]);
  // c_print("New First element of v_outer: %f\n", v_outer_data[0]);
  // c_print("New First element of x_prev: %f\n", x_prev_data[0]);
  // c_print("New First element of v_prev: %f\n", v_prev_data[0]);
}

// void update_reflected_halpern(OSQPSolver* solver, OSQPInt k) {
//   OSQPSettings*  settings = solver->settings;
//   OSQPWorkspace* work     = solver->work;

//   OSQPFloat scalling = (k + 1.) / (k + 2.);

//   OSQPVectorf_add_scaled(work->v, 1. + settings->lambd, work->v, -settings->lambd, work->v_prev);
//   OSQPVectorf_add_scaled(work->x, 1. + settings->lambd, work->x, -settings->lambd, work->x_prev);

//   OSQPVectorf_mult_scalar(work->v, scalling);
//   OSQPVectorf_mult_scalar(work->x, scalling);

//   OSQPVectorf_add_scaled(work->v, 1., work->v, (1. - scalling), work->v_outer);
//   OSQPVectorf_add_scaled(work->x, 1., work->x, (1. - scalling), work->x_outer);

//   OSQPVectorf_ew_bound_vec(work->z, work->v, work->data->l, work->data->u);

//   OSQPVectorf_minus(work->y, work->v, work->z);
//   OSQPVectorf_ew_prod(work->y, work->y, work->rho_vec);

// }

OSQPFloat smoothed_duality_gap(OSQPSolver*  solver,
                               OSQPVectorf* x,
                               OSQPVectorf* z,
                               OSQPVectorf* y,
                               OSQPVectorf* w,
                               OSQPVectorf* xz_tilde,
                               OSQPVectorf* xtilde_view,
                               OSQPVectorf* ztilde_view) {
  // At current implementation w is equal to 0 therefore it can be omitted from most steps

  OSQPSettings*  settings = solver->settings;
  OSQPWorkspace* work     = solver->work;

  OSQPFloat smoothed_gap;
  OSQPFloat norm;

  // c_print("xi %f\n", settings->xi);
  OSQPFloat xi_reciprocal     = 1. / settings->xi;
  OSQPFloat sigma_reciprocal  = 1. / settings->sigma;
  OSQPFloat rho_reciprocal    = 1. / settings->rho;

  OSQPFloat xi_over_2     = settings->xi / 2.;
  OSQPFloat sigma_over_2  = settings->sigma / 2.;
  OSQPFloat rho_over_2    = settings->sigma / 2.;

  OSQPFloat sigma_over_xi = settings->sigma / settings->xi;
  OSQPFloat rho_over_xi   = settings->rho / settings->xi;

  // y_z = (1 / xi) y + z
  OSQPVectorf_add_scaled(work->y_z, xi_reciprocal, work->y, 1., work->z);

  // OSQPVectorf* temp_1_m;  
  // OSQPVectorf* temp_2_m;  
  // OSQPVectorf* temp_3_m;  
  // OSQPVectorf* temp_1_n;  
  // OSQPVectorf* temp_2_n; 
  // OSQPVectorf* temp_3_n;  
  
  // self.J = (self.settings.sigma / self.settings.xi) * sparse.eye(self.data.n, format='csc')
  // self.L = (self.settings.rho / self.settings.xi) * sparse.eye(self.data.m, format='csc')


  // 0.5 * x^T P x + q^T x
  OSQPMatrix_Axpy(work->data->P, xtilde_view, work->temp_1_n, 1., 0.);
  smoothed_gap = 0.5 * OSQPVectorf_dot_prod(xtilde_view, work->temp_1_n);
  smoothed_gap += OSQPVectorf_dot_prod(work->data->q, xtilde_view);

  // 0.5 * ||-q - w + sigma * x_tilde + A^T * (-y + rho z_tilde)||_{(P + sigma * I + rho * A^T A)^-1}
  OSQPVectorf_add_scaled(work->temp_1_m, -1., y, settings->rho, ztilde_view);
  OSQPMatrix_Atxpy(work->data->A, work->temp_1_m, work->temp_1_m, 1., 0.);
  OSQPVectorf_add_scaled3(work->temp_1_m, -1., work->data->q, settings->sigma, xtilde_view, 1., work->temp_1_m);
  OSQPVectorf_minus(work->temp_1_m, work->temp_1_m, w);
  // Still need to do the semi-norm computation (need to compute the inverse)
  //    If w=0 then we need to solve the same exact KKT system as for the tilde
  //      iterates, but with the new choices of x, z, and y. After obtaining the
  //      x_tilde value, we just compute the dot product of it with work->temp_1_m.

  // temp_xz_tilde, temp_xtilde_view, temp_ztilde_view

  // Preparing to solve the linear system for the inverse (same process as compute_rhs())
  // OSQPVectorf_copy(work->temp_xz_tilde, xz_tilde);

  //part related to x variables
  OSQPVectorf_add_scaled3(work->temp_xtilde_view,
                         settings->sigma,x,
                         -1., work->data->q, -1., w);

  if (settings->vector_rho_in_averaged_KKT) {
    // c_print("vector_rho_in_averaged_KKT %d\n", settings->vector_rho_in_averaged_KKT);
    OSQPVectorf_ew_prod(work->temp_ztilde_view, work->rho_inv_vec, y);
    OSQPVectorf_add_scaled(work->temp_ztilde_view,
                          -1.0, work->temp_ztilde_view,
                          1.0, z);
  }
  else {
    OSQPVectorf_add_scaled(work->temp_ztilde_view,
                          1.0, z,
                          -work->rho_inv, y);
  }

  // Solve linear system
  work->linsys_solver->solve(work->linsys_solver, work->temp_xz_tilde, solver->info->iter);

  norm = OSQPVectorf_dot_prod_signed(work->temp_1_m, work->temp_xtilde_view, 0);
  smoothed_gap += 0.5 * norm * norm;

  // -(sigma / 2) ||x_tilde||^2
  norm = OSQPVectorf_norm_2(xtilde_view);
  smoothed_gap += - sigma_over_2 * norm * norm;

  // -(rho / 2) ||z_tilde||^2
  norm = OSQPVectorf_norm_2(ztilde_view);
  smoothed_gap += - rho_over_2 * norm * norm;

  // (xi / 2) ||(1 / xi) w + x||^2
  OSQPVectorf_copy(work->temp_1_n, w);
  OSQPVectorf_mult_scalar(work->temp_1_n, xi_reciprocal);
  OSQPVectorf_plus(work->temp_1_n, work->temp_1_n, x);
  norm = OSQPVectorf_norm_2(work->temp_1_n);
  smoothed_gap += xi_over_2 * norm * norm;

  // -(xi / 2) ||x||^2
  norm = OSQPVectorf_norm_2(x);
  smoothed_gap += - xi_over_2 * norm * norm;

  // -(xi / 2) ||P_C (y_z) - y_z||^2
  OSQPVectorf_ew_bound_vec(work->temp_1_m, work->y_z, work->data->l, work->data->u);
  OSQPVectorf_minus(work->temp_1_m, work->temp_1_m, work->y_z);
  norm = OSQPVectorf_norm_2(work->temp_1_m);
  smoothed_gap += - xi_over_2 * norm * norm;

  // (xi / 2) ||y_z||^2
  norm = OSQPVectorf_norm_2(work->y_z);
  smoothed_gap += xi_over_2 * norm * norm;

  // -(xi / 2) ||z||^2
  norm = OSQPVectorf_norm_2(z);
  smoothed_gap += - xi_over_2 * norm * norm;

  // (xi / 2) ||(1 / sigma) w + (sigma / xi) (x_tilde - x)||^2
  OSQPVectorf_add_scaled3(work->temp_1_n, sigma_reciprocal, w, sigma_over_xi, xtilde_view, -sigma_over_xi, x);
  norm = OSQPVectorf_norm_2(work->temp_1_n);
  smoothed_gap += xi_over_2 * norm * norm;

  // -(xi / 2) ||(1 / sigma) w||^2
  OSQPVectorf_copy(work->temp_1_n, w);
  OSQPVectorf_mult_scalar(work->temp_1_n, sigma_reciprocal);
  norm = OSQPVectorf_norm_2(work->temp_1_n);
  smoothed_gap += - xi_over_2 * norm * norm;

  // (xi / 2) ||(1 / rho) y + (rho / xi) (z_tilde - z)||^2
  OSQPVectorf_add_scaled3(work->temp_1_m, rho_reciprocal, y, rho_over_xi, ztilde_view, -rho_over_xi, z);
  norm = OSQPVectorf_norm_2(work->temp_1_m);
  smoothed_gap += xi_over_2 * norm * norm;

  // -(xi / 2) ||y||^2
  OSQPVectorf_copy(work->temp_1_m, y);
  OSQPVectorf_mult_scalar(work->temp_1_m, rho_reciprocal);
  norm = OSQPVectorf_norm_2(work->temp_1_m);
  smoothed_gap += - xi_over_2 * norm * norm;

  return smoothed_gap;
}

void fixed_point_norm(OSQPSolver* solver) {
  OSQPSettings*  settings = solver->settings;
  OSQPWorkspace* work     = solver->work;

  // Compute x - T(x)
  //  (can technically use work->x_prev instead of norm_delta_x as we will not use x_prev's value again)
  OSQPVectorf_minus(work->delta_x, work->x, work->x_prev);

  // Compute v - T(v)
  //  (can technically use work->v_prev instead of norm_delta_v as we will not use v_prev's value again)
  OSQPVectorf_minus(work->delta_v, work->v, work->v_prev);

  OSQPFloat x_norm2 = OSQPVectorf_norm_2(work->delta_x);
  OSQPFloat v_norm2 = OSQPVectorf_norm_2(work->delta_v);

  work->norm_cur = c_sqrt(
    settings->sigma * x_norm2 * x_norm2 + settings->rho * v_norm2 * v_norm2
    // settings->sigma * x_norm2 + settings->rho * v_norm2
  );
}

OSQPInt should_restart(OSQPSolver* solver) {
  OSQPSettings*  settings = solver->settings;
  OSQPWorkspace* work     = solver->work;
  OSQPInfo*      info     = solver->info;

  if (strcmp(settings->restart_type, "halpern") == 0 ||
        strcmp(settings->restart_type, "reflected halpern") == 0) {
    // c_print("beta %f\n", settings->beta);
    if (work->norm_cur <= settings->beta * work->norm_outer)
      return 1;
    else if (settings->adaptive_rest) {
      // c_print("adaptive_rest %d\n", settings->adaptive_rest);
      // c_print("restart_necessary %f\n", settings->restart_necessary);
      // c_print("restart_artificial %f\n", settings->restart_artificial);
      if (
        work->norm_cur <= settings->restart_necessary * work->norm_outer &&
        work->norm_cur > work->norm_prev
      )
        return 1;
      else if (
        info->inner_loop_iter >= settings->restart_artificial * solver->info->iter
      )
        return 1;
      else
        return 0;
    }
    else
      return 0;
  }
  else if (strcmp(settings->restart_type, "averaged") == 0) {
    if (work->duality_gap_cur <= settings->beta * settings->beta * work->duality_gap_outer)
      return 1;
    else if (settings->adaptive_rest) {
      // c_print("adaptive_rest %d\n", settings->adaptive_rest);
      // c_print("restart_necessary %f\n", settings->restart_necessary);
      // c_print("restart_artificial %f\n", settings->restart_artificial);
      if (
        work->norm_cur <= settings->restart_necessary * work->norm_outer &&
        work->norm_cur > work->norm_prev
      )
        return 1;
      else if (
        info->inner_loop_iter >= settings->restart_artificial * solver->info->iter
      )
        return 1;
      else
        return 0;
    }
    else
      return 0;
  }
  else 
    return 0;
}

void compute_obj_val_dual_gap(const OSQPSolver*  solver,
                              const OSQPVectorf* x,
                              const OSQPVectorf* y,
                                    OSQPFloat*   prim_obj_val,
                                    OSQPFloat*   dual_obj_val,
                                    OSQPFloat*   duality_gap) {
  OSQPFloat quad_term = 0.0;
  OSQPFloat lin_term  = 0.0;
  OSQPFloat sup_term  = 0.0;
  OSQPWorkspace* work = solver->work;

  /* NB: The function is always called after dual_res is computed */
  quad_term = OSQPVectorf_dot_prod(work->Px, x);
  lin_term  = OSQPVectorf_dot_prod(work->data->q, x);

  /* Compute the support function of the constraints, SC(y) = u'*max(y, 0) + l'*min(y, 0)
     by projecting y onto the polar of the recession cone of C=[l,u], then doing the dot products */
  OSQPVectorf_copy(work->z_prev, y);
  OSQPVectorf_project_polar_reccone(work->z_prev,
                                    work->data->l,
                                    work->data->u,
                                    OSQP_INFTY * OSQP_MIN_SCALING);

  // Round anything in the range [-OSQP_ZERO_DEADZONE, OSQP_ZERO_DEADZONE] to 0 to
  // prevent very small (i.e., 1e-20) values from blowing up the numerics.
  OSQPVectorf_round_to_zero(work->z_prev, OSQP_ZERO_DEADZONE);

  sup_term  = OSQPVectorf_dot_prod_signed(work->data->u, work->z_prev, +1);
  sup_term += OSQPVectorf_dot_prod_signed(work->data->l, work->z_prev, -1);

  /* Primal objective value is 0.5*x^T P x + q^T x */
  *prim_obj_val = 0.5 * quad_term + lin_term;

  /* Dual objective value is -0.5*x^T P x - SC(y)*/
  *dual_obj_val = -0.5 * quad_term - sup_term;

  /* Duality gap is x^T P x + q^T x + SC(y) */
  work->scaled_dual_gap = quad_term + lin_term + sup_term;

  if (solver->settings->scaling) {
    *prim_obj_val *= work->scaling->cinv;
    *dual_obj_val *= work->scaling->cinv;

    // We always store the duality gap in the info as unscaled (since it is for the user),
    // but we keep the scaled version to use as a termination check when requested.
    *duality_gap = work->scaling->cinv * work->scaled_dual_gap;
  } else {
    *duality_gap = work->scaled_dual_gap;
  }

  /* Save cost values for later use in termination tolerance computation */
  work->xtPx = quad_term;
  work->qtx  = lin_term;
  work->SC   = sup_term;
}

static OSQPFloat compute_duality_gap_tol(const OSQPSolver* solver,
                                               OSQPFloat   eps_abs,
                                               OSQPFloat   eps_rel) {
  OSQPFloat max_rel_eps = 0.0;
  OSQPSettings*  settings = solver->settings;
  OSQPWorkspace* work     = solver->work;

  /* Compute max{ |x'*P*x|, |q'*x|, |SC(y)|} */
  max_rel_eps = c_absval(work->xtPx);                     /* |x'P*x| */
  max_rel_eps = c_max(max_rel_eps, c_absval(work->qtx));  /* |q'*x| */
  max_rel_eps = c_max(max_rel_eps, c_absval(work->SC));   /* |SC(y)| */

  /* Unscale the termination tolerance if required*/
  if (settings->scaling && !settings->scaled_termination) {
    max_rel_eps = work->scaling->cinv * max_rel_eps;
  }

  // eps_duality_gap
  return eps_abs + eps_rel * max_rel_eps;
}

static OSQPFloat compute_prim_res(OSQPSolver*        solver,
                                  const OSQPVectorf* x,
                                  const OSQPVectorf* z) {

  // NB: Use z_prev as working vector
  // pr = Ax - z

  OSQPSettings*  settings = solver->settings;
  OSQPWorkspace* work     = solver->work;
  OSQPFloat prim_res;

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

static OSQPFloat compute_prim_tol(const OSQPSolver* solver,
                                  OSQPFloat         eps_abs,
                                  OSQPFloat         eps_rel) {

  OSQPFloat max_rel_eps, temp_rel_eps;
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

static OSQPFloat compute_dual_res(OSQPSolver*        solver,
                                  const OSQPVectorf* x,
                                  const OSQPVectorf* y) {

  // NB: Use x_prev as temporary vector
  // NB: Only upper triangular part of P is stored.
  // dr = q + A'*y + P*x

  OSQPSettings*  settings = solver->settings;
  OSQPWorkspace* work     = solver->work;
  OSQPFloat      dual_res;

  // dr = q
  OSQPVectorf_copy(work->x_prev, work->data->q);

  // Px = P * x
  OSQPMatrix_Axpy(work->data->P, x, work->Px, 1.0, 0.0);

  // dr += Px
  OSQPVectorf_plus(work->x_prev, work->x_prev, work->Px);

  // dr += A' * y
  if (work->data->m) {
    OSQPMatrix_Atxpy(work->data->A, y, work->Aty, 1.0, 0.0);
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

static OSQPFloat compute_dual_tol(const OSQPSolver* solver,
                                  OSQPFloat         eps_abs,
                                  OSQPFloat         eps_rel) {

  OSQPFloat max_rel_eps, temp_rel_eps;
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

OSQPInt is_primal_infeasible(OSQPSolver* solver,
                             OSQPFloat   eps_prim_inf) {

  /* This function checks the primal infeasibility criteria
     1) || A' * delta_y || < eps * ||delta_y||
     2) u'*max(delta_y, 0) + l'*min(delta_y, 0) < 0
   */

  OSQPFloat norm_delta_y;
  OSQPFloat ineq_lhs = 0.0;
  OSQPSettings*  settings = solver->settings;
  OSQPWorkspace* work     = solver->work;

  // Project delta_y onto the polar of the recession cone of C=[l,u]
  OSQPVectorf_project_polar_reccone(work->delta_y,
                                    work->data->l,
                                    work->data->u,
                                    OSQP_INFTY * OSQP_MIN_SCALING);

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

OSQPInt is_dual_infeasible(OSQPSolver* solver,
                           OSQPFloat   eps_dual_inf) {

  /* This function checks the dual infeasibility criteria
     1) q * delta_x < 0
     2) ||P * delta_x || < eps * || delta_x ||
     3) (A * delta_x)_i > -eps * || delta_x ||,    l_i != -inf
        (A * delta_x)_i <  eps * || delta_x ||,    u_i != +inf
   */

  OSQPFloat norm_delta_x;
  OSQPFloat cost_scaling;
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

        // If you get this far, then all tests passed, so return results from final test
        // Test whether Adelta_x is in the recession cone of C = [l, u]
        return OSQPVectorf_in_reccone(work->Adelta_x,
                                      work->data->l,
                                      work->data->u,
                                      OSQP_INFTY * OSQP_MIN_SCALING,
                                      eps_dual_inf * norm_delta_x);
      }
    }
  }

  // Conditions not satisfied -> not dual infeasible
  return 0;
}

OSQPInt has_solution(const OSQPInfo* info) {
  return ((info->status_val != OSQP_PRIMAL_INFEASIBLE) &&
      (info->status_val != OSQP_PRIMAL_INFEASIBLE_INACCURATE) &&
      (info->status_val != OSQP_DUAL_INFEASIBLE) &&
      (info->status_val != OSQP_DUAL_INFEASIBLE_INACCURATE) &&
      (info->status_val != OSQP_NON_CVX));
}

void store_solution(OSQPSolver *solver, OSQPSolution* solution) {

#ifndef OSQP_EMBEDDED_MODE
  OSQPFloat norm_vec;
#endif /* ifndef OSQP_EMBEDDED_MODE */

  OSQPInfo*      info     = solver->info;
  OSQPSettings*  settings = solver->settings;
  OSQPWorkspace* work     = solver->work;

  /* Bypass function if solution wasn't allocated */
  if (!solution)
    return;


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


#ifndef OSQP_EMBEDDED_MODE

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

#endif /* ifndef OSQP_EMBEDDED_MODE */
  }
}

void update_info(OSQPSolver* solver,
                 OSQPInt     iter,
                 OSQPInt     polishing) {

  OSQPVectorf* x;
  OSQPVectorf* z;
  OSQPVectorf* y;                   // Allocate pointers to vectors

  // objective value, residuals
  OSQPFloat* prim_obj_val;
  OSQPFloat* dual_obj_val;
  OSQPFloat* dual_gap;
  OSQPFloat* prim_res;
  OSQPFloat* dual_res;

  OSQPInfo*      info     = solver->info;
  OSQPWorkspace* work     = solver->work;

#ifdef OSQP_ENABLE_PROFILING
  OSQPFloat* run_time;                    // Execution time
#endif /* ifdef OSQP_ENABLE_PROFILING */

#ifndef OSQP_EMBEDDED_MODE

  if (polishing) {
    x            = work->pol->x;
    y            = work->pol->y;
    z            = work->pol->z;
    prim_obj_val = &work->pol->obj_val;
    dual_obj_val = &work->pol->dual_obj_val;
    dual_gap     = &work->pol->duality_gap;
    prim_res     = &work->pol->prim_res;
    dual_res     = &work->pol->dual_res;
# ifdef OSQP_ENABLE_PROFILING
    run_time     = &info->polish_time;
# endif /* ifdef OSQP_ENABLE_PROFILING */
  }
  else {
#endif // OSQP_EMBEDDED_MODE
    x            = work->x;
    y            = work->y;
    z            = work->z;
    prim_obj_val = &info->obj_val;
    dual_obj_val = &info->dual_obj_val;
    dual_gap     = &info->duality_gap;
    prim_res     = &info->prim_res;
    dual_res     = &info->dual_res;
    info->iter   = iter;
#ifdef OSQP_ENABLE_PROFILING
    run_time     = &info->solve_time;
#endif /* ifdef OSQP_ENABLE_PROFILING */
#ifndef OSQP_EMBEDDED_MODE
}

#endif /* ifndef OSQP_EMBEDDED_MODE */

  // Compute primal residual
  if (work->data->m == 0) {
    // No constraints -> Always primal feasible
    *prim_res = 0.;
  } else {
    *prim_res = compute_prim_res(solver, x, z);
  }

  // Compute dual residual; store P*x in work->Px
  *dual_res = compute_dual_res(solver, x, y);

  // Compute the objective and duality gap, store various temp values in work
  compute_obj_val_dual_gap(solver, x, y, prim_obj_val, dual_obj_val, dual_gap);

  // Compute the duality gap integral
  if (!polishing) {
    info->primdual_int += c_absval(*dual_gap);
  }

  // Update timing
#ifdef OSQP_ENABLE_PROFILING
  *run_time = osqp_toc(work->timer);
#endif /* ifdef OSQP_ENABLE_PROFILING */

  // Compute the relative KKT error
  info->rel_kkt_error = c_max( c_max(*dual_res, *prim_res), *dual_gap);

#ifdef OSQP_ENABLE_PRINTING
  work->summary_printed = 0; // The just updated info have not been printed
#endif /* ifdef OSQP_ENABLE_PRINTING */
}


void reset_info(OSQPInfo *info) {
#ifdef OSQP_ENABLE_PROFILING

  // Initialize info values.
  info->solve_time = 0.0;  // Solve time to zero
# ifndef OSQP_EMBEDDED_MODE
  info->polish_time = 0.0; // Polish time to zero
# endif /* ifndef OSQP_EMBEDDED_MODE */

  // NB: We do not reset the setup_time because it is performed only once
#endif /* ifdef OSQP_ENABLE_PROFILING */

  update_status(info, OSQP_UNSOLVED); // Problem is unsolved

#if OSQP_EMBEDDED_MODE != 1
  info->rho_updates = 0;              // Rho updates are now 0
#endif /* if OSQP_EMBEDDED_MODE != 1 */
}

const char *OSQP_STATUS_MESSAGE[] = {
  "",   // Status messages start from 1, so add a buffer
  "solved",
  "solved inaccurate",
  "primal infeasible",
  "primal infeasible inaccurate",
  "dual infeasible",
  "dual infeasible inaccurate",
  "maximum iterations reached",
  "run time limit reached",
  "problem non convex",
  "interrupted",
  "unsolved"
};

void update_status(OSQPInfo* info,
                   OSQPInt   status_val) {
  // Update status value
  info->status_val = status_val;

  // Update status string depending on status val
  c_strcpy(info->status, OSQP_STATUS_MESSAGE[status_val]);
}

OSQPInt check_termination(OSQPSolver* solver,
                          OSQPInt     approximate) {

  OSQPFloat eps_prim, eps_dual, eps_duality_gap, eps_prim_inf, eps_dual_inf;
  OSQPInt   exitflag;
  OSQPInt   prim_res_check, dual_res_check, duality_gap_check, prim_inf_check, dual_inf_check;
  OSQPFloat eps_abs, eps_rel;

  OSQPInfo*      info     = solver->info;
  OSQPSettings*  settings = solver->settings;
  OSQPWorkspace* work     = solver->work;

  // Initialize variables to 0
  exitflag       = 0;
  prim_res_check = 0; dual_res_check = 0;
  prim_inf_check = 0; dual_inf_check = 0;
  duality_gap_check = 0;

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
    eps_prim = compute_prim_tol(solver, eps_abs, eps_rel);

    // Primal feasibility check
    if (info->prim_res < eps_prim) {
      prim_res_check = 1;
    } else {
      // Primal infeasibility check
      prim_inf_check = is_primal_infeasible(solver, eps_prim_inf);
    }
  } // End check if m == 0

  // Compute dual tolerance
  eps_dual = compute_dual_tol(solver, eps_abs, eps_rel);

  // Dual feasibility check
  if (info->dual_res < eps_dual) {
    dual_res_check = 1;
  } else {
    // Check dual infeasibility
    dual_inf_check = is_dual_infeasible(solver, eps_dual_inf);
  }

  if (settings->check_dualgap ) {
    // Compute duality gap tolerance
    eps_duality_gap = compute_duality_gap_tol(solver, eps_abs, eps_rel);

    // Duality gap check
    if (settings->scaling && !settings->scaled_termination) {
      // Use the unscaled duality gap value
      if (c_absval(info->duality_gap) < eps_duality_gap) {
        duality_gap_check = 1;
      }
    } else {
      // Use the scaled duality gap value
      if (c_absval(work->scaled_dual_gap) < eps_duality_gap) {
        duality_gap_check = 1;
      }
    }
  } else {
    // Force to 1 to bypass the check
    duality_gap_check = 1;
  }

  // Compare checks to determine solver status
  if (prim_res_check && dual_res_check && duality_gap_check) {
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


#ifndef OSQP_EMBEDDED_MODE

OSQPInt validate_data(const OSQPCscMatrix* P,
                      const OSQPFloat*     q,
                      const OSQPCscMatrix* A,
                      const OSQPFloat*     l,
                      const OSQPFloat*     u,
                            OSQPInt        m,
                            OSQPInt        n) {
  OSQPInt j, ptr;

  if (!P) {
    c_eprint("Missing quadratic cost matrix P");
    return 1;
  }

  if (!A) {
    c_eprint("Missing constraint matrix A");
    return 1;
  }

  if (!q) {
    c_eprint("Missing linear cost vector q");
    return 1;
  }

  // General dimensions Tests
  if ((n <= 0) || (m < 0)) {
    c_eprint("n must be positive and m nonnegative; n = %i, m = %i",
             (int)n, (int)m);
    return 1;
  }

  // Matrix P
  if (P->m != n) {
    c_eprint("P does not have dimension n x n with n = %i", (int)n);
    return 1;
  }

  if (P->m != P->n) {
    c_eprint("P is not square");
    return 1;
  }

  for (j = 0; j < n; j++) { // COLUMN
    for (ptr = P->p[j]; ptr < P->p[j + 1]; ptr++) {
      if (P->i[ptr] > j) {  // if ROW > COLUMN
        c_eprint("P is not upper triangular");
        return 1;
      }
    }
  }

  // Matrix A
  if ((A->m != m) || (A->n != n)) {
    c_eprint("A does not have dimension %i x %i", (int)m, (int)n);
    return 1;
  }

  // Lower and upper bounds
  for (j = 0; j < m; j++) {
    if (l[j] > u[j]) {
      c_eprint("Lower bound at index %d is greater than upper bound: %.4e > %.4e",
               (int)j, l[j], u[j]);
      return 1;
    }
  }

  return 0;
}

#endif /* ifndef OSQP_EMBEDDED_MODE */


OSQPInt validate_linsys_solver(OSQPInt linsys_solver) {
  /* Verify the algebra backend supports the requested indirect solver */
  if ( (linsys_solver == OSQP_INDIRECT_SOLVER) &&
     (osqp_algebra_linsys_supported() & OSQP_CAPABILITY_INDIRECT_SOLVER) ) {
    return 0;
  }

  /* Verify the algebra backend supports the requested direct solver */
  if ( (linsys_solver == OSQP_DIRECT_SOLVER) &&
     (osqp_algebra_linsys_supported() & OSQP_CAPABILITY_DIRECT_SOLVER) ) {
    return 0;
  }

  // Invalid solver
  return 1;
}


OSQPInt validate_settings(const OSQPSettings* settings,
                          OSQPInt             from_setup) {

  if (!settings) {
    c_eprint("Missing settings!");
    return 1;
  }

  if (from_setup &&
      validate_linsys_solver(settings->linsys_solver)) {
    c_eprint("linsys_solver not recognized");
    return 1;
  }

  if (from_setup &&
      settings->allocate_solution != 0 &&
      settings->allocate_solution != 1) {
    c_eprint("allocate_solution must be either 0 or 1");
    return 1;
  }

  if (settings->verbose != 0 &&
      settings->verbose != 1) {
    c_eprint("verbose must be either 0 or 1");
    return 1;
  }

  if (settings->profiler_level != 0 &&
      settings->profiler_level != 1 &&
      settings->profiler_level != 2) {
    c_eprint("profiler_level must be either 0, 1 or 2");
    return 1;
  }

  if (settings->warm_starting != 0 &&
      settings->warm_starting != 1) {
    c_eprint("warm_start must be either 0 or 1");
    return 1;
  }

  if (from_setup && settings->scaling < 0) {
    c_eprint("scaling must be nonnegative");
    return 1;
  }

  if (settings->polishing != 0 &&
      settings->polishing != 1) {
    c_eprint("polishing must be either 0 or 1");
    return 1;
  }

  if (from_setup && settings->rho <= 0.0) {
    c_eprint("rho must be positive");
    return 1;
  }

  if (from_setup &&
      (settings->rho_is_vec != 0) &&
      (settings->rho_is_vec != 1)) {
    c_eprint("rho_is_vec must be either 0 or 1");
    return 1;
  }
  
  if ((settings->vector_rho_in_averaged_KKT != 0) &&
      (settings->vector_rho_in_averaged_KKT != 1)) {
    c_eprint("vector_rho_in_averaged_KKT must be either 0 or 1");
    return 1;
  }

  if ((settings->rho_is_vec == 0) && 
      (settings->vector_rho_in_averaged_KKT != 0)) {
    c_eprint("To perform vector_rho_in_averaged_KKT, rho_is_vec must be 1");
    return 1;
  }

  if (from_setup && settings->sigma <= 0.0) {
    c_eprint("sigma must be positive");
    return 1;
  }

  if (settings->alpha <= 0.0 ||
      settings->alpha >= 2.0) {
    c_eprint("alpha must be strictly between 0 and 2");
    return 1;
  }

  if (settings->beta <= 0.0 ||
      settings->beta > 1.0) {
    c_eprint("beta must be in (0,1]");
    return 1;
  }

  if (settings->lambd < 0.0 ||
      settings->lambd > 1.0) {
    c_eprint("lambda must be in [0,1]");
    return 1;
  }

  if (settings->restart_necessary <= 0.0 ||
      settings->restart_necessary > 1.0) {
    c_eprint("restart_necessary must be in (0,1]");
    return 1;
  }

  if (settings->restart_artificial <= 0.0 ||
      settings->restart_artificial > 1.0) {
    c_eprint("restart_artificial must be in (0,1]");
    return 1;
  }

  if (settings->adaptive_rest != 0 &&
      settings->adaptive_rest != 1) {
    c_eprint("adaptive_rest must be either 0 or 1");
    return 1;
  }

  if (settings->alpha_adjustment_reflected_halpern != 0 &&
      settings->alpha_adjustment_reflected_halpern != 1) {
    c_eprint("alpha_adjustment_reflected_halpern must be either 0 or 1");
    return 1;
  }

  if ((settings->alpha_adjustment_reflected_halpern != 0) &&
      (settings->lambd > (2. / settings->alpha) - 1.)) {
    c_eprint("lambda must be <= [(2 / alpha) - 1] if we use alpha_adjustment_reflected_halpern");
    return 1;
  }

  if (settings->rho_custom_condition != 0 &&
      settings->rho_custom_condition != 1) {
    c_eprint("rho_custom_condition must be either 0 or 1");
    return 1;
  }

  if (settings->custom_average_rest != 0 &&
      settings->custom_average_rest != 1) {
    c_eprint("custom_average_rest must be either 0 or 1");
    return 1;
  }

  if (settings->adapt_rho_on_restart != 0 &&
      settings->adapt_rho_on_restart != 1) {
    c_eprint("adapt_rho_on_restart must be either 0 or 1");
    return 1;
  }

  if (settings->pid_controller != 0 &&
      settings->pid_controller != 1) {
    c_eprint("pid_controller must be either 0 or 1");
    return 1;
  }

  if (settings->pid_controller_sqrt != 0 &&
      settings->pid_controller_sqrt != 1) {
    c_eprint("pid_controller_sqrt must be either 0 or 1");
    return 1;
  }

  if (settings->pid_controller_sqrt_mult != 0 &&
      settings->pid_controller_sqrt_mult != 1) {
    c_eprint("pid_controller_sqrt_mult must be either 0 or 1");
    return 1;
  }

  if (settings->pid_controller_sqrt_mult_2 != 0 &&
      settings->pid_controller_sqrt_mult_2 != 1) {
    c_eprint("pid_controller_sqrt_mult_2 must be either 0 or 1");
    return 1;
  }

  if (settings->pid_controller_log != 0 &&
      settings->pid_controller_log != 1) {
    c_eprint("pid_controller_log must be either 0 or 1");
    return 1;
  }

  if (settings->rho_custom_tolerance <= 0) {
    c_eprint("rho_custom_tolerance must be greater than 0");
    return 1;
  }

  if (settings->ini_rest_len <= 0.0) {
    c_eprint("Initial restart period must be larger than 0");
    return 1;
  }

  if (settings->cg_max_iter <= 0) {
    c_eprint("cg_max_iter must be positive");
    return 1;
  }

  if (settings->cg_tol_reduction <= 0) {
    c_eprint("cg_tol_reduction must be positive");
    return 1;
  }

  if (settings->cg_tol_fraction <= 0.0 ||
      settings->cg_tol_fraction >= 1.0) {
    c_eprint("cg_tol_fraction must be strictly between 0 and 1");
    return 1;
  }

  if (from_setup &&
      settings->adaptive_rho < 0 ||
      settings->adaptive_rho >= _OSQP_ADAPTIVE_RHO_UPDATE_LAST_VALUE) {
    c_eprint("adaptive_rho not a valid choice");
    return 1;
  }

#if !defined(OSQP_ENABLE_PROFILING)
  if (from_setup && settings->adaptive_rho == OSQP_ADAPTIVE_RHO_UPDATE_TIME) {
    c_eprint("adaptive_rho time-based adaptation requires profiling to be enabled");
    return 1;
  }
#endif /* !defined(OSQP_ENABLE_PROFILING) */

  if (from_setup && settings->adaptive_rho_interval < 0) {
    c_eprint("adaptive_rho_interval must be nonnegative");
    return 1;
  }

  if (from_setup && settings->adaptive_rho_fraction <= 0) {
    c_eprint("adaptive_rho_fraction must be positive");
    return 1;
  }

  if (from_setup && settings->adaptive_rho_tolerance_greater < 0.0) {
    c_eprint("adaptive_rho_tolerance_greater must be >= 0");
    return 1;
  }

  if (from_setup && settings->adaptive_rho_tolerance_less < 0.0) {
    c_eprint("adaptive_rho_tolerance_less must be >= 0");
    return 1;
  }

  if (settings->max_iter <= 0) {
    c_eprint("max_iter must be positive");
    return 1;
  }

  if (settings->eps_abs < 0.0) {
    c_eprint("eps_abs must be nonnegative");
    return 1;
  }

  if (settings->eps_rel < 0.0) {
    c_eprint("eps_rel must be nonnegative");
    return 1;
  }

  if (settings->eps_rel == 0.0 &&
      settings->eps_abs == 0.0) {
    c_eprint("at least one of eps_abs and eps_rel must be positive");
    return 1;
  }

  if (settings->eps_prim_inf <= 0.0) {
    c_eprint("eps_prim_inf must be positive");
    return 1;
  }

  if (settings->eps_dual_inf <= 0.0) {
    c_eprint("eps_dual_inf must be positive");
    return 1;
  }

  if (settings->scaled_termination != 0 &&
      settings->scaled_termination != 1) {
    c_eprint("scaled_termination must be either 0 or 1");
    return 1;
  }

  if (settings->check_termination < 0) {
    c_eprint("check_termination must be nonnegative");
    return 1;
  }

  if (settings->check_dualgap != 0 &&
      settings->check_dualgap != 1) {
    c_eprint("check_dualgap must be either 0 or 1");
    return 1;
  }

  if (settings->time_limit <= 0.0) {
    c_eprint("time_limit must be positive\n");
    return 1;
  }

  if (settings->delta <= 0.0) {
    c_eprint("delta must be positive");
    return 1;
  }

  if (settings->polish_refine_iter < 0) {
    c_eprint("polish_refine_iter must be nonnegative");
    return 1;
  }

  return 0;
}
