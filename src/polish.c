#include "polish.h"
#include "lin_alg.h"
#include "util.h"
#include "auxil.h"
#include "lin_sys.h"
#include "proj.h"
#include "error.h"

/**
 * Form reduced matrix A that contains only rows that are active at the
 * solution.
 * Ared = vstack[Alow, Aupp]
 * Active constraints are guessed from the primal and dual solution returned by
 * the ADMM.
 * @param  work Workspace
 * @return      Number of rows in Ared, negative if error
 */
static c_int form_Ared(OSQPWorkspace *work){

  c_int j, ptr, n_active;
  c_int Ared_nnz = 0;

  c_int* active_flags       = OSQPVectori_data(work->pol->active_flags);
  c_float* z = OSQPVectorf_data(work->z);
  c_float* y = OSQPVectorf_data(work->y);
  c_float* l = OSQPVectorf_data(work->data->l);
  c_float* u = OSQPVectorf_data(work->data->u);

  // Initialize counters for active constraints
  n_active = 0;

  /* Guess which linear constraints are lower-active, upper-active and free
   *
   *    active_flags is -1/0/1 to indicate  lower/ inactive / upper.
   *    equality constraints are treated as lower active
   *
   *    Ared is formed by selecting all of the active rows.
   */

  for (j = 0; j < work->data->m; j++) {

    if ((z[j] - l[j] < -y[j]) || (l[j] == u[j]) ) { // lower-active or equality
      active_flags[j] = -1;
      n_active++;
    }
    else if (u[j] - z[j] < y[j]) { // upper-active
      active_flags[j] = +1;
      n_active++;
    }
    else{
      active_flags[j] = 0;
    }
  }

  //total active constraints
  work->pol->n_active = n_active;

  //extract the relevant rows
  work->pol->Ared = OSQPMatrix_submatrix_byrows(work->data->A, work->pol->active_flags, n_active);

  // Return number of rows in Ared
  return n_active;
}

/**
 * Form reduced right-hand side rhs_red = vstack[-q, l_low, u_upp]
 * @param  work Workspace
 * @param  rhs  right-hand-side
 * @return      reduced rhs
 */
static void form_rhs_red(OSQPWorkspace *work, OSQPVectorf *rhs) {

  c_int j, counter;

  c_float* rhsv = OSQPVectorf_data(rhs);
  c_float* q   = OSQPVectorf_data(work->data->q);
  c_float* l   = OSQPVectorf_data(work->data->l);
  c_float* u   = OSQPVectorf_data(work->data->u);
  c_int* active_flags = OSQPVectori_data(work->pol->active_flags);

  for(j = 0; j < work->data->n; j++){
    rhsv[j] = -q[j];
  }

  counter = 0;

  for (j = 0; j < work->data->m; j++) {
    if(active_flags[j] == -1){ // lower active
       rhsv[work->data->n + counter] = l[j];
       counter++;
    }
    else if(active_flags[j] == 1){ //upper actice
       rhsv[work->data->n + counter] = u[j];
       counter++;
    }
  }
}

/**
 * Perform iterative refinement on the polished solution:
 *    (repeat)
 *    1. (K + dK) * dz = b - K*z
 *    2. z <- z + dz
 * @param  work Solver workspace
 * @param  p    Private variable for solving linear system
 * @param  z    Initial z value
 * @param  b    RHS of the linear system
 * @return      Exitflag
 */
static c_int iterative_refinement(OSQPSolver    *solver,
                                  LinSysSolver  *p,
                                  OSQPVectorf   *z,
                                  OSQPVectorf   *b) {
  c_int i, j;
  OSQPVectorf *rhs, *rhs1, *rhs2;
  OSQPVectorf *z1, *z2;

  OSQPSettings*  settings = solver->settings;
  OSQPWorkspace* work     = solver->work;

  if (settings->polish_refine_iter > 0) {

    // Allocate dz and rhs vectors
    rhs = OSQPVectorf_malloc(work->data->n + OSQPMatrix_get_m(work->pol->Ared));

    //form views of the top/bottom parts of rhs and z
    rhs1 = OSQPVectorf_view(rhs,0,work->data->n);
    rhs2 = OSQPVectorf_view(rhs,work->data->n,work->data->m);
    z1   = OSQPVectorf_view(z,0,work->data->n);
    z2   = OSQPVectorf_view(z,work->data->n,work->data->m);

    if (!rhs || !rhs1 || !rhs2 || !z1 || !z2) {
      return osqp_error(OSQP_MEM_ALLOC_ERROR);
    }

    for (i = 0; i < settings->polish_refine_iter; i++) {

      // Form the RHS for the iterative refinement:  b - K*z
      OSQPVectorf_copy(rhs,b);

      // Upper Part: R^{n}
      // -= Px  (in the top partition)
      OSQPMatrix_Axpy(work->data->P, z1, rhs1, 1.0, -1.0);

      // -= Ared'*y_red  (in the top partition)
      OSQPMatrix_Atxpy(work->pol->Ared, z2, rhs1, 1.0, -1.0);

      // Lower Part: R^{m}
      // -= A*x  (in the bottom partition)
      OSQPMatrix_Axpy(work->pol->Ared, z1, rhs2, 1.0, -1.0);

      // Solve linear system. Store solution in rhs
      p->solve(p, rhs);

      // Update solution
      OSQPVectorf_plus(z,z,rhs);
    }

    if (rhs)  OSQPVectorf_free(rhs);
    if (rhs1) OSQPVectorf_view_free(rhs1);
    if (rhs2) OSQPVectorf_view_free(rhs2);
    if (z1)   OSQPVectorf_view_free(z1);
    if (z2)   OSQPVectorf_view_free(z2);
  }
  return 0;
}

/**
 * Compute dual variable y from yred
 * @param work Workspace
 * @param yred Dual variables associated to active constraints
 */
static void get_ypol_from_yred(OSQPWorkspace *work, OSQPVectorf *yred_vf) {

  c_int j, counter;

  c_int* active_flags = OSQPVectori_data(work->pol->active_flags);
  c_float* y     = OSQPVectorf_data(work->pol->y);
  c_float* yred  = OSQPVectorf_data(yred_vf);

  // If there are no active constraints
  if (work->pol->n_active == 0) {
    OSQPVectorf_set_scalar(work->pol->y, 0.);
    return;
  }

  counter = 0;

  for (j = 0; j < work->data->m; j++) {

    if (active_flags[j] == 0) { //inactive
      y[j] = 0;
    }
    else {  // active
      y[j] = yred[counter];
      counter++;
    }
  }
}

c_int polish(OSQPSolver *solver) {

  c_int mred, polish_successful, exitflag;

  LinSysSolver *plsh;
  OSQPVectorf *rhs_red;
  OSQPVectorf *pol_sol; // Polished solution (x and reduced y)
  OSQPVectorf *pol_sol_xview; // view into x part of polished solution
  OSQPVectorf *pol_sol_yview; // view into (reduced) y part of polished solutions

  OSQPInfo*      info      = solver->info;
  OSQPSettings*  settings  = solver->settings;
  OSQPWorkspace* work      = solver->work;

#ifdef PROFILING
  osqp_tic(work->timer); // Start timer
#endif /* ifdef PROFILING */

  // Form Ared by assuming the active constraints and store in work->pol->Ared
  mred = form_Ared(work);
  if (mred < 0) {
    // Polishing failed
    info->status_polish = -1;
    return -1;
  }

  // Form and factorize reduced KKT
  exitflag = init_linsys_solver(&plsh, work->data->P, work->pol->Ared,
                                settings->delta, OSQP_NULL,
                                settings->linsys_solver, 1);

  if (exitflag) {
    // Polishing failed
    info->status_polish = -1;

    // Memory clean-up
    if (work->pol->Ared) OSQPMatrix_free(work->pol->Ared);

    return 1;
  }

  // Form reduced right-hand side rhs_red
  rhs_red = OSQPVectorf_malloc(work->data->n + mred);
  if (!rhs_red) {
    // Polishing failed
    info->status_polish = -1;

    // Memory clean-up
    OSQPMatrix_free(work->pol->Ared);

    return -1;
  }
  form_rhs_red(work, rhs_red);

  pol_sol = OSQPVectorf_copy_new(rhs_red);
  pol_sol_xview = OSQPVectorf_view(pol_sol,0,work->data->n);
  pol_sol_yview = OSQPVectorf_view(pol_sol,work->data->n,mred);

  if (!pol_sol || !pol_sol_xview || !pol_sol_yview) {

    // Polishing failed
    info->status_polish = -1;

    // Memory clean-up
    OSQPMatrix_free(work->pol->Ared);
    OSQPVectorf_free(rhs_red);
    OSQPVectorf_free(pol_sol);
    OSQPVectorf_view_free(pol_sol_xview);
    OSQPVectorf_view_free(pol_sol_yview);

    return -1;
  }

  // Solve the reduced KKT system
  plsh->solve(plsh, pol_sol);

  // Perform iterative refinement to compensate for the regularization error
  exitflag = iterative_refinement(solver, plsh, pol_sol, rhs_red);

  if (exitflag) {
    // Polishing failed
    info->status_polish = -1;

    // Memory clean-up
    OSQPMatrix_free(work->pol->Ared);
    OSQPVectorf_free(rhs_red);
    OSQPVectorf_free(pol_sol);
    OSQPVectorf_view_free(pol_sol_xview);
    OSQPVectorf_view_free(pol_sol_yview);

    return -1;
  }

  // Store the polished solution (x,z,y)
  OSQPVectorf_copy(work->pol->x, pol_sol_xview);   // pol->x
  OSQPMatrix_Axpy(work->data->A, work->pol->x, work->pol->z, 0.0, 1.0);
  get_ypol_from_yred(work, pol_sol_yview);     // pol->y

  // Ensure (z,y) satisfies normal cone constraint
  project_normalcone(work, work->pol->z, work->pol->y);

  // Compute primal and dual residuals at the polished solution
  update_info(solver, 0, 1, 1);

  // Check if polish was successful
  polish_successful = (work->pol->pri_res < info->pri_res &&
                       work->pol->dua_res < info->dua_res) || // Residuals
                                                                    // are
                                                                    // reduced
                      (work->pol->pri_res < info->pri_res &&
                       info->dua_res < 1e-10) ||              // Dual
                                                                    // residual
                                                                    // already
                                                                    // tiny
                      (work->pol->dua_res < info->dua_res &&
                       info->pri_res < 1e-10);                // Primal
                                                                    // residual
                                                                    // already
                                                                    // tiny

  if (polish_successful) {
    // Update solver information
    info->obj_val       = work->pol->obj_val;
    info->pri_res       = work->pol->pri_res;
    info->dua_res       = work->pol->dua_res;
    info->status_polish = 1;

    // Update (x, z, y) in ADMM iterations
    // NB: z needed for warm starting
    OSQPVectorf_copy(work->x, work->pol->x);
    OSQPVectorf_copy(work->z, work->pol->z);
    OSQPVectorf_copy(work->y, work->pol->y);

    // Print summary
#ifdef PRINTING

    if (settings->verbose) print_polish(solver);
#endif /* ifdef PRINTING */
  } else { // Polishing failed
    info->status_polish = -1;

    // TODO: Try to find a better solution on the line connecting ADMM
    //       and polished solution
  }

  // Memory clean-up
  plsh->free(plsh);

  // Checks that they are not NULL are already performed earlier
  OSQPMatrix_free(work->pol->Ared);
  OSQPVectorf_free(rhs_red);
  OSQPVectorf_free(pol_sol);
  OSQPVectorf_view_free(pol_sol_xview);
  OSQPVectorf_view_free(pol_sol_yview);
  return 0;
}
