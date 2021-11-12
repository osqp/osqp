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

  c_int j, n_active;
  c_int m = work->data->m;

  c_int *active_flags;
  c_float *z, *y, *u, *l;

  // Allocate raw arrays
  active_flags = (c_int *) c_malloc(m * sizeof(c_int));
  z = (c_float *) c_malloc(m * sizeof(c_float));
  y = (c_float *) c_malloc(m * sizeof(c_float));
  l = (c_float *) c_malloc(m * sizeof(c_float));
  u = (c_float *) c_malloc(m * sizeof(c_float));

  // Copy data to raw arrays
  OSQPVectori_to_raw(active_flags, work->pol->active_flags);
  OSQPVectorf_to_raw(z, work->z);
  OSQPVectorf_to_raw(y, work->y);
  OSQPVectorf_to_raw(l, work->data->l);
  OSQPVectorf_to_raw(u, work->data->u);

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

  // Copy raw vector into OSQPVectori structure
  OSQPVectori_from_raw(work->pol->active_flags, active_flags);

  //total active constraints
  work->pol->n_active = n_active;

  //extract the relevant rows
  work->pol->Ared = OSQPMatrix_submatrix_byrows(work->data->A, work->pol->active_flags);

  // Memory clean-up
  c_free(active_flags);
  c_free(z);
  c_free(y);
  c_free(l);
  c_free(u);

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
  c_int n = work->data->n;
  c_int m = work->data->m;
  c_int n_plus_mred = OSQPVectorf_length(rhs);

  c_int *active_flags;
  c_float *rhsv, *q, *l, *u;

  // Allocate raw arrays
  active_flags = (c_int *)   c_malloc(m           * sizeof(c_int));
  rhsv         = (c_float *) c_malloc(n_plus_mred * sizeof(c_float));
  q            = (c_float *) c_malloc(n           * sizeof(c_float));
  l            = (c_float *) c_malloc(m           * sizeof(c_float));
  u            = (c_float *) c_malloc(m           * sizeof(c_float));

  // Copy data to raw arrays
  OSQPVectori_to_raw(active_flags, work->pol->active_flags);
  OSQPVectorf_to_raw(rhsv, rhs);
  OSQPVectorf_to_raw(q, work->data->q);
  OSQPVectorf_to_raw(l, work->data->l);
  OSQPVectorf_to_raw(u, work->data->u);

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

  // Copy raw vector into OSQPVectorf structure
  OSQPVectorf_from_raw(rhs, rhsv);

  // Memory clean-up
  c_free(active_flags);
  c_free(rhsv);
  c_free(q);
  c_free(l);
  c_free(u);
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
  c_int i, mred;
  OSQPVectorf *rhs, *rhs1, *rhs2;
  OSQPVectorf *z1, *z2;

  OSQPSettings*  settings = solver->settings;
  OSQPWorkspace* work     = solver->work;

  if (settings->polish_refine_iter > 0) {
    mred = OSQPMatrix_get_m(work->pol->Ared);

    // Allocate dz and rhs vectors
    rhs = OSQPVectorf_malloc(work->data->n + mred);

    //form views of the top/bottom parts of rhs and z
    rhs1 = OSQPVectorf_view(rhs,0,work->data->n);
    rhs2 = OSQPVectorf_view(rhs,work->data->n,mred);
    z1   = OSQPVectorf_view(z,0,work->data->n);
    z2   = OSQPVectorf_view(z,work->data->n,mred);

    if (!rhs || !rhs1 || !rhs2 || !z1 || !z2) {
      return osqp_error(OSQP_MEM_ALLOC_ERROR);
    }

    for (i = 0; i < settings->polish_refine_iter; i++) {

      // Form the RHS for the iterative refinement:  b - K*z
      OSQPVectorf_copy(rhs,b);

      // Upper Part: R^{n}
      // -= Px  (in the top partition)
      OSQPMatrix_Axpy(work->data->P, z1, rhs1, -1.0, 1.0);

      // -= Ared'*y_red  (in the top partition)
      OSQPMatrix_Atxpy(work->pol->Ared, z2, rhs1, -1.0, 1.0);

      // Lower Part: R^{m}
      // -= A*x  (in the bottom partition)
      OSQPMatrix_Axpy(work->pol->Ared, z1, rhs2, -1.0, 1.0);

      // Solve linear system. Store solution in rhs
      p->solve(p, rhs, 1);

      // Update solution
      OSQPVectorf_plus(z,z,rhs);
    }

    OSQPVectorf_free(rhs);
    OSQPVectorf_view_free(rhs1);
    OSQPVectorf_view_free(rhs2);
    OSQPVectorf_view_free(z1);
    OSQPVectorf_view_free(z2);
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
  c_int m = work->data->m;
  c_int mred = OSQPVectorf_length(yred_vf);

  c_int *active_flags;
  c_float *y, *yred;

  // Allocate raw arrays
  active_flags = (c_int *)   c_malloc(m    * sizeof(c_int));
  y            = (c_float *) c_malloc(m    * sizeof(c_float));
  yred         = (c_float *) c_malloc(mred * sizeof(c_float));

  // Copy data to raw arrays
  OSQPVectori_to_raw(active_flags, work->pol->active_flags);
  OSQPVectorf_to_raw(y, work->y);
  OSQPVectorf_to_raw(yred, yred_vf);

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

  // Copy raw vector into OSQPVectorf structure
  OSQPVectorf_from_raw(work->pol->y, y);

  // Memory clean-up
  c_free(active_flags);
  c_free(y);
  c_free(yred);
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
                                OSQP_NULL, settings, OSQP_NULL, OSQP_NULL, 1);

  if (exitflag) {
    // Polishing failed
    info->status_polish = -1;

    // Memory clean-up
    OSQPMatrix_free(work->pol->Ared);

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

  // Warm start the polished solution
  plsh->warm_start(plsh, work->x);

  // Solve the reduced KKT system
  plsh->solve(plsh, pol_sol, 1);

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
  OSQPMatrix_Axpy(work->data->A, work->pol->x, work->pol->z, 1.0, 0.0);
  get_ypol_from_yred(work, pol_sol_yview);     // pol->y

  // Ensure z is in C and y is in N_C(z)
  project_normalcone(work->pol->z, work->pol->y, work->data->l, work->data->u);

  // Compute primal and dual residuals at the polished solution
  update_info(solver, 0, 1, 1);

  // Check if polish was successful
  polish_successful = (work->pol->prim_res < info->prim_res &&
                       work->pol->dual_res < info->dual_res) || // Residuals
                                                                    // are
                                                                    // reduced
                      (work->pol->prim_res < info->prim_res &&
                       info->dual_res < 1e-10) ||              // Dual
                                                                    // residual
                                                                    // already
                                                                    // tiny
                      (work->pol->dual_res < info->dual_res &&
                       info->prim_res < 1e-10);                // Primal
                                                                    // residual
                                                                    // already
                                                                    // tiny

  if (polish_successful) {
    // Update solver information
    info->obj_val       = work->pol->obj_val;
    info->prim_res      = work->pol->prim_res;
    info->dual_res      = work->pol->dual_res;
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
