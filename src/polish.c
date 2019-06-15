#include "polish.h"
#include "lin_alg.h"
#include "util.h"
#include "auxil.h"
#include "lin_sys.h"
#include "kkt.h"
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

  c_int j, ptr;
  c_int Ared_nnz = 0;

  c_int* Alow_to_A = OSQPVectori_data(work->pol->Alow_to_A);
  c_int* A_to_Alow = OSQPVectori_data(work->pol->A_to_Alow);
  c_int* Aupp_to_A = OSQPVectori_data(work->pol->Aupp_to_A);
  c_int* A_to_Aupp = OSQPVectori_data(work->pol->A_to_Aupp);
  c_float* z = OSQPVectorf_data(work->z);
  c_float* y = OSQPVectorf_data(work->y);
  c_float* l = OSQPVectorf_data(work->data->l);
  c_float* u = OSQPVectorf_data(work->data->u);


  // Initialize counters for active constraints
  work->pol->n_low = 0;
  work->pol->n_upp = 0;

  /* Guess which linear constraints are lower-active, upper-active and free
   *    A_to_Alow[j] = -1    (if j-th row of A is not inserted in Alow)
   *    A_to_Alow[j] =  i    (if j-th row of A is inserted at i-th row of Alow)
   * Aupp is formed in the equivalent way.
   * Ared is formed by stacking vertically Alow and Aupp.
   */
  for (j = 0; j < work->data->m; j++) {
    if (z[j] - l[j] < y[j]) { // lower-active
      Alow_to_A[work->pol->n_low] = j;
      A_to_Alow[j]                = work->pol->n_low++;
    } else {
      A_to_Alow[j] = -1;
    }
  }

  for (j = 0; j < work->data->m; j++) {
    if (u[j] - z[j] < y[j]) { // upper-active
      Aupp_to_A[work->pol->n_upp] = j;
      A_to_Aupp[j]                = work->pol->n_upp++;
    } else {
      A_to_Aupp[j] = -1;
    }
  }

  // Check if there are no active constraints
  if (work->pol->n_low + work->pol->n_upp == 0) {
    // Form empty Ared
    work->pol->Ared = csc_spalloc(0, work->data->n, 0, 1, 0);
    if (!(work->pol->Ared)) return -1;
    int_vec_set_scalar(work->pol->Ared->p, 0, work->data->n + 1);
    return 0; // mred = 0
  }

  // Count number of elements in Ared
  for (j = 0; j < work->data->A->p[work->data->A->n]; j++) {
    if ((A_to_Alow[work->data->A->i[j]] != -1) ||
        (A_to_Aupp[work->data->A->i[j]] != -1)) Ared_nnz++;
  }

  // Form Ared
  // Ared = vstack[Alow, Aupp]
  work->pol->Ared = csc_spalloc(work->pol->n_low + work->pol->n_upp,
                                work->data->n, Ared_nnz, 1, 0);
  if (!(work->pol->Ared)) return -1;
  Ared_nnz = 0; // counter

  for (j = 0; j < work->data->n; j++) { // Cycle over columns of A
    work->pol->Ared->p[j] = Ared_nnz;

    for (ptr = work->data->A->p[j]; ptr < work->data->A->p[j + 1]; ptr++) {
      // Cycle over elements in j-th column
      if (A_to_Alow[work->data->A->i[ptr]] != -1) {
        // Lower-active rows of A
        work->pol->Ared->i[Ared_nnz] =
          A_to_Alow[work->data->A->i[ptr]];
        work->pol->Ared->x[Ared_nnz++] = work->data->A->x[ptr];
      } else if (A_to_Aupp[work->data->A->i[ptr]] != -1) {
        // Upper-active rows of A
        work->pol->Ared->i[Ared_nnz] = A_to_Aupp[work->data->A->i[ptr]] \
                                       + work->pol->n_low;
        work->pol->Ared->x[Ared_nnz++] = work->data->A->x[ptr];
      }
    }
  }

  // Update the last element in Ared->p
  work->pol->Ared->p[work->data->n] = Ared_nnz;

  // Return number of rows in Ared
  return work->pol->n_low + work->pol->n_upp;
}

/**
 * Form reduced right-hand side rhs_red = vstack[-q, l_low, u_upp]
 * @param  work Workspace
 * @param  rhs  right-hand-side
 * @return      reduced rhs
 */
static void form_rhs_red(OSQPWorkspace *work, OSQPVectorf *rhs) {
  c_int j;

  c_float* rhsv = OSQPVectorf_data(rhs);
  c_float* q   = OSQPVectorf_data(work->data->q);
  c_float* l   = OSQPVectorf_data(work->data->l);
  c_float* u   = OSQPVectorf_data(work->data->u);
  c_int* Alow_to_A = OSQPVectori_data(work->pol->Alow_to_A);
  c_int* Aupp_to_A = OSQPVectori_data(work->pol->Aupp_to_A);

  for(j = 0; j < work->data->n; j++){
    rhsv[j] = -q[j];
  }

  for (j = 0; j < work->pol->n_low; j++) { // l_low
    rhsv[work->data->n + j] = l[Alow_to_A[j]];
  }

  for (j = 0; j < work->pol->n_upp; j++) { // u_upp
    rhsv[work->data->n + work->pol->n_low + j] = u[Aupp_to_A[j]];
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
static c_int iterative_refinement(OSQPWorkspace *work,
                                  LinSysSolver  *p,
                                  c_float       *z,
                                  c_float       *b) {
  c_int i, j, n;
  c_float *dz;
  c_float *rhs;

  if (work->settings->polish_refine_iter > 0) {

    // Assign dimension n
    n = work->data->n + work->pol->Ared->m;

    // Allocate dz and rhs vectors
    dz  = (c_float *)c_malloc(sizeof(c_float) * n);
    rhs = (c_float *)c_malloc(sizeof(c_float) * n);

    if (!dz || !rhs) {
      return osqp_error(OSQP_MEM_ALLOC_ERROR);
    } else {
      for (i = 0; i < work->settings->polish_refine_iter; i++) {
        // Form the RHS for the iterative refinement:  b - K*z
        prea_vec_copy(b, rhs, n);

        // Upper Part: R^{n}
        // -= Px (upper triang)
        mat_vec(work->data->P, z, rhs, -1);

        // -= Px (lower triang)
        mat_tpose_vec(work->data->P, z, rhs, -1, 1);

        // -= Ared'*y_red
        mat_tpose_vec(work->pol->Ared, z + work->data->n, rhs, -1, 0);

        // Lower Part: R^{m}
        mat_vec(work->pol->Ared, z, rhs + work->data->n, -1);

        // Solve linear system. Store solution in rhs
        p->solve(p, rhs);

        // Update solution
        for (j = 0; j < n; j++) {
          z[j] += rhs[j];
        }
      }
    }
    if (dz)  c_free(dz);
    if (rhs) c_free(rhs);
  }
  return 0;
}

/**
 * Compute dual variable y from yred
 * @param work Workspace
 * @param yred Dual variables associated to active constraints
 */
static void get_ypol_from_yred(OSQPWorkspace *work, OSQPVectorf *yred_vf) {

  c_int j;

  c_int* A_to_Alow = OSQPVectori_data(work->pol->A_to_Alow);
  c_int* A_to_Aupp = OSQPVectori_data(work->pol->A_to_Aupp);
  c_float* y       = OSQPVectorf_data(work->pol->y);
  c_float* yred    = OSQPVectorf_data(yred_vf);

  // If there are no active constraints
  if (work->pol->n_low + work->pol->n_upp == 0) {
    OSQPVectorf_set_scalar(work->pol->y, 0.);
    return;
  }

  // NB: yred = vstack[ylow, yupp]
  for (j = 0; j < work->data->m; j++) {
    if (A_to_Alow[j] != -1) {
      // lower-active
      y[j] = yred[A_to_Alow[j]];
    } else if (A_to_Aupp[j] != -1) {
      // upper-active
      y[j] = yred[A_to_Aupp[j] + work->pol->n_low];
    } else {
      // inactive
      y[j] = 0.0;
    }
  }
}

c_int polish(OSQPWorkspace *work) {
  c_int mred, polish_successful, exitflag;
  OSQPVectorf *rhs_red;
  LinSysSolver *plsh;
  OSQPVectorf *pol_sol; // Polished solution

#ifdef PROFILING
  osqp_tic(work->timer); // Start timer
#endif /* ifdef PROFILING */

  // Form Ared by assuming the active constraints and store in work->pol->Ared
  mred = form_Ared(work);
  if (mred < 0) { // work->pol->red = OSQP_NULL
    // Polishing failed
    work->info->status_polish = -1;

    return -1;
  }

  // Form and factorize reduced KKT
  exitflag = init_linsys_solver(&plsh, work->data->P, work->pol->Ared,
                                work->settings->delta, OSQP_NULL,
                                work->settings->linsys_solver, 1);

  if (exitflag) {
    // Polishing failed
    work->info->status_polish = -1;

    // Memory clean-up
    if (work->pol->Ared) csc_spfree(work->pol->Ared);

    return 1;
  }

  // Form reduced right-hand side rhs_red
  rhs_red = OSQPVectorf_malloc(work->data->n + mred);
  if (!rhs_red) {
    // Polishing failed
    work->info->status_polish = -1;

    // Memory clean-up
    csc_spfree(work->pol->Ared);

    return -1;
  }
  form_rhs_red(work, rhs_red);

  pol_sol = OSQPVectorf_copy_new(rhs_red);

  if (!pol_sol) {
    // Polishing failed
    work->info->status_polish = -1;

    // Memory clean-up
    csc_spfree(work->pol->Ared);
    OSQPVectorf_free(rhs_red);

    return -1;
  }

  // Solve the reduced KKT system
  plsh->solve(plsh, OSQPVectorf_data(pol_sol));

  // Perform iterative refinement to compensate for the regularization error
  exitflag = iterative_refinement(work, plsh, OSQPVectorf_data(pol_sol), OSQPVectorf_data(rhs_red));

  if (exitflag) {
    // Polishing failed
    work->info->status_polish = -1;

    // Memory clean-up
    csc_spfree(work->pol->Ared);
    OSQPVectorf_free(rhs_red);
    OSQPVectorf_free(pol_sol);

    return -1;
  }

  // Store the polished solution (x,z,y)
  OSQPVectorf_copy(work->pol->x, pol_sol);   // pol->x
  mat_vec(work->data->A, OSQPVectorf_data(work->pol->x), OSQPVectorf_data(work->pol->z), 0); // pol->z
  get_ypol_from_yred(work, pol_sol + work->data->n);     // pol->y

  // Ensure (z,y) satisfies normal cone constraint
  project_normalcone(work, work->pol->z, work->pol->y);

  // Compute primal and dual residuals at the polished solution
  update_info(work, 0, 1, 1);

  // Check if polish was successful
  polish_successful = (work->pol->pri_res < work->info->pri_res &&
                       work->pol->dua_res < work->info->dua_res) || // Residuals
                                                                    // are
                                                                    // reduced
                      (work->pol->pri_res < work->info->pri_res &&
                       work->info->dua_res < 1e-10) ||              // Dual
                                                                    // residual
                                                                    // already
                                                                    // tiny
                      (work->pol->dua_res < work->info->dua_res &&
                       work->info->pri_res < 1e-10);                // Primal
                                                                    // residual
                                                                    // already
                                                                    // tiny

  if (polish_successful) {
    // Update solver information
    work->info->obj_val       = work->pol->obj_val;
    work->info->pri_res       = work->pol->pri_res;
    work->info->dua_res       = work->pol->dua_res;
    work->info->status_polish = 1;

    // Update (x, z, y) in ADMM iterations
    // NB: z needed for warm starting
    OSQPVectorf_copy(work->x, work->pol->x);
    OSQPVectorf_copy(work->z, work->pol->z);
    OSQPVectorf_copy(work->y, work->pol->y);

    // Print summary
#ifdef PRINTING

    if (work->settings->verbose) print_polish(work);
#endif /* ifdef PRINTING */
  } else { // Polishing failed
    work->info->status_polish = -1;

    // TODO: Try to find a better solution on the line connecting ADMM
    //       and polished solution
  }

  // Memory clean-up
  plsh->free(plsh);

  // Checks that they are not NULL are already performed earlier
  csc_spfree(work->pol->Ared);
  OSQPVectorf_free(rhs_red);
  OSQPVectorf_free(pol_sol);

  return 0;
}
