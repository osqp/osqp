#include "scaling.h"

#if OSQP_EMBEDDED_MODE != 1


// Set values lower than threshold SCALING_REG to 1

OSQPFloat limit_scaling_scalar(OSQPFloat v) {
    v = v < OSQP_MIN_SCALING ? 1.0 : v;
    v = v > OSQP_MAX_SCALING ? OSQP_MAX_SCALING : v;
    return v;
}

void limit_scaling_vector(OSQPVectorf* v) {
  OSQPVectorf_set_scalar_if_lt(v,v,OSQP_MIN_SCALING,1.0);
  OSQPVectorf_set_scalar_if_gt(v,v,OSQP_MAX_SCALING,OSQP_MAX_SCALING);
}

/**
 * Compute infinite norm of the columns of the KKT matrix without forming it
 *
 * The norm is stored in the vector v = (D, E)
 *
 * @param P        Cost matrix
 * @param A        Constraints matrix
 * @param D        Norm of columns related to variables
 * @param D_temp_A Temporary vector for norm of columns of A
 * @param E        Norm of columns related to constraints
 */
void compute_inf_norm_cols_KKT(const OSQPMatrix*  P,
                               const OSQPMatrix*  A,
                                     OSQPVectorf* D,
                                     OSQPVectorf* D_temp_A,
                                     OSQPVectorf* E) {
  // First half
  //  [ P ]
  //  [ A ]
  OSQPMatrix_col_norm_inf(P,D);
  OSQPMatrix_col_norm_inf(A, D_temp_A);
  OSQPVectorf_ew_max_vec(D, D_temp_A, D);

  // Second half
  //  [ A']
  //  [ 0 ]
  OSQPMatrix_row_norm_inf(A,E);
}

OSQPInt scale_data(OSQPSolver* solver) {
  // Scale KKT matrix
  //
  //    [ P   A']
  //    [ A   0 ]
  //
  // with diagonal matrix
  //
  //  S = [ D    ]
  //      [    E ]
  //

  OSQPInt   i;          // Iterations index
  OSQPInt   n;          // Number of variables
  OSQPFloat c_temp;     // Objective function scaling
  OSQPFloat inf_norm_q; // Infinity norm of q

  OSQPSettings*  settings = solver->settings;
  OSQPWorkspace* work     = solver->work;

  n = work->data->n;

  // Initialize scaling to 1
  work->scaling->c = 1.0;
  OSQPVectorf_set_scalar(work->scaling->D,    1.);
  OSQPVectorf_set_scalar(work->scaling->Dinv, 1.);
  OSQPVectorf_set_scalar(work->scaling->E,    1.);
  OSQPVectorf_set_scalar(work->scaling->Einv, 1.);


  for (i = 0; i < settings->scaling; i++) {
    //
    // First Ruiz step
    //

    // Compute norm of KKT columns
    compute_inf_norm_cols_KKT(work->data->P, work->data->A,
                              work->D_temp,
                              work->D_temp_A,
                              work->E_temp);

    // Set to 1 values with 0 norms (avoid crazy scaling)
    limit_scaling_vector(work->D_temp);
    limit_scaling_vector(work->E_temp);

    // Take square root of norms
    OSQPVectorf_ew_sqrt(work->D_temp);
    OSQPVectorf_ew_sqrt(work->E_temp);

    // Copy inverses of D/E over themselves
    OSQPVectorf_ew_reciprocal(work->D_temp, work->D_temp);
    OSQPVectorf_ew_reciprocal(work->E_temp, work->E_temp);

    // Equilibrate matrices P and A and vector q
    // P <- DPD
    OSQPMatrix_lmult_diag(work->data->P,work->D_temp);
    OSQPMatrix_rmult_diag(work->data->P,work->D_temp);

    // A <- EAD
    OSQPMatrix_lmult_diag(work->data->A,work->E_temp);
    OSQPMatrix_rmult_diag(work->data->A,work->D_temp);

    // q <- Dq
    OSQPVectorf_ew_prod(work->data->q, work->data->q, work->D_temp);

    // Update equilibration matrices D and E
    OSQPVectorf_ew_prod(work->scaling->D, work->scaling->D, work->D_temp);
    OSQPVectorf_ew_prod(work->scaling->E, work->scaling->E, work->E_temp);

    //
    // Cost normalization step
    //

    // Compute avg norm of cols of P.
    OSQPMatrix_col_norm_inf(work->data->P, work->D_temp);
    c_temp = OSQPVectorf_norm_1(work->D_temp);
    c_temp = c_temp / n;

    // Compute inf norm of q
    inf_norm_q = OSQPVectorf_norm_inf(work->data->q);

    // If norm_q == 0, set it to 1 (ignore it in the scaling)
    // NB: Using the same function as with vectors here
    inf_norm_q = limit_scaling_scalar(inf_norm_q);

    // Compute max between avg norm of cols of P and inf norm of q
    c_temp = c_max(c_temp, inf_norm_q);

    // Limit scaling (use same function as with vectors)
    c_temp = limit_scaling_scalar(c_temp);

    // Invert scaling c = 1 / cost_measure
    c_temp = 1. / c_temp;

    // Scale P
    OSQPMatrix_mult_scalar(work->data->P,c_temp);

    // Scale q
    OSQPVectorf_mult_scalar(work->data->q, c_temp);

    // Update cost scaling
    work->scaling->c *= c_temp;
  }


  // Store cinv, Dinv, Einv
  work->scaling->cinv = 1. / work->scaling->c;
  OSQPVectorf_ew_reciprocal(work->scaling->Dinv, work->scaling->D);
  OSQPVectorf_ew_reciprocal(work->scaling->Einv, work->scaling->E);


  // Scale problem vectors l, u
  OSQPVectorf_ew_prod(work->data->l, work->data->l, work->scaling->E);
  OSQPVectorf_ew_prod(work->data->u, work->data->u, work->scaling->E);

  return 0;
}

#endif /* if OSQP_EMBEDDED_MODE != 1 */


OSQPInt unscale_data(OSQPSolver* solver) {

  OSQPWorkspace* work     = solver->work;

  // Unscale cost
  OSQPMatrix_mult_scalar(work->data->P, work->scaling->cinv);
  OSQPMatrix_lmult_diag(work->data->P,  work->scaling->Dinv);
  OSQPMatrix_rmult_diag(work->data->P,  work->scaling->Dinv);
  OSQPVectorf_mult_scalar(work->data->q,work->scaling->cinv);
  OSQPVectorf_ew_prod(work->data->q, work->data->q, work->scaling->Dinv);

  // Unscale constraints
  OSQPMatrix_lmult_diag(work->data->A,work->scaling->Einv);
  OSQPMatrix_rmult_diag(work->data->A,work->scaling->Dinv);

  OSQPVectorf_ew_prod(work->data->l,
                      work->data->l,
                      work->scaling->Einv);
  OSQPVectorf_ew_prod(work->data->u,
                      work->data->u,
                      work->scaling->Einv);

  return 0;
}

OSQPInt unscale_solution(OSQPVectorf*       usolx,
                         OSQPVectorf*       usoly,
                         const OSQPVectorf* solx,
                         const OSQPVectorf* soly,
                         OSQPWorkspace*     work) {

  // primal
  OSQPVectorf_ew_prod(usolx,solx,work->scaling->D);

  // dual
  OSQPVectorf_ew_prod(usoly,soly,work->scaling->E);

  OSQPVectorf_mult_scalar(usoly,work->scaling->cinv);
  return 0;
}
