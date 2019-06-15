#include "scaling.h"

#if EMBEDDED != 1


// Set values lower than threshold SCALING_REG to 1

c_float limit_scaling_scalar(c_float v) {
    v = v < MIN_SCALING ? 1.0 : v;
    v = v > MAX_SCALING ? MAX_SCALING : v;
    return v;
}

void limit_scaling_vector(OSQPVectorf* v) {
  OSQPVectorf_set_scalar_if_lt(v,v,MIN_SCALING,1.0);
  OSQPVectorf_set_scalar_if_gt(v,v,MAX_SCALING,MAX_SCALING);
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
 * @param n        Dimension of KKT matrix
 */
void compute_inf_norm_cols_KKT(const csc *P, const csc *A,
                               c_float *D, c_float *D_temp_A,
                               c_float *E, c_int n) {
  // First half
  //  [ P ]
  //  [ A ]
  mat_inf_norm_cols_sym_triu(P, D);
  mat_inf_norm_cols(A, D_temp_A);
  vec_ew_max_vec(D, D_temp_A, D, n);

  // Second half
  //  [ A']
  //  [ 0 ]
  mat_inf_norm_rows(A, E);
}

c_int scale_data(OSQPWorkspace *work) {
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

  c_int   i;          // Iterations index
  c_int   n, m;       // Number of constraints and variables
  c_float c_temp;     // Cost function scaling
  c_float inf_norm_q; // Infinity norm of q

  n = work->data->n;
  m = work->data->m;

  // Initialize scaling to 1
  work->scaling->c = 1.0;
  OSQPVectorf_set_scalar(work->scaling->D,    1.);
  OSQPVectorf_set_scalar(work->scaling->Dinv, 1.);
  OSQPVectorf_set_scalar(work->scaling->E,    1.);
  OSQPVectorf_set_scalar(work->scaling->Einv, 1.);


  for (i = 0; i < work->settings->scaling; i++) {
    //
    // First Ruiz step
    //

    // Compute norm of KKT columns
    compute_inf_norm_cols_KKT(work->data->P, work->data->A,
                              OSQPVectorf_data(work->D_temp),
                              OSQPVectorf_data(work->D_temp_A),
                              OSQPVectorf_data(work->E_temp), n);

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
    mat_premult_diag(work->data->P, OSQPVectorf_data(work->D_temp));
    mat_postmult_diag(work->data->P, OSQPVectorf_data(work->D_temp));

    // A <- EAD
    mat_premult_diag(work->data->A, OSQPVectorf_data(work->E_temp));
    mat_postmult_diag(work->data->A, OSQPVectorf_data(work->D_temp));

    // q <- Dq
    OSQPVectorf_ew_prod(work->data->q, work->data->q, work->D_temp);

    // Update equilibration matrices D and E
    OSQPVectorf_ew_prod(work->scaling->D, work->scaling->D, work->D_temp);
    OSQPVectorf_ew_prod(work->scaling->E, work->scaling->E, work->E_temp);

    //
    // Cost normalization step
    //

    // Compute avg norm of cols of P
    mat_inf_norm_cols_sym_triu(work->data->P, OSQPVectorf_data(work->D_temp));
    c_temp = OSQPVectorf_mean(work->D_temp);

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
    mat_mult_scalar(work->data->P, c_temp);

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

#endif // EMBEDDED

c_int unscale_data(OSQPWorkspace *work) {
  // Unscale cost
  mat_mult_scalar(work->data->P, work->scaling->cinv);
  mat_premult_diag(work->data->P, OSQPVectorf_data(work->scaling->Dinv));
  mat_postmult_diag(work->data->P, OSQPVectorf_data(work->scaling->Dinv));
  OSQPVectorf_mult_scalar(work->data->q, work->scaling->cinv);
  OSQPVectorf_ew_prod(work->data->q, work->data->q, work->scaling->Dinv);

  // Unscale constraints
  mat_premult_diag(work->data->A, OSQPVectorf_data(work->scaling->Einv));
  mat_postmult_diag(work->data->A, OSQPVectorf_data(work->scaling->Dinv));

  OSQPVectorf_ew_prod(work->data->l,
                      work->data->l,
                      work->scaling->Einv);
  OSQPVectorf_ew_prod(work->data->u,
                      work->data->u,
                      work->scaling->Einv);

  return 0;
}

c_int unscale_solution(OSQPWorkspace *work) {
  // primal
  OSQPVectorf_ew_prod(work->solution->x,
                      work->solution->x,
                      work->scaling->D);

  // dual
  OSQPVectorf_ew_prod(work->solution->y,
                      work->solution->y,
                      work->scaling->E);

  OSQPVectorf_mult_scalar(work->solution->y,
                          work->scaling->cinv);
  return 0;
}
