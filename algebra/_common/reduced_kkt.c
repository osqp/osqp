#include "reduced_kkt.h"
#include "algebra_matrix.h"
#include "algebra_vector.h"


/*
 * Compute v = (P + sigma*I + A'*diag(rho)*A)*x
 */
void reduced_kkt_mv_times(const OSQPMatrix*  P,
                          const OSQPMatrix*  A,
                          const OSQPVectorf* rho_vec,
                                OSQPFloat    sigma,
                          const OSQPVectorf* x,
                                OSQPVectorf* v,
                                OSQPVectorf* work) {

  /* x and v are successive columns of tmp in the MKL CG, unclear if
     we can overwrite x, so avoid it by using the work vector. */
  OSQPMatrix_Axpy(A, x, work, 1.0, 0.0);    /* scratch space for (rho)*A*x */
  OSQPVectorf_ew_prod(work, work, rho_vec);
  OSQPVectorf_copy(v, x);
  OSQPMatrix_Axpy(P, x, v, 1.0, sigma);     /* v = (P + sigma*I) x */
  OSQPMatrix_Atxpy(A, work, v, 1.0, 1.0);
}


/*
 * Compute the diagonal and its inverse for
 *  P + sigma*I + A'*diag(rho)*A
 */
void reduced_kkt_diagonal(const OSQPMatrix*  P,
                          const OSQPMatrix*  A,
                          const OSQPVectorf* rho_vec,
                                OSQPFloat    sigma,
                                OSQPVectorf* diag,
                                OSQPVectorf* diag_inv) {

  /* 1st part: sigma */
  OSQPVectorf_set_scalar(diag, sigma);

  /* 2nd part: P matrix diagonal */
  OSQPMatrix_extract_diag(P, diag_inv);
  OSQPVectorf_plus(diag, diag, diag_inv);

  /* 3rd part: Diagonal of At*rho*A */
  OSQPMatrix_AtDA_extract_diag(A, rho_vec, diag_inv);
  OSQPVectorf_plus(diag, diag, diag_inv);

  /* 4th part: Invert the diagonal */
  OSQPVectorf_ew_reciprocal(diag_inv, diag);
}


void reduced_kkt_compute_rhs(const OSQPMatrix*  A,
                             const OSQPVectorf* rho_vec,
                                   OSQPVectorf* b1,
                             const OSQPVectorf* b2,
                                   OSQPVectorf* work) {

  /* 1st part: Set work = rho.*b2 */
  OSQPVectorf_ew_prod(work, b2, rho_vec);

  /* 2nd part: Compute b1 = b1 + A' (rho.*b2) */
  OSQPMatrix_Atxpy(A, work, b1, 1.0, 1.0);
}
