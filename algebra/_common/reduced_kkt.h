#ifndef REDUCED_KKT_H_
#define REDUCED_KKT_H_

#include "algebra_matrix.h"
#include "algebra_vector.h"

/**
 * Mathematical functions for performing operations using the reduced
 * KKT system.
 */

/**
 * Compute a matrix-vector multiplication with the matrix from the
 * reduced KKT:
 *   v = (P + sigma*I + A'*diag(rho)*A)*x
 *
 * @param P       The P matrix in the KKT matrix
 * @param A       The A matrix in the KKT matrix
 * @param rho_vec The vector containing the rho values
 * @param sigma   The value of sigma
 * @param x       The vector to multiply against
 * @param v       The output vector
 * @param work    Temporary work vector (same size as rho_vec)
 */
void reduced_kkt_mv_times(const OSQPMatrix*  P,
                          const OSQPMatrix*  A,
                          const OSQPVectorf* rho_vec,
                                OSQPFloat    sigma,
                          const OSQPVectorf* x,
                                OSQPVectorf* v,
                                OSQPVectorf* work);

/**
 * Compute the diagonal (and its inverse) of the reduced KKT:
 *   P + sigma*I + A'*diag(rho)*A
 *
 * @param P        The P matrix in the KKT matrix
 * @param A        The A matrix in the KKT matrix
 * @param rho_vec  The vector containing the rho values
 * @param sigma    The value of sigma
 * @param diag     The vector to store the diagonal in
 * @param diag_inv The vector to store the inverse of the diagonal in
 */
void reduced_kkt_diagonal(const OSQPMatrix*  P,
                          const OSQPMatrix*  A,
                          const OSQPVectorf* rho_vec,
                                OSQPFloat    sigma,
                                OSQPVectorf* diag,
                                OSQPVectorf* diag_inv);


#endif /* REDUCED_SYSTEM_H_ */
