#ifndef REDUCED_KKT_H_
#define REDUCED_KKT_H_

#include "algebra_matrix.h"
#include "algebra_vector.h"

#ifdef __cplusplus
extern "C" {
#endif

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

/**
 * Compute the right-hand side of the reduced KKT system:
 *     b1 = b1 + A' (rho.*b2)
 * where b1 and b2 are the upper and lower parts of the normal KKT
 * RHS, respectively.
 *
 * Note: This function overwrites b1 with the output value.
 *
 * @param A        The A matrix in the KKT matrix
 * @param rho_vec  The vector containing the rho values
 * @param b1       The upper part of the normal KKT RHS
 * @param b2       The lower part of the normal KKT RHS
 * @param work     Temporary work vector (same size as b2)
 */
void reduced_kkt_compute_rhs(const OSQPMatrix*  A,
                             const OSQPVectorf* rho_vec,
                                   OSQPVectorf* b1,
                             const OSQPVectorf* b2,
                                   OSQPVectorf* work);

#ifdef __cplusplus
}
#endif

#endif /* REDUCED_SYSTEM_H_ */
