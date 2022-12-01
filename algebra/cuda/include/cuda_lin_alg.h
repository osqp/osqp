/**
 *  Copyright (c) 2019-2021 ETH Zurich, Automatic Control Lab,
 *  Michel Schubiger, Goran Banjac.
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#ifndef CUDA_LIN_ALG_H
# define CUDA_LIN_ALG_H

#include <cusparse.h>
#include "algebra_types.h"


/*******************************************************************************
 *                           Vector Functions                                  *
 *******************************************************************************/

void cuda_vec_create(cusparseDnVecDescr_t* vec,
                     const OSQPFloat*      d_x,
                     OSQPInt               n);

void cuda_vec_destroy(cusparseDnVecDescr_t vec);

/*
 * d_y[i] = d_x[i] for i in [0,n-1]
*/
void cuda_vec_copy_d2d(OSQPFloat*       d_y,
                       const OSQPFloat* d_x,
                       OSQPInt          n);

/*
 * d_y[i] = h_x[i] for i in [0,n-1]
*/
void cuda_vec_copy_h2d(OSQPFloat*       d_y,
                       const OSQPFloat* h_x,
                       OSQPInt          n);

/*
 * h_y[i] = d_x[i] for i in [0,n-1]
*/
void cuda_vec_copy_d2h(OSQPFloat*       h_y,
                       const OSQPFloat* d_x,
                       OSQPInt          n);

/*
 * d_y[i] = h_x[i] for i in [0,n-1] (integers)
*/
void cuda_vec_int_copy_h2d(OSQPInt*       d_y,
                           const OSQPInt* h_x,
                           OSQPInt        n);

/*
 * h_y[i] = d_x[i] for i in [0,n-1] (integers)
*/
void cuda_vec_int_copy_d2h(OSQPInt*       h_y,
                           const OSQPInt* d_x,
                           OSQPInt        n);

/**
 * d_a[i] = sc for i in [0,n-1]
 */
void cuda_vec_set_sc(OSQPFloat* d_a,
                     OSQPFloat  sc,
                     OSQPInt    n);

/**
 *           | sc_if_neg   d_test[i]  < 0
 * d_a[i] = <  sc_if_zero  d_test[i] == 0   for i in [0,n-1]
 *           | sc_if_pos   d_test[i]  > 0
 */
void cuda_vec_set_sc_cond(OSQPFloat*     d_a,
                          const OSQPInt* d_test,
                          OSQPFloat      sc_if_neg,
                          OSQPFloat      sc_if_zero,
                          OSQPFloat      sc_if_pos,
                          OSQPInt        n);

/**
 * d_a[i] *= sc for i in [0,n-1]
 */
void cuda_vec_mult_sc(OSQPFloat* d_a,
                      OSQPFloat  sc,
                      OSQPInt    n);

/**
 * d_x[i] = sca * d_a[i] + scb * d_b[i] for i in [0,n-1]
 */
void cuda_vec_add_scaled(OSQPFloat*       d_x,
                         const OSQPFloat* d_a,
                         const OSQPFloat* d_b,
                         OSQPFloat        sca,
                         OSQPFloat        scb,
                         OSQPInt          n);

/**
 * d_x[i] = sca * d_a[i] + scb * d_b[i] + scc * d_c[i] for i in [0,n-1]
 */
void cuda_vec_add_scaled3(OSQPFloat*       d_x,
                          const OSQPFloat* d_a,
                          const OSQPFloat* d_b,
                          const OSQPFloat* d_c,
                          OSQPFloat        sca,
                          OSQPFloat        scb,
                          OSQPFloat        scc,
                          OSQPInt          n);

/**
 * h_res = |d_x|_inf
 */
void cuda_vec_norm_inf(const OSQPFloat* d_x,
                             OSQPInt    n,
                             OSQPFloat* h_res);

/**
 * res = |d_x|_2
 */
void cuda_vec_norm_2(const OSQPFloat* d_x,
                           OSQPInt    n,
                           OSQPFloat* h_res);

/**
 * h_res = |S*v|_inf
 */
void cuda_vec_scaled_norm_inf(const OSQPFloat* d_S,
                              const OSQPFloat* d_v,
                                    OSQPInt    n,
                                    OSQPFloat* h_res);

/**
 * h_res = |d_a - d_b|_inf
 */
void cuda_vec_diff_norm_inf(const OSQPFloat* d_a,
                            const OSQPFloat* d_b,
                                  OSQPInt    n,
                                  OSQPFloat* h_res);

/**
 * h_res = sum(|d_x|)
 */
void cuda_vec_norm_1(const OSQPFloat* d_x,
                           OSQPInt    n,
                           OSQPFloat* h_res);

/**
 * h_res = d_a' * d_b
 */
void cuda_vec_prod(const OSQPFloat* d_a,
                   const OSQPFloat* d_b,
                         OSQPInt    n,
                         OSQPFloat* h_res);

/**
 *          | d_a' * max(d_b, 0)  sign ==  1
 * h_res = <  d_a' * min(d_b, 0)  sign == -1
 *          | d_a' * d_b          otherwise
 */
void cuda_vec_prod_signed(const OSQPFloat* d_a,
                          const OSQPFloat* d_b,
                                OSQPInt    sign,
                                OSQPInt    n,
                                OSQPFloat* h_res);

/**
 * d_c[i] = d_a[i] * d_b[i] for i in [0,n-1]
 */
void cuda_vec_ew_prod(OSQPFloat*       d_c,
                      const OSQPFloat* d_a,
                      const OSQPFloat* d_b,
                      OSQPInt          n);

/**
 * h_res = all(a == b)
 */
void cuda_vec_eq(const OSQPFloat* a,
                 const OSQPFloat* b,
                       OSQPFloat  tol,
                       OSQPInt    n,
                       OSQPInt*   h_res);

/**
 * h_res = all(d_l <= d_u)
 */
void cuda_vec_leq(const OSQPFloat* d_l,
                  const OSQPFloat* d_u,
                        OSQPInt    n,
                        OSQPInt*   h_res);

/**
 * d_x[i] = min( max(d_z[i], d_l[i]), d_u[i] ) for i in [0,n-1]
 */
void cuda_vec_bound(OSQPFloat*       d_x,
                    const OSQPFloat* d_z,
                    const OSQPFloat* d_l,
                    const OSQPFloat* d_u,
                    OSQPInt          n);

/**
 *           | 0.0               d_l < -infval AND d_u > +infval
 * d_y[i] = <  min(d_y[i], 0.0)  d_u > +infval
 *           | max(d_y[i], 0.0)  d_l < -infval
 */
void cuda_vec_project_polar_reccone(OSQPFloat*       d_y,
                                    const OSQPFloat* d_l,
                                    const OSQPFloat* d_u,
                                    OSQPFloat        infval,
                                    OSQPInt          n);

/**
 *          | d_y[i] \in [-tol,tol]  d_l[i] > -infval AND d_u[i] < +infval
 * h_res = <  d_y[i] < +tol          d_l[i] < -infval AND d_u[i] < +infval
 *          | d_y[i] > -tol          d_l[i] > -infval AND d_u[i] > +infval
 */
void cuda_vec_in_reccone(const OSQPFloat* d_y,
                         const OSQPFloat* d_l,
                         const OSQPFloat* d_u,
                               OSQPFloat  infval,
                               OSQPFloat  tol,
                               OSQPInt    n,
                               OSQPInt*   h_res);

/**
 * d_b[i] = 1 / d_a[i] for i in [0,n-1]
 */
void cuda_vec_reciprocal(OSQPFloat*       d_b,
                         const OSQPFloat* d_a,
                         OSQPInt          n);

/**
 * d_a[i] = sqrt(d_a[i]) for i in [0,n-1]
 */
void cuda_vec_sqrt(OSQPFloat* d_a,
                   OSQPInt    n);

/**
 * d_c[i] = max(d_a[i], d_b[i]) for i in [0,n-1]
 */
void cuda_vec_max(OSQPFloat*       d_c,
                  const OSQPFloat* d_a,
                  const OSQPFloat* d_b,
                  OSQPInt          n);

/**
 * d_c[i] = min(d_a[i], d_b[i]) for i in [0,n-1]
 */
void cuda_vec_min(OSQPFloat*       d_c,
                  const OSQPFloat* d_a,
                  const OSQPFloat* d_b,
                  OSQPInt          n);

void cuda_vec_bounds_type(OSQPInt*         d_iseq,
                          const OSQPFloat* d_l,
                          const OSQPFloat* d_u,
                          OSQPFloat        infval,
                          OSQPFloat        tol,
                          OSQPInt          n,
                          OSQPInt*         h_has_changed);

void cuda_vec_set_sc_if_lt(OSQPFloat*       d_x,
                           const OSQPFloat* d_z,
                           OSQPFloat        testval,
                           OSQPFloat        newval,
                           OSQPInt          n);

void cuda_vec_set_sc_if_gt(OSQPFloat*       d_x,
                           const OSQPFloat* d_z,
                           OSQPFloat        testval,
                           OSQPFloat        newval,
                           OSQPInt          n);

void cuda_vec_segmented_sum(const OSQPFloat* d_values,
                            const OSQPInt*   d_keys,
                            OSQPFloat*       d_res,
                            void*            d_buffer,
                            OSQPInt          num_segments,
                            OSQPInt          num_elements);

void cuda_vec_gather(OSQPInt          nnz,
                     const OSQPFloat* d_y,
                     OSQPFloat*       d_xVal,
                     const OSQPInt*   d_xInd);


/*******************************************************************************
 *                           Matrix Functions                                  *
 *******************************************************************************/

/**
 * S = sc * S
 */
void cuda_mat_mult_sc(csr*      S,
                      csr*      At,
                      OSQPFloat sc);

/**
 * S = D * S
 */
void cuda_mat_lmult_diag(csr*             S,
                         csr*             At,
                         const OSQPFloat* d_diag);

/**
 * S = S * D
 */
void cuda_mat_rmult_diag(csr*             S,
                         csr*             At,
                         const OSQPFloat* d_diag);

/**
 * X = S * D
 * X->val values are stored in d_buffer.
 */
void cuda_mat_rmult_diag_new(const csr*       S,
                                   OSQPFloat* d_buffer,
                             const OSQPFloat* d_diag);

/**
 * d_y = alpha * A*d_x + beta*d_y
 */
void cuda_mat_Axpy(const csr*                 A,
                   const cusparseDnVecDescr_t vecx,
                         cusparseDnVecDescr_t vecy,
                         OSQPFloat            alpha,
                         OSQPFloat            beta);

/**
 * d_res[i] = |S_i|_inf where S_i is i-th row of S
 */
void cuda_mat_row_norm_inf(const csr*       S,
                                 OSQPFloat* d_res);


#endif /* ifndef CUDA_LIN_ALG_H */
