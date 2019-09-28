#ifndef CUDA_LIN_ALG_H
# define CUDA_LIN_ALG_H

#include "osqp_api_types.h"

# ifdef __cplusplus
extern "C" {
# endif // ifdef __cplusplus


/*
 * d_y[i] = d_x[i] for i in [0,n-1]
*/
void cuda_vec_copy_d2d(c_float       *d_y,
                       const c_float *d_x,
                       c_int          n);

/*
 * d_y[i] = h_x[i] for i in [0,n-1]
*/
void cuda_vec_copy_h2d(c_float       *d_y,
                       const c_float *h_x,
                       c_int          n);

/*
 * h_y[i] = d_x[i] for i in [0,n-1]
*/
void cuda_vec_copy_d2h(c_float       *h_y,
                       const c_float *d_x,
                       c_int          n);

/**
 * d_a[i] = sc for i in [0,n-1]
 */
void cuda_vec_set_sc(c_float *d_a,
                     c_float  sc,
                     c_int    n);

/**
 *           | sc_if_neg   d_test[i]  < 0
 * d_a[i] = <  sc_if_zero  d_test[i] == 0   for i in [0,n-1]
 *           | sc_if_pos   d_test[i]  > 0
 */
void cuda_vec_set_sc_cond(c_float     *d_a,
                          const c_int *d_test,
                          c_float      sc_if_neg,
                          c_float      sc_if_zero,
                          c_float      sc_if_pos,
                          c_float      n);

/**
 * d_a[i] *= sc for i in [0,n-1]
 */
void cuda_vec_mult_sc(c_float *d_a,
                      c_float  sc,
                      c_int    n);

/**
 * d_x[i] = sca * d_a[i] + scb * d_b[i] for i in [0,n-1]
 */
void cuda_vec_add_scaled(c_float       *d_x,
                         const c_float *d_a,
                         const c_float *d_b,
                         c_float        sca,
                         c_float        scb,
                         c_int          n);

/**
 * d_x[i] = sca * d_a[i] + scb * d_b[i] + scc * d_c[i] for i in [0,n-1]
 */
void cuda_vec_add_scaled3(c_float       *d_x,
                          const c_float *d_a,
                          const c_float *d_b,
                          const c_float *d_c,
                          c_float        sca,
                          c_float        scb,
                          c_float        scc,
                          c_int          n);

/**
 * h_res = |d_x|_inf
 */
void cuda_vec_norm_inf(const c_float *d_x,
                       c_int          n,
                       c_float       *h_res);

/**
 * h_res = |d_x|_1
 */
void cuda_vec_norm_1(const c_float *d_x,
                     c_int          n,
                     c_float       *h_res);

/**
 * h_res = |S*v|_inf
 */
void cuda_vec_scaled_norm_inf(const c_float *d_S,
                              const c_float *d_v,
                              c_int          n,
                              c_float       *h_res);

/**
 * h_res = |d_a - d_b|_inf
 */
void cuda_vec_diff_norm_inf(const c_float *d_a,
                            const c_float *d_b,
                            c_int          n,
                            c_float       *h_res);

/**
 * h_res = sum(d_x) / n
 */
void cuda_vec_mean(const c_float *d_x,
                   c_int          n,
                   c_float       *h_res);

/**
 * h_res = d_a' * d_b
 */
void cuda_vec_prod(const c_float *d_a,
                   const c_float *d_b,
                   c_int          n,
                   c_float       *h_res);

/**
 *          | d_a' * max(d_b, 0)  sign ==  1
 * h_res = <  d_a' * min(d_b, 0)  sign == -1
 *          | d_a' * d_b          otherwise
 */
void cuda_vec_prod_signed(const c_float *d_a,
                          const c_float *d_b,
                          c_int          sign,
                          c_int          n,
                          c_float       *h_res);

/**
 * d_c[i] = d_a[i] * d_b[i] for i in [0,n-1]
 */
void cuda_vec_ew_prod(c_float       *d_c,
                      const c_float *d_a,
                      const c_float *d_b,
                      c_int          n);

/**
 * h_res = all(d_l <= d_u)
 */
void cuda_vec_leq(const c_float *d_l,
                  const c_float *d_u,
                  c_int          n,
                  c_int         *h_res);

/**
 * d_x[i] = min( max(d_z[i], d_l[i]), d_u[i] ) for i in [0,n-1]
 */
void cuda_vec_bound(c_float       *d_x,
                    const c_float *d_z,
                    const c_float *d_l,
                    const c_float *d_u,
                    c_int          n);

/**
 *           | 0.0               d_l < -infval AND d_u > +infval
 * d_y[i] = <  min(d_y[i], 0.0)  d_u > +infval
 *           | max(d_y[i], 0.0)  d_l < -infval
 */
void cuda_vec_project_polar_reccone(c_float       *d_y,
                                    const c_float *d_l,
                                    const c_float *d_u,
                                    c_float        infval,
                                    c_int          n);

/**
 *          | d_y[i] \in [-tol,tol]  d_l[i] > -infval AND d_u[i] < +infval
 * h_res = <  d_y[i] < +tol          d_l[i] < -infval AND d_u[i] < +infval
 *          | d_y[i] > -tol          d_l[i] > -infval AND d_u[i] > +infval
 */
void cuda_vec_in_reccone(const c_float *d_y,
                         const c_float *d_l,
                         const c_float *d_u,
                         c_float        infval,
                         c_float        tol,
                         c_int          n,
                         c_int         *h_res);

/**
 * d_b[i] = 1 / d_a[i] for i in [0,n-1]
 */
void cuda_vec_reciprocal(c_float       *d_b,
                         const c_float *d_a,
                         c_int          n);

/**
 * d_a[i] = sqrt(d_a[i]) for i in [0,n-1]
 */
void cuda_vec_sqrt(c_float *d_a,
                      c_int    n);

/**
 * d_c[i] = max(d_a[i], d_b[i]) for i in [0,n-1]
 */
void cuda_vec_max(c_float       *d_c,
                  const c_float *d_a,
                  const c_float *d_b,
                  c_int          n);

/**
 * d_c[i] = min(d_a[i], d_b[i]) for i in [0,n-1]
 */
void cuda_vec_min(c_float       *d_c,
                  const c_float *d_a,
                  const c_float *d_b,
                  c_int          n);

void cuda_vec_bounds_type(c_int         *d_iseq,
                          const c_float *d_l,
                          const c_float *d_u,
                          c_float        infval,
                          c_float        tol,
                          c_int          n,
                          c_int         *h_has_changed);

void cuda_vec_set_sc_if_lt(c_float       *d_x,
                           const c_float *d_z,
                           c_float        testval,
                           c_float        newval,
                           c_int          n);

void cuda_vec_set_sc_if_gt(c_float       *d_x,
                           const c_float *d_z,
                           c_float        testval,
                           c_float        newval,
                           c_int          n);

# ifdef __cplusplus
}
# endif /* ifdef __cplusplus */


#endif /* ifndef CUDA_LIN_ALG_H */
