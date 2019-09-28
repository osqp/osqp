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
 * h_res = sum(d_x) / n
 */
void cuda_vec_mean(const c_float *d_x,
                   c_int          n,
                   c_float       *h_res);

void cuda_vec_prod(const c_float *d_a,
                   const c_float *d_b,
                   c_int          n,
                   c_float       *h_res);

# ifdef __cplusplus
}
# endif /* ifdef __cplusplus */


#endif /* ifndef CUDA_LIN_ALG_H */
