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


# ifdef __cplusplus
}
# endif /* ifdef __cplusplus */


#endif /* ifndef CUDA_LIN_ALG_H */
