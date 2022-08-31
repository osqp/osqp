#ifndef OSQP_API_UTILS_H
#define OSQP_API_UTILS_H

/* Types required by the OSQP utility functions */
# include "osqp_api_types.h"
# include "osqp_export_define.h"

# ifdef __cplusplus
extern "C" {
# endif

/********************
* OSQP User utilities  *
********************/
/**
 * Populates a Compressed-Column-Sparse matrix from existing arrays
   (just assigns the pointers - no malloc or copying is done)
 * @param  M     Matrix pointer
 * @param  m     First dimension
 * @param  n     Second dimension
 * @param  nzmax Maximum number of nonzero elements
 * @param  x     Vector of data
 * @param  i     Vector of row indices
 * @param  p     Vector of column pointers
 */
OSQP_API void csc_set_data(OSQPCscMatrix* M,
                           OSQPInt        m,
                           OSQPInt        n,
                           OSQPInt        nzmax,
                           OSQPFloat*     x,
                           OSQPInt*       i,
                           OSQPInt*       p);

# ifdef __cplusplus
}
# endif

#endif /* ifndef OSQP_API_FUNCTIONS_H */
