#ifndef CSC_TYPE_H
# define CSC_TYPE_H

# ifdef __cplusplus
extern "C" {
# endif // ifdef __cplusplus

#include "osqp_api_types.h"  //for c_int, c_float

/**
 *  Matrix data in csc or triplet form
 */
typedef struct {
  c_int    m;     ///< number of rows
  c_int    n;     ///< number of columns
  c_int   *p;     ///< column pointers (size n+1); col indices (size nzmax) starting from 0 for triplet format
  c_int   *i;     ///< row indices, size nzmax starting from 0
  c_float *x;     ///< numerical values, size nzmax
  c_int    nzmax; ///< maximum number of entries
  c_int    nz;    ///< number of entries in triplet matrix, -1 for csc
} csc;


# ifdef __cplusplus
}
# endif // ifdef __cplusplus

#endif // ifndef CSC_TYPE_H
