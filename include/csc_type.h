#ifndef CSC_TYPE_H
# define CSC_TYPE_H

#include "osqp_api_types.h"


/**
 *  Matrix in compressed-column form.
 *  The structure is used internally to store matrices in the triplet form as well,
 *  but the API requires that the matrices are in the CSC format. 
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


#endif /* ifndef CSC_TYPE_H */
