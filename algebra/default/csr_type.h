#ifndef CSR_TYPE_H
# define CSR_TYPE_H

# ifdef __cplusplus
extern "C" {
# endif /* ifdef __cplusplus */


#include <cusparse_v2.h>
#include "osqp_api_types.h"   /* --> c_int, c_float */


/* CSR matrix structure */
typedef struct {
  c_int               m;          ///< number of rows
  c_int               n;          ///< number of columns
  c_int              *row_ptr;    ///< column pointers (size m+1)
  c_int              *row_ind;    ///< uncompressed row indices (size nnz), NULL if not needed 
  c_int              *col_ind;    ///< row indices (size nnz)
  c_float            *val;        ///< numerical values (size nnz)
  c_int               nnz;        ///< number of non-zero entries in matrix

  void               *buffer;
  size_t              bufferSizeInBytes;
  cusparseAlgMode_t   alg;
  cusparseMatDescr_t  MatDescription;
} csr;


# ifdef __cplusplus
}
# endif /* ifdef __cplusplus */

#endif /* ifndef CSR_TYPE_H */
