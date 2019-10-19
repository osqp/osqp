#ifndef CSR_TYPE_H
# define CSR_TYPE_H


#include <cusparse_v2.h>
#include "osqp_api_types.h"   /* --> c_int, c_float */


/* CSR matrix structure */
struct csr_t {
  c_int               m;          ///< number of rows
  c_int               n;          ///< number of columns
  c_int              *row_ptr;    ///< row pointers (size m+1)
  c_int              *row_ind;    ///< uncompressed row indices (size nnz), NULL if not needed 
  c_int              *col_ind;    ///< column indices (size nnz)
  c_float            *val;        ///< numerical values (size nnz)
  c_int               nnz;        ///< number of non-zero entries in matrix

  void               *buffer;
  size_t              bufferSizeInBytes;
  cusparseAlgMode_t   alg;
  cusparseMatDescr_t  MatDescription;
};


#endif /* ifndef CSR_TYPE_H */

