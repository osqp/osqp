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

