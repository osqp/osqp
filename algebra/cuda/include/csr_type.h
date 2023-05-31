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


#include <cusparse.h>
#include "osqp_api_types.h"   /* --> OSQPInt, OSQPFloat */


/* CSR matrix structure */
struct csr_t {
  OSQPInt m;            ///< number of rows
  OSQPInt n;            ///< number of columns
  OSQPInt nnz;          ///< number of non-zero entries

  OSQPInt*   row_ptr;   ///< row pointers (size m+1)
  OSQPInt*   row_ind;   ///< uncompressed row indices (size nnz), NULL if not needed
  OSQPInt*   col_ind;   ///< column indices (size nnz)
  OSQPFloat* val;       ///< numerical values (size nnz)

  size_t               SpMatBufferSize;
  void*                SpMatBuffer;
  cusparseSpMatDescr_t SpMatDescr;
};


#endif /* ifndef CSR_TYPE_H */
