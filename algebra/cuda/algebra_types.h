/**
 *  Copyright (c) 2019-2020 ETH Zurich, Automatic Control Lab,
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

#ifndef ALGEBRA_TYPES_H
# define ALGEBRA_TYPES_H

# ifdef __cplusplus
extern "C" {
# endif


#include "osqp_api_types.h"


/*********************************************
*   Internal definition of OSQPVector types
*   and supporting definitions
*********************************************/

struct OSQPVectori_ {
  c_int *d_val;
  c_int  length;
};

struct OSQPVectorf_ {
  c_float *d_val;
  c_int    length;
};


/*********************************************
*   Internal definition of OSQPMatrix type
*   and supporting definitions
*********************************************/

/* Matrix in CSR format stored in GPU memory */
typedef struct csr_t csr;

struct OSQPMatrix_ {
  csr     *S;   /* P or A */
  csr     *At;
  c_int   *d_A_to_At_ind;
  c_float *d_P_triu_val;
  c_int   *d_P_triu_to_full_ind;
  c_int   *d_P_diag_ind;
  c_int    P_triu_nnz;
  c_int    symmetric;
};


# ifdef __cplusplus
}
# endif

#endif /* ifndef ALGEBRA_TYPES_H */
