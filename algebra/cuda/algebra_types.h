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

#ifndef ALGEBRA_TYPES_H
# define ALGEBRA_TYPES_H


#include <cusparse.h>
#include "osqp_api_types.h"


/*********************************************
*   Internal definition of OSQPVector types
*   and supporting definitions
*********************************************/

struct OSQPVectori_ {
  OSQPInt* d_val;
  OSQPInt  length;
};

struct OSQPVectorf_ {
  OSQPFloat*           d_val;
  OSQPInt              length;
  cusparseDnVecDescr_t vec;
};


/*********************************************
*   Internal definition of OSQPMatrix type
*   and supporting definitions
*********************************************/

/* Matrix in CSR format stored in GPU memory */
typedef struct csr_t csr;

struct OSQPMatrix_ {
  csr*       S;   /* P or A */
  csr*       At;  /* NULL if symmetric */
  OSQPInt*   d_A_to_At_ind;
  OSQPFloat* d_P_triu_val;
  OSQPInt*   d_P_triu_to_full_ind;
  OSQPInt*   d_P_diag_ind;
  OSQPInt    P_triu_nnz;
};


#endif /* ifndef ALGEBRA_TYPES_H */
