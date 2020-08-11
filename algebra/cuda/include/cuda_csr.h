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

#ifndef CUDA_CSR_H
# define CUDA_CSR_H

#include "algebra_types.h"
#include "csc_type.h"

# ifdef __cplusplus
extern "C" {
# endif /* ifdef __cplusplus */


void cuda_mat_init_P(const csc  *mat,
                     csr       **P,
                     c_float   **d_P_triu_val,
                     c_int     **d_P_triu_to_full_ind,
                     c_int     **d_P_diag_ind);
                     
void cuda_mat_init_A(const csc  *mat,
                     csr       **A,
                     csr       **At,
                     c_int     **d_A_to_At_ind);

void cuda_mat_update_P(const c_float  *Px,
                       const c_int    *Px_idx,
                       c_int           Px_n,
                       csr           **P,
                       c_float        *d_P_triu_val,
                       c_int          *d_P_triu_to_full_ind,
                       c_int          *d_P_diag_ind,
                       c_int           P_triu_nnz);

void cuda_mat_update_A(const c_float  *Ax,
                       const c_int    *Ax_idx,
                       c_int           Ax_n,
                       csr           **A,
                       csr           **At,
                       c_int          *d_A_to_At_ind);

void cuda_mat_free(csr *mat);

void cuda_submat_byrows(const csr    *A,
                        const c_int  *d_rows,
                        csr         **Ared,
                        csr         **Aredt);

# ifdef __cplusplus
}
# endif /* ifdef __cplusplus */

#endif /* ifndef CUDA_CSR_H */
