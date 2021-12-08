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

#include "algebra_types.h"
#include "lin_alg.h"

#include "csr_type.h"
#include "cuda_csr.h"
#include "cuda_lin_alg.h"
#include "cuda_malloc.h"


/*******************************************************************************
 *                           API Functions                                     *
 *******************************************************************************/

OSQPMatrix* OSQPMatrix_new_from_csc(const csc *M,
                                    c_int      is_triu) {

  OSQPMatrix* out = (OSQPMatrix *) c_calloc(1, sizeof(OSQPMatrix));
  if (!out) return OSQP_NULL;

  if (is_triu) {
    /* Initialize P */
    out->At = NULL;   /* indicates a symmetric matrix */
    out->P_triu_nnz = M->p[M->n];
    cuda_mat_init_P(M, &out->S, &out->d_P_triu_val, &out->d_P_triu_to_full_ind, &out->d_P_diag_ind);
  }
  else {
    /* Initialize A */
    cuda_mat_init_A(M, &out->S, &out->At, &out->d_A_to_At_ind);
  }

  return out;
}

void OSQPMatrix_update_values(OSQPMatrix    *mat,
                              const c_float *Mx_new,
                              const c_int   *Mx_new_idx,
                              c_int          Mx_new_n) {

  if (mat->At) { /* not symmetric */
    cuda_mat_update_A(Mx_new, Mx_new_idx, Mx_new_n, &mat->S, &mat->At, mat->d_A_to_At_ind);
  }
  else {
    cuda_mat_update_P(Mx_new, Mx_new_idx, Mx_new_n, &mat->S, mat->d_P_triu_val,
                      mat->d_P_triu_to_full_ind, mat->d_P_diag_ind, mat->P_triu_nnz);
  }
}

c_int OSQPMatrix_get_m( const OSQPMatrix *mat) { return mat->S->m; }

c_int OSQPMatrix_get_n( const OSQPMatrix *mat) { return mat->S->n; }

c_int OSQPMatrix_get_nz(const OSQPMatrix *mat) { return mat->At ? mat->S->nnz : mat->P_triu_nnz; }

void OSQPMatrix_mult_scalar(OSQPMatrix *mat,
                            c_float     sc) {

  cuda_mat_mult_sc(mat->S, mat->At, sc);
}

void OSQPMatrix_lmult_diag(OSQPMatrix        *mat,
                           const OSQPVectorf *D) {

  cuda_mat_lmult_diag(mat->S, mat->At, D->d_val);
}

void OSQPMatrix_rmult_diag(OSQPMatrix        *mat,
                           const OSQPVectorf *D) {

  cuda_mat_rmult_diag(mat->S, mat->At, D->d_val);
}

void OSQPMatrix_Axpy(const OSQPMatrix  *mat,
                     const OSQPVectorf *x,
                     OSQPVectorf       *y,
                     c_float            alpha,
                     c_float            beta) {

  cuda_mat_Axpy(mat->S, x->d_val, y->d_val, alpha, beta);
}

void OSQPMatrix_Atxpy(const OSQPMatrix  *mat,
                      const OSQPVectorf *x,
                      OSQPVectorf       *y,
                      c_float            alpha,
                      c_float            beta) {

  cuda_mat_Axpy(mat->At, x->d_val, y->d_val, alpha, beta);
}

void OSQPMatrix_col_norm_inf(const OSQPMatrix *mat,
                             OSQPVectorf      *res) {

  if (mat->At) cuda_mat_row_norm_inf(mat->At, res->d_val);
  else         cuda_mat_row_norm_inf(mat->S,  res->d_val);
}

void OSQPMatrix_row_norm_inf(const OSQPMatrix *mat,
                             OSQPVectorf      *res) {

  cuda_mat_row_norm_inf(mat->S, res->d_val);
}



void OSQPMatrix_free(OSQPMatrix *mat){
  if (mat) {
    cuda_mat_free(mat->S);
    cuda_mat_free(mat->At);
    cuda_free((void **) &mat->d_A_to_At_ind);
    cuda_free((void **) &mat->d_P_triu_to_full_ind);
    cuda_free((void **) &mat->d_P_diag_ind);
    cuda_free((void **) &mat->d_P_triu_val);
    c_free(mat);
  }
}

OSQPMatrix* OSQPMatrix_submatrix_byrows(const OSQPMatrix  *mat,
                                        const OSQPVectori *rows) {

  OSQPMatrix *out;

  if (!mat->At) {
#ifdef PRINTING
    c_eprint("row selection not implemented for partially filled matrices");
#endif
    return OSQP_NULL;
  }

  out = (OSQPMatrix *) c_calloc(1, sizeof(OSQPMatrix));
  if (!out) return OSQP_NULL;

  cuda_submat_byrows(mat->S, rows->d_val, &out->S, &out->At);

  return out;
}

