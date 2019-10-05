#include "cuda_pcg.h"
#include "csr_type.h"
#include "cuda_handler.h"
#include "cuda_malloc.h"
#include "cuda_lin_alg.h"
#include "cuda_wrapper.h"
#include "helper_cuda.h"    /* --> checkCudaErrors */

extern CUDA_Handle_t *CUDA_handle;


void cuda_pcg_update_precond(cudapcg_solver *s,
                             c_int           P_updated,
                             c_int           A_updated,
                             c_int           R_updated) {

  void    *buffer;
  c_float *tmp;
  c_int    n  = s->n;
  csr     *At = s->At;

  size_t Buffer_size_in_bytes = n * (sizeof(c_float) + sizeof(c_int));

  if (!P_updated && !A_updated && !R_updated) return;

  if (R_updated) cuda_vec_copy_h2d(s->d_rho, s->h_rho, 1);

  if (P_updated) {
    /* Update d_P_diag_val */
    checkCudaErrors(cusparseTgthr(CUDA_handle->cusparseHandle, n, s->P->val, s->d_P_diag_val, s->d_P_diag_ind, CUSPARSE_INDEX_BASE_ZERO));
  }

  if (A_updated || R_updated) {
    /* Allocate memory */
    cuda_malloc((void **) &tmp, At->nnz * sizeof(c_float));
    cuda_malloc((void **) &buffer, Buffer_size_in_bytes);

    /* Update d_AtRA_diag_val */
    if (!s->d_rho_vec) {  /* R = rho*I  -->  A'*R*A = rho * A'*A */

      if (A_updated) {
        /* Update d_AtA_diag_val */
        cuda_vec_ew_prod(tmp, At->val, At->val, At->nnz);
        cuda_vec_segmented_sum(tmp, At->row_ind, s->d_AtA_diag_val, buffer, n, At->nnz);
      }

      /* d_AtRA_diag_val = rho * d_AtA_diag_val */
      cuda_vec_add_scaled(s->d_AtRA_diag_val, s->d_AtA_diag_val, NULL, *s->h_rho, 0.0, n);
    }
    else {    /* R = diag(d_rho_vec)  -->  A'*R*A = A' * diag(d_rho_vec) * A */
      cuda_mat_rmult_diag_new(At, tmp, s->d_rho_vec);   /* tmp = A' * R */
      cuda_vec_ew_prod(tmp, tmp, At->val, At->nnz);     /* tmp = tmp * A */
      cuda_vec_segmented_sum(tmp, At->row_ind, s->d_AtRA_diag_val, buffer, n, At->nnz);
    }

    /* Free memory */
    cuda_free((void **) &tmp);
    cuda_free((void **) &buffer);
  }

  /* d_diag_precond = sigma */
  cuda_vec_set_sc(s->d_diag_precond, *s->h_sigma, n);

  /* d_diag_precond += d_P_diag_val + d_AtRA_diag_val */
  cuda_vec_add_scaled3(s->d_diag_precond, s->d_diag_precond, s->d_P_diag_val, s->d_AtRA_diag_val, 1.0, 1.0, 1.0, n);

  /* d_diag_precond_inv = 1 / d_diag_precond */
  cuda_vec_reciprocal(s->d_diag_precond_inv, s->d_diag_precond, n);
}
