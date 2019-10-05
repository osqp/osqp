#ifndef CUDA_CSR_H
# define CUDA_CSR_H

#include "algebra_types.h"

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

void cuda_mat_get_m(const csr *mat,
                    c_int     *m);

void cuda_mat_get_n(const csr *mat,
                    c_int     *n);

void cuda_mat_get_nnz(const csr *mat,
                      c_int     *nnz);

# ifdef __cplusplus
}
# endif /* ifdef __cplusplus */

#endif /* ifndef CUDA_CSR_H */
