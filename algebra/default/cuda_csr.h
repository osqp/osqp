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

void cuda_mat_free(csr *dev_mat);

void cuda_submat_byrows(csr         *submat,
                        csr         *submat_tpose,
                        const csr   *mat,
                        const c_int *h_rows);


# ifdef __cplusplus
}
# endif /* ifdef __cplusplus */

#endif /* ifndef CUDA_CSR_H */
