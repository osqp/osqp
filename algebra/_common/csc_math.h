#ifndef CSC_MATH_H
# define CSC_MATH_H


# include "osqp_api_types.h"

/****************************************************************************
* CSC Matrix updates                                                        *
*****************************************************************************/

 /**
  * Update elements of a previously allocated csc matrix
  * without changing its sparsity structure.
  *
  *  If Mx_new_idx is OSQP_NULL, Mx_new is assumed to be as long as M->x
  *  and all matrix entries are replaced
  *
  *  The caller is responsible for ensuring that P_new_n is not greater
  *  the number of nonzeros in the matrix
  *
  * @param  M          csc matrix
  * @param  Mx_new     Vector of new elements in M->x
  * @param  Mx_new_idx Index mapping new elements to positions in M->x
  * @param  P_new_n    Number of new elements to be changed
  *
  */

void csc_update_values(OSQPCscMatrix* M,
                       const c_float* Mx_new,
                       const c_int*   Mx_new_idx,
                       c_int          P_new_n);

/*****************************************************************************
* CSC Algebraic Operations                                                   *
******************************************************************************/

// A = sc*A
void csc_scale(OSQPCscMatrix* A, c_float sc);

// A = diag(L)*A
void csc_lmult_diag(OSQPCscMatrix* A, const c_float* L);

// A = A*diag(R)
void csc_rmult_diag(OSQPCscMatrix* A, const c_float* R);

//y = alpha*A*x + beta*y, where A is symmetric and only triu is stored
void csc_Axpy_sym_triu(const OSQPCscMatrix* A,
                       const c_float*       x,
                             c_float*       y,
                             c_float        alpha,
                             c_float        beta);

//y = alpha*A*x + beta*y
void csc_Axpy(const OSQPCscMatrix* A,
              const c_float*       x,
                    c_float*       y,
                    c_float        alpha,
                    c_float        beta);

//y = alpha*A^T*x + beta*y
void csc_Atxpy(const OSQPCscMatrix* A,
               const c_float*       x,
                     c_float*       y,
                     c_float        alpha,
                     c_float        beta);

// // returns 1/2 x'*P*x
// c_float csc_quad_form(const csc *P, const c_float *x);

// E[i] = inf_norm(M(:,i))
void csc_col_norm_inf(const OSQPCscMatrix* M, c_float* E);

// E[i] = inf_norm(M(i,:))
void csc_row_norm_inf(const OSQPCscMatrix* M, c_float* E);

// E[i] = inf_norm(M(i,:)), where M stores triu part only
void csc_row_norm_inf_sym_triu(const OSQPCscMatrix* M, c_float* E);


#endif /* ifndef CSC_MATH_H */
