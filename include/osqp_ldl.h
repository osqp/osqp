#ifndef OSQP_LDL_H
# define OSQP_LDL_H

# ifdef __cplusplus
extern "C" {
# endif // ifdef __cplusplus

# include "types.h"
# include "glob_opts.h" //XXXXXX DEBUG JULIA.  REMOVE

/**
 * Create Compressed-Column-Sparse matrix from existing arrays
    (no MALLOC to create inner arrays x, i, p)
 * @param  m     First dimension
 * @param  n     Second dimension
 * @param  nzmax Maximum number of nonzero elements
 * @param  x     Vector of data
 * @param  i     Vector of row indices
 * @param  p     Vector of column pointers
 * @return       New matrix pointer
 */

/**
  * Compute the elimination tree for a quasidefinite matrix
  * in compressed sparse column form, where the input matrix is
  * assumed to contain data for the upper triangular part of A only,
  * and there are no duplicate indices.
  *
  * Returns an elimination tree for the factorization A = LDL^T and a
  * count of the nonzeros in each column of L that are strictly below the
  * diagonal.
  *
  * Does not use MALLOC.  It is assumed that the arrays work, Lnz, and
  * etree will be allocated with a number of elements equal to A->n.
  *
  * @param  A      CscMatrix (upper triangular part only)
  * @param  work   work vector (no meaning on return)
  * @param  Lnz    count of nonzeros in each column of L below diagonal
  * @param  etree  elimination tree
  * @return total  sum of Lnz (i.e. total nonzeros in L below diagonal)
  *
  *
*/

 c_int osqp_ldl_etree(const CscMatrix *A,
                c_int* work,
                c_int* Lnz,
                c_int* etree);






void osqp_ldl_factor(CscMatrix *A,
                     CscMatrix *L,
                     c_float* D,
                     c_float* Dinv,
                     c_int* Lnz,
                     c_int* etree,
                     c_int* iwork
                     c_float* fwork);


# ifdef __cplusplus
}
# endif // ifdef __cplusplus

#endif // ifndef OSQP_LDL_H
