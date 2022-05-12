#ifndef ALGEBRA_VECTOR_H
#define ALGEBRA_VECTOR_H

#include "glob_opts.h"

# ifdef __cplusplus
extern "C" {
# endif

/*
 *  OSQPVector[fi] types.  Not defined here since it
 *  is implementation-specific
 */

/* integer valued vectors */
typedef struct OSQPVectori_ OSQPVectori;

/* float valued vectors*/
typedef struct OSQPVectorf_ OSQPVectorf;


/* VECTOR FUNCTIONS ----------------------------------------------------------*/

# ifndef EMBEDDED

/* malloc/calloc for floats and ints (USES MALLOC/CALLOC) */
OSQPVectorf* OSQPVectorf_malloc(c_int length);
OSQPVectorf* OSQPVectorf_calloc(c_int length);
OSQPVectori* OSQPVectori_malloc(c_int length);
OSQPVectori* OSQPVectori_calloc(c_int length);

/* Return a float vector using a raw array as input (Uses MALLOC) */
OSQPVectorf* OSQPVectorf_new(const c_float *a,
                             c_int          length);

/* Return a copy of a float vector a as output (Uses MALLOC) */
OSQPVectorf* OSQPVectorf_copy_new(const OSQPVectorf *a);

/* Free a float vector */
void OSQPVectorf_free(OSQPVectorf *a);

/* Free an int vector */
void OSQPVectori_free(OSQPVectori *a);

OSQPVectorf* OSQPVectorf_subvector_byrows(const OSQPVectorf  *A,
                                          const OSQPVectori *rows);

OSQPVectorf* OSQPVectorf_concat(const OSQPVectorf *A,
                                const OSQPVectorf *B);

/* Create subview of a larger vector.  Internal data should not be freed.
 * Behavior is otherwise identical to OSQPVectorf (Uses MALLOC)
 */
OSQPVectorf* OSQPVectorf_view(const OSQPVectorf *a,
                              c_int              head,
                              c_int              length);

/* Points existing subview somewhere else.  (Does not use MALLOC)
 * TODO: Get rid of this function
 */
void OSQPVectorf_view_update(OSQPVectorf *a, const OSQPVectorf *b, c_int head, c_int length);

/* Free a view of a float vector */
void OSQPVectorf_view_free(OSQPVectorf *a);

# endif /* ifndef EMBEDDED */


/* Length of the vector (floats) */
c_int OSQPVectorf_length(const OSQPVectorf *a);

/* Pointer to vector data (floats) */
c_float* OSQPVectorf_data(const OSQPVectorf *a);

/* Copy a float vector a into another vector b (pre-allocated) */
void OSQPVectorf_copy(OSQPVectorf       *b,
                      const OSQPVectorf *a);

/* Copy an array of floats into a into a vector b (pre-allocated) */
void OSQPVectorf_from_raw(OSQPVectorf   *b,
                          const c_float *a);

/* copy an array of ints into a into a vector b (pre-allocated) */
void OSQPVectori_from_raw(OSQPVectori *b,
                          const c_int *a);

/* copy a vector into an array of floats (pre-allocated) */
void OSQPVectorf_to_raw(c_float           *bv,
                        const OSQPVectorf *a);

/* copy a vector into an array of ints (pre-allocated) */
void OSQPVectori_to_raw(c_int             *bv,
                        const OSQPVectori *a);

/* set float vector to scalar */
void OSQPVectorf_set_scalar(OSQPVectorf *a,
                            c_float      sc);

/* Set float vector to one of three scalars based on sign of vector of ints */
void OSQPVectorf_set_scalar_conditional(OSQPVectorf       *a,
                                        const OSQPVectori *test,
                                        c_float            val_if_neg,
                                        c_float            val_if_zero,
                                        c_float            val_if_pos);

/* multiply float vector by float */
void OSQPVectorf_mult_scalar(OSQPVectorf *a,
                             c_float      sc);

/* x = a + b.  Set x == a for x += b. */
void OSQPVectorf_plus(OSQPVectorf       *x,
                      const OSQPVectorf *a,
                      const OSQPVectorf *b);

/* x = a - b.  x==a or x==-b at entry is possible */
void OSQPVectorf_minus(OSQPVectorf      *x,
                      const OSQPVectorf *a,
                      const OSQPVectorf *b);

/* x = sca*a + scb*b.  Set (x == a, sca==1.) for x += scb*b. */
void OSQPVectorf_add_scaled(OSQPVectorf       *x,
                            c_float            sca,
                            const OSQPVectorf *a,
                            c_float            scb,
                            const OSQPVectorf *b);

/* x = sca*a + scb*b + scc*c.  Set (x == a, sca==1.) for x += scb*b scc*c. */
void OSQPVectorf_add_scaled3(OSQPVectorf       *x,
                             c_float            sca,
                             const OSQPVectorf *a,
                             c_float            scb,
                             const OSQPVectorf *b,
                             c_float            scc,
                             const OSQPVectorf *c);

/* ||v||_inf */
c_float OSQPVectorf_norm_inf(const OSQPVectorf *v);

/* ||Sv||_inf */
c_float OSQPVectorf_scaled_norm_inf(const OSQPVectorf *S,
                                    const OSQPVectorf *v);

/* ||a - b||_inf */
c_float OSQPVectorf_norm_inf_diff(const OSQPVectorf *a,
                                  const OSQPVectorf *b);

/* mean of vector elements */
c_float OSQPVectorf_mean(const OSQPVectorf *a);

/* Inner product a'b */
c_float OSQPVectorf_dot_prod(const OSQPVectorf *a,
                             const OSQPVectorf *b);

/* Inner product a'b, but using only the positive or negative
 * terms in b.  Use sign = 1 for positive terms, sign = -1 for
 * negative terms.  Setting any other value for sign will return
 * the normal dot product
 */
c_float OSQPVectorf_dot_prod_signed(const OSQPVectorf *a,
                                    const OSQPVectorf *b,
                                    c_int              sign);

/* Elementwise product a.*b stored in c.  Set c==a for c *= b */
void OSQPVectorf_ew_prod(OSQPVectorf       *c,
                         const OSQPVectorf *a,
                         const OSQPVectorf *b);

/* check l <= u elementwise.  Returns 1 if inequality is true
 * for every element pair in both vectors
 */
c_int OSQPVectorf_all_leq(const OSQPVectorf *l,
                          const OSQPVectorf* u);

/* Elementwise bounding vectors x = min(max(z,l),u)
 * It is acceptable to assign x = z in this call, so
 * that x = min(max(x,l),u) is allowed
 */
void OSQPVectorf_ew_bound_vec(OSQPVectorf       *x,
                              const OSQPVectorf *z,
                              const OSQPVectorf *l,
                              const OSQPVectorf *u);


/* Elementwise projection of y onto the polar recession cone
   of the set [l u].  Values of +/- infval or larger are
   treated as infinite
 */
void OSQPVectorf_project_polar_reccone(OSQPVectorf       *y,
                                       const OSQPVectorf *l,
                                       const OSQPVectorf *u,
                                       c_float            infval);

/* Elementwise test of whether y is in the polar recession
   cone of the set [l u].  Values of +/- infval or larger are
   treated as infinite.  Values in y within tol of zero are treated
   as zero.
 */
c_int OSQPVectorf_in_reccone(const OSQPVectorf *y,
                             const OSQPVectorf *l,
                             const OSQPVectorf *u,
                             c_float            infval,
                             c_float            tol);

# if EMBEDDED != 1

/* Vector mean value*/
c_float OSQPVectorf_mean(const OSQPVectorf *a);

/* Vector elementwise reciprocal b = 1./a (needed for scaling)*/
void OSQPVectorf_ew_reciprocal(OSQPVectorf       *b,
                               const OSQPVectorf *a);

/* elementwise sqrt of the vector elements */
void OSQPVectorf_ew_sqrt(OSQPVectorf *a);

/* Elementwise maximum between vectors c = max(a, b) */
void OSQPVectorf_ew_max_vec(OSQPVectorf       *c,
                            const OSQPVectorf *a,
                            const OSQPVectorf *b);

/* Elementwise minimum between vectors c = min(a, b) */
void OSQPVectorf_ew_min_vec(OSQPVectorf       *c,
                            const OSQPVectorf *a,
                            const OSQPVectorf *b);

/* Elementwise check for constraint type.
   if u[i] - l[i] < tol, iseq[i] = 1 otherwise iseq[i] = 0,
   unless values exceed +/- infval, in which case marked
   as iseq[i] = -1.

   Returns 1 if any value in iseq has been modified.   O otherwise.
 */
c_int OSQPVectorf_ew_bounds_type(OSQPVectori      *iseq,
                                const OSQPVectorf *l,
                                const OSQPVectorf *u,
                                c_float            tol,
                                c_float            infval);


/* Elementwise replacement based on lt comparison.
   x[i] = z[i] < testval ? newval : z[i];
*/
void OSQPVectorf_set_scalar_if_lt(OSQPVectorf       *x,
                                  const OSQPVectorf *z,
                                  c_float            testval,
                                  c_float            newval);

/* Elementwise replacement based on gt comparison.
 * x[i] = z[i] > testval ? newval : z[i];
 */
void OSQPVectorf_set_scalar_if_gt(OSQPVectorf       *x,
                                  const OSQPVectorf *z,
                                  c_float            testval,
                                  c_float            newval);

# endif /* if EMBEDDED != 1 */


# ifdef __cplusplus
}
# endif

#endif /* ifndef ALGEBRA_VECTOR_H */
