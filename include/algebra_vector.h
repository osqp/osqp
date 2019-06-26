#ifndef ALGEBRA_VECTOR_H
# define ALGEBRA_VECTOR_H


# ifdef __cplusplus
extern "C" {
# endif /* ifdef __cplusplus */

# include "types.h"

/* VECTOR FUNCTIONS ---------------------------------------------------------- */

# ifndef EMBEDDED

/* copy vector a into output (Uses MALLOC) */
c_float* vec_copy(c_float *a,
                  c_int    n);
# endif /* ifndef EMBEDDED */

/* copy vector a into preallocated vector b */
void prea_vec_copy(const c_float *a,
                   c_float       *b,
                   c_int          n);

/* copy integer vector a into preallocated vector b */
void prea_int_vec_copy(const c_int *a,
                       c_int       *b,
                       c_int        n);

/* set float vector to scalar */
void vec_set_scalar(c_float *a,
                    c_float  sc,
                    c_int    n);

/* set integer vector to scalar */
void int_vec_set_scalar(c_int *a,
                        c_int  sc,
                        c_int  n);

/* add scalar to vector */
void vec_add_scalar(c_float *a,
                    c_float  sc,
                    c_int    n);

/* multiply scalar to vector */
void vec_mult_scalar(c_float *a,
                     c_float  sc,
                     c_int    n);

/* c = a + sc*b */
void vec_add_scaled(c_float       *c,
                    const c_float *a,
                    const c_float *b,
                    c_int          n,
                    c_float        sc);

/* ||v||_inf */
c_float vec_norm_inf(const c_float *v,
                     c_int          l);

/* ||Sv||_inf */
c_float vec_scaled_norm_inf(const c_float *S,
                            const c_float *v,
                            c_int          l);

/* ||a - b||_inf */
c_float vec_norm_inf_diff(const c_float *a,
                          const c_float *b,
                          c_int          l);

/* mean of vector elements */
c_float vec_mean(const c_float *a,
                 c_int          n);

# if EMBEDDED != 1

/* Vector elementwise reciprocal b = 1./a (needed for scaling)*/
void vec_ew_recipr(const c_float *a,
                   c_float       *b,
                   c_int          n);
# endif /* if EMBEDDED != 1 */

/* Inner product a'b */
c_float vec_prod(const c_float *a,
                 const c_float *b,
                 c_int          n);

/* Elementwise product a.*b stored in c*/
void vec_ew_prod(const c_float *a,
                 const c_float *b,
                 c_float       *c,
                 c_int          n);

# if EMBEDDED != 1

/* Elementwise sqrt of the vector elements */
void vec_ew_sqrt(c_float *a,
                 c_int    n);

/* Elementwise max between each vector component and max_val */
void vec_ew_max(c_float *a,
                c_int    n,
                c_float  max_val);

/* Elementwise min between each vector component and max_val */
void vec_ew_min(c_float *a,
                c_int    n,
                c_float  min_val);

/* Elementwise maximum between vectors c = max(a, b) */
void vec_ew_max_vec(const c_float *a,
                    const c_float *b,
                    c_float       *c,
                    c_int          n);

/* Elementwise minimum between vectors c = min(a, b) */
void vec_ew_min_vec(const c_float *a,
                    const c_float *b,
                    c_float       *c,
                    c_int          n);


# endif /* if EMBEDDED != 1 */

/* VECTOR FUNCTIONS ----------------------------------------------------------*/

typedef struct OSQPVectori_ {
  c_int* values;
  c_int length;
} OSQPVectori;

typedef struct OSQPVectorf_ {
  c_float* values;
  c_int length;
} OSQPVectorf;


# ifndef EMBEDDED

/* malloc/calloc for floats and ints (USES MALLOC/CALLOC) */
OSQPVectorf* OSQPVectorf_malloc(c_int length);
OSQPVectorf* OSQPVectorf_calloc(c_int length);
OSQPVectori* OSQPVectori_malloc(c_int length);
OSQPVectori* OSQPVectori_calloc(c_int length);

/* Return a float vector using a raw array as input (Uses MALLOC) */
OSQPVectorf* OSQPVectorf_new(const c_float *a, c_int length);

/* Return an in vector using a raw array as input (Uses MALLOC) */
OSQPVectori* OSQPVectori_new(const c_int *a, c_int length);

/* Return a copy of a float vector a as output (Uses MALLOC) */
OSQPVectorf* OSQPVectorf_copy_new(const OSQPVectorf *a);

/* Return a copy of an int vector a as output (Uses MALLOC) */
OSQPVectori* OSQPVectori_copy_new(const OSQPVectori *a);

/* Free a float vector */
void OSQPVectorf_free(OSQPVectorf *a);

/* Free an int vector */
void OSQPVectori_free(OSQPVectori *a);

/* Create subview of a larger vector.  Internal data should not be freed.
 * Behavior is otherwise identical to OSQPVectorf (Uses MALLOC)
 */
OSQPVectorf* OSQPVectorf_view(const OSQPVectorf *a, c_int head, c_int length);

/* Free a view of a float vector */
void OSQPVectorf_view_free(OSQPVectorf *a);

# endif /* ifndef EMBEDDED */


/* Length of the vector (floats) */
c_int OSQPVectorf_length(const OSQPVectorf *a);

/* Length of the vector (ints) */
c_int OSQPVectori_length(const OSQPVectori *a);

/* Pointer to vector data (floats) */
c_float* OSQPVectorf_data(const OSQPVectorf *a);

/* Pointer to vector data (ints) */
c_int* OSQPVectori_data(const OSQPVectori *a);

/* Copy a float vector a into another vector b (pre-allocated) */
void OSQPVectorf_copy(OSQPVectorf *b, const OSQPVectorf *a);

/* Copy an int vector a into another vector b (pre-allocated) */
void OSQPVectori_copy(OSQPVectori *b, const OSQPVectori *a);

/* Copy an array of floats into a into a vector b (pre-allocated) */
void OSQPVectorf_from_raw(OSQPVectorf *b, const c_float *a);

/* copy an array of ints into a into a vector b (pre-allocated) */
void OSQPVectori_from_raw(OSQPVectori *b, const c_int *a);

/* copy a vector into an array of floats (pre-allocated) */
void OSQPVectorf_to_raw(c_float *bv, const OSQPVectorf *a);

/* copy a vector into an array of ints (pre-allocated) */
void OSQPVectori_to_raw(c_int *bv, const OSQPVectori *a);

/* set float vector to scalar */
void OSQPVectorf_set_scalar(OSQPVectorf *a, c_float sc);

/* Set float vector to one of three scalars based on sign of vector of ints.
 */
void OSQPVectorf_set_scalar_conditional(OSQPVectorf *a,
                                        OSQPVectori *test,
                                        c_float val_if_neg,
                                        c_float val_if_zero,
                                        c_float val_if_pos);

/* set int vector to scalar */
void OSQPVectori_set_scalar(OSQPVectori *a, c_int sc);

/* add scalar to float vector */
void OSQPVectorf_add_scalar(OSQPVectorf *a, c_float sc);

/* add scalar to int vector */
void OSQPVectori_add_scalar(OSQPVectori *a, c_int sc);

/* multiply float vector by float */
void OSQPVectorf_mult_scalar(OSQPVectorf *a, c_float sc);

/* change sign */
void OSQPVectorf_negate(OSQPVectorf *a);

/* x = a + b.  Set x == a for x += b.  */
void OSQPVectorf_plus(OSQPVectorf      *x,
                      const OSQPVectorf *a,
                      const OSQPVectorf *b);

/* x = a - b.  Set x == a for x -= b. */
void OSQPVectorf_minus(OSQPVectorf      *x,
                      const OSQPVectorf *a,
                      const OSQPVectorf *b);

/* x = sca*a + scb*b.  Set (x == a, sca==1.) for x += scb*b. */
void OSQPVectorf_add_scaled(OSQPVectorf      *x,
                            c_float           sca,
                            const OSQPVectorf *a,
                            c_float           scb,
                            const OSQPVectorf *b);

/* x = sca*a + scb*b + scc*c.   Set (x == a, sca==1.) for x += scb*b scc*c. */
void OSQPVectorf_add_scaled3(OSQPVectorf       *x,
                             c_float           sca,
                             const OSQPVectorf *a,
                             c_float           scb,
                             const OSQPVectorf *b,
                             c_float           scc,
                             const OSQPVectorf *c);

/* ||v||_inf */
c_float OSQPVectorf_norm_inf(const OSQPVectorf *v);

/* ||v||_1 */
c_float OSQPVectorf_norm_1(const OSQPVectorf *v);

/* ||Sv||_inf */
c_float OSQPVectorf_scaled_norm_inf(const OSQPVectorf *S, const OSQPVectorf *v);

/* ||Sv||_1 */
c_float OSQPVectorf_scaled_norm_1(const OSQPVectorf *S, const OSQPVectorf *v);

/* ||a - b||_inf */
c_float OSQPVectorf_norm_inf_diff(const OSQPVectorf *a,
                                  const OSQPVectorf *b);


/* sum of vector elements */
c_float OSQPVectorf_sum(const OSQPVectorf *a);

/* mean of vector elements */
c_float OSQPVectorf_mean(const OSQPVectorf *a);



/* Inner product a'b */
c_float OSQPVectorf_dot_prod(const OSQPVectorf *a,
                            const OSQPVectorf *b);

/* Inner product a'b, but using only the positive or Negative
 * terms in b.   Use sign = 1 for positive terms, sign = -1 for
 * negative terms.   Setting any other value for sign will return
 * the normal dot product
 */
c_float OSQPVectorf_dot_prod_signed(const OSQPVectorf *a,
                                    const OSQPVectorf *b,
                                    c_int sign);

/* elementwise product a.*b stored in c.  Set c==a for c *= b*/
void OSQPVectorf_ew_prod(OSQPVectorf       *c,
                         const OSQPVectorf *a,
                         const OSQPVectorf *b);

/* check l <= u elementwise */
c_int OSQPVectorf_all_leq(OSQPVectorf *l, OSQPVectorf* u);

/* Elementwise bounding vectors x = min(max(z,l),u)
 * It is acceptable to assign x = z in this call, so
 * that x = min(max(x,l),u) is allowed
 */
void OSQPVectorf_ew_bound_vec(OSQPVectorf *x,
                               OSQPVectorf *z,
                               const OSQPVectorf *l,
                               const OSQPVectorf *u);


/* Elementwise projection of y onto the polar recession cone
   of the set [l u].  Values of +/- infval or larger are
   treated as infinite
 */
void OSQPVectorf_project_polar_reccone(OSQPVectorf       *y,
                                       const OSQPVectorf *l,
                                       const OSQPVectorf *u,
                                       c_float        infval);

/* Elementwise test of whether y is in the polar recession
   cone of the set [l u].  Values of +/- infval or larger are
   treated as infinite.  Values in y within tol of zero are treated
   as zero.
 */
c_int OSQPVectorf_in_polar_reccone(const OSQPVectorf *y,
                                   const OSQPVectorf *l,
                                   const OSQPVectorf *u,
                                   c_float           infval,
                                   c_float           tol);

/* vector permutation x[:] = b(p[:]) */
void OSQPVectorf_permute(OSQPVectorf *x, const OSQPVectorf *b, const OSQPVectori *p);
void OSQPVectori_permute(OSQPVectori *x, const OSQPVectori *b, const OSQPVectori *p);

/* vector inverse permutation x(p[:]) = b */
void OSQPVectorf_ipermute(OSQPVectorf *x, const OSQPVectorf *b, const OSQPVectori *p);
void OSQPVectori_ipermute(OSQPVectori *x, const OSQPVectori *b, const OSQPVectori *p);


# if EMBEDDED != 1

/* Vector mean value*/
c_float OSQPVectorf_mean(const OSQPVectorf *a);

/* Vector elementwise reciprocal b = 1./a (needed for scaling)*/
void OSQPVectorf_ew_reciprocal( OSQPVectorf *b, const OSQPVectorf *a);

/* elementwise sqrt of the vector elements */
void OSQPVectorf_ew_sqrt(OSQPVectorf *a);

/* Elementwise maximum between vector and scalar c = max(a, sc) */
void OSQPVectorf_ew_max(OSQPVectorf *c, const OSQPVectorf *a, c_float sc);

/* Elementwise minimum between vector and scalar c = min(a, sc) */
void OSQPVectorf_ew_min(OSQPVectorf *c, const OSQPVectorf *a, c_float sc);

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
c_int OSQPVectorf_ew_bounds_type(OSQPVectori* iseq,
                                const OSQPVectorf* l,
                                const OSQPVectorf* u,
                                c_float tol,
                                c_float infval);


/* Elementwise replacement based on lt comparison.
   x[i] = z[i] < testval ? newval : z[i];
*/
void OSQPVectorf_set_scalar_if_lt(OSQPVectorf *x,
                                  const OSQPVectorf *z,
                                  c_float testval,
                                  c_float newval);

/* Elementwise replacement based on gt comparison.
 * x[i] = z[i] > testval ? newval : z[i];
 */
void OSQPVectorf_set_scalar_if_gt(OSQPVectorf *x,
                                  const OSQPVectorf *z,
                                  c_float testval,
                                  c_float newval);

# endif /* if EMBEDDED != 1 */


# ifdef __cplusplus
}
# endif /* ifdef __cplusplus */

#endif /* ifndef ALGEBRA_VECTOR_H */
