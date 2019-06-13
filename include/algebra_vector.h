#ifndef ALGEBRA_VECTOR_H
# define ALGEBRA_VECTOR_H


# ifdef __cplusplus
extern "C" {
# endif // ifdef __cplusplus

# include "types.h"

/* VECTOR FUNCTIONS ----------------------------------------------------------*/

# ifndef EMBEDDED

/* copy vector a into output (Uses MALLOC)*/
c_float* vec_copy(c_float *a,
                  c_int    n);
# endif // ifndef EMBEDDED

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

/* add scalar to vector*/
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
# endif // if EMBEDDED != 1

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


# endif // if EMBEDDED != 1

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

 /* Return a copy of a float vector a as output (Uses MALLOC)*/
 OSQPVectorf* OSQPVectorf_copy_new(OSQPVectorf *a);

 /* Return a copy of an int vector a as output (Uses MALLOC)*/
 OSQPVectori* OSQPVectori_copy_new(OSQPVectori *a);

 /* Free a float vector*/
 OSQPVectorf* OSQPVectorf_free(OSQPVectorf *a);

 /* Free an int vector*/
 OSQPVectori* OSQPVectori_free(OSQPVectori *a);

 # endif // ifndef EMBEDDED

 /* Length of the vector (floats) */
 c_int OSQPVectorf_length(OSQPVectorf *a);

 /* Length of the vector (ints)*/
 c_int OSQPVectori_length(OSQPVectori *a);

 /* Pointer to vector data (floats) */
 c_float* OSQPVectorf_data(OSQPVectorf *a);

 /* Pointer to vector data (ints) */
 c_int* OSQPVectori_data(OSQPVectori *a);

 /* copy a float vector a into another vector b (pre-allocated) */
 void OSQPVectorf_copy(OSQPVectorf *a,OSQPVectorf *b);

 /* copy an int vector a into another vector b (pre-allocated) */
 void OSQPVectori_copy(OSQPVectori *a,OSQPVectori *b);

 /* set float vector to scalar */
 void OSQPVectorf_set_scalar(OSQPVectorf *a, c_float sc);

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


 /* c = a + sc*b */
 void OSQPVectorf_add_scaled(OSQPVectorf       *c,
                              const OSQPVectorf *a,
                              const OSQPVectorf *b,
                              c_float           sc);


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

 /* elementwise product a.*b stored in c*/
 void OSQPVectorf_ew_prod(const OSQPVectorf *a,
                          const OSQPVectorf *b,
                          OSQPVectorf       *c);

 /* vector permutation x[:] = b(p[:]) */
 void OSQPVectorf_permute(OSQPVectorf *x, OSQPVectorf *b, OSQPVectori *p);
 void OSQPVectori_permute(OSQPVectori *x, OSQPVectori *b, OSQPVectori *p);

 /* vector inverse permutation x(p[:]) = b */
 void OSQPVectorf_ipermute(OSQPVectorf *x, OSQPVectorf *b, OSQPVectori *p);
 void OSQPVectori_ipermute(OSQPVectori *x, OSQPVectori *b, OSQPVectori *p);


 # if EMBEDDED != 1

 /* Vector elementwise reciprocal b = 1./a (needed for scaling)*/
 void OSQPVectorf_ew_reciprocal(const OSQPVectorf *a, OSQPVectorf *b);

 /* elementwise sqrt of the vector elements */
 void OSQPVectorf_ew_sqrt(OSQPVectorf *a);

 /* elementwise max between each vector component and max_val */
 void OSQPVectorf_ew_max(OSQPVectorf *a, c_float max_val);

 /* elementwise max between each vector component and max_val */
 void OSQPVectorf_ew_min(OSQPVectorf *a, c_float min_val);

 /* Elementwise maximum between vectors c = max(a, b) */
 void OSQPVectorf_ew_max_vec(const OSQPVectorf *a,
                             const OSQPVectorf *b,
                             OSQPVectorf       *c);

 /* Elementwise minimum between vectors c = min(a, b) */
 void OSQPVectorf_ew_min_vec(const OSQPVectorf *a,
                             const OSQPVectorf *b,
                             OSQPVectorf       *c);

 /* Elementwise bounding vectors x = min(max(z,l),u)
  * It is acceptable to assign x = z in this call, so
  * that x = min(max(x,l),u) is allowed
  */
 void OSQPVectorf_ew_bound_vec(const OSQPVectorf *l,
                               const OSQPVectorf *u,
                                     OSQPVectorf *z,
                                     OSQPVectorf *x);

 # endif // if EMBEDDED != 1


# ifdef __cplusplus
}
# endif // ifdef __cplusplus

#endif // ifndef ALGEBRA_VECTOR_H
