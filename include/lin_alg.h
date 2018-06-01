#ifndef LIN_ALG_H
# define LIN_ALG_H


# ifdef __cplusplus
extern "C" {
# endif // ifdef __cplusplus

# include "types.h"


/* VECTOR FUNCTIONS ----------------------------------------------------------*/

# ifndef EMBEDDED

/* Return a copy of a float vector a as output (Uses MALLOC)*/
OsqpVectorf* OsqpVectorf_copy_new(OsqpVectorf *a);

/* Return a copy of an int vector a as output (Uses MALLOC)*/
OsqpVectori* OsqpVectori_copy_new(OsqpVectori *a);

# endif // ifndef EMBEDDED

/* copy a float vector a into another vector b (pre-allocated) */
void OsqpVectorf_copy(OsqpVectorf *a,OsqpVectorf *b);

/* copy an int vector a into another vector b (pre-allocated) */
void OsqpVectori_copy(OsqpVectori *a,OsqpVectori *b);

/* set float vector to scalar */
void OsqpVectorf_set_scalar(OsqpVectorf *a, c_float sc);

/* set int vector to scalar */
void OsqpVectori_set_scalar(OsqpVectori *a, c_int sc);

/* add scalar to float vector */
void OsqpVectorf_add_scalar(OsqpVectorf *a, c_float sc);

/* add scalar to int vector */
void OsqpVectori_add_scalar(OsqpVectori *a, c_int sc);

/* multiply float vector by float */
void OsqpVectorf_mult_scalar(OsqpVectorf *a, c_float sc);

/* change sign */
void OsqpVectorf_negate(OsqpVectorf *a);


/* c = a + sc*b */
void OsqpVectorf_add_scaled(OsqpVectorf       *c,
                             const OsqpVectorf *a,
                             const OsqpVectorf *b,
                             c_float           sc);


/* ||v||_inf */
c_float OsqpVectorf_norm_inf(const OsqpVectorf *v);

/* ||v||_1 */
c_float OsqpVectorf_norm_1(const OsqpVectorf *v);

/* ||Sv||_inf */
c_float OsqpVectorf_scaled_norm_inf(const OsqpVectorf *S, const OsqpVectorf *v);

/* ||Sv||_1 */
c_float OsqpVectorf_scaled_norm_1(const OsqpVectorf *S, const OsqpVectorf *v);

/* ||a - b||_inf */
c_float OsqpVectorf_norm_inf_diff(const OsqpVectorf *a,
                                  const OsqpVectorf *b);


/* sum of vector elements */
c_float OsqpVectorf_sum(const OsqpVectorf *a);

/* mean of vector elements */
c_float OsqpVectorf_mean(const OsqpVectorf *a);



/* Inner product a'b */
c_float OsqpVectorf_dot_prod(const OsqpVectorf *a,
                             const OsqpVectorf *b);

/* elementwise product a.*b stored in c*/
void OsqpVectorf_ew_prod(const OsqpVectorf *a,
                         const OsqpVectorf *b,
                         OsqpVectorf       *c);


# if EMBEDDED != 1

/* Vector elementwise reciprocal b = 1./a (needed for scaling)*/
void OsqpVectorf_ew_reciprocal(const OsqpVectorf *a, OsqpVectorf *b);

/* elementwise sqrt of the vector elements */
void OsqpVectorf_ew_sqrt(OsqpVectorf *a);

/* elementwise max between each vector component and max_val */
void OsqpVectorf_ew_max(OsqpVectorf *a, c_float max_val);

/* elementwise max between each vector component and max_val */
void OsqpVectorf_ew_min(OsqpVectorf *a, c_float min_val);

/* Elementwise maximum between vectors c = max(a, b) */
void OsqpVectorf_ew_max_vec(const OsqpVectorf *a,
                            const OsqpVectorf *b,
                            OsqpVectorf       *c);

/* Elementwise minimum between vectors c = min(a, b) */
void OsqpVectorf_ew_min_vec(const OsqpVectorf *a,
                            const OsqpVectorf *b,
                            OsqpVectorf       *c);

# endif // if EMBEDDED != 1


/* MATRIX FUNCTIONS ----------------------------------------------------------*/

/* multiply matrix by scalar */
void OsqpMatrix_mult_scalar(OsqpMatrix *A, c_float sc);
void CscMatrix_mult_scalar(CscMatrix *A, c_float sc);
void CsrMatrix_mult_scalar(CsrMatrix *A, c_float sc);

/* Premultiply (i.e. left) matrix A by diagonal matrix with diagonal d,
   i.e. scale the rows of A by d
 */
void OsqpMatrix_premult_diag(OsqpMatrix *A, const OsqpVectorf *d);
void CscMatrix_premult_diag(CscMatrix *A, const OsqpVectorf *d);
void CsrMatrix_premult_diag(CsrMatrix *A, const OsqpVectorf *d);

/* Postmultiply (i.e. right) matrix A by diagonal matrix with diagonal d,
   i.e. scale the columns of A by d
 */
void OsqpMatrix_postmult_diag(OsqpMatrix *A, const OsqpVectorf *d);
void CscMatrix_postmult_diag(CscMatrix *A, const OsqpVectorf *d);
void CsrMatrix_postmult_diag(CsrMatrix *A, const OsqpVectorf *d);


/* Matrix-vector multiplication
 *    y  =  A*x  (if plus_eq == 0)
 *    y +=  A*x  (if plus_eq == 1)
 *    y -=  A*x  (if plus_eq == -1)
 */
void OsqpMatrix_Ax(const OsqpMatrix  *A,
                   const OsqpVectorf *x,
                   OsqpVectorf       *y,
                   c_int             sign);
void CscMatrix_Ax(const CscMatrix  *A,
                  const OsqpVectorf *x,
                  OsqpVectorf       *y,
                  c_int             sign,
                  c_int             skip_diag);
void CsrMatrix_Ax(const CsrMatrix  *A,
                  const OsqpVectorf *x,
                  OsqpVectorf       *y,
                  c_int             sign,
                  c_int             skip_diag);


/* Matrix-transpose-vector multiplication
 *    y  =  A'*x  (if plus_eq == 0)
 *    y +=  A'*x  (if plus_eq == 1)
 *    y -=  A'*x  (if plus_eq == -1)
 */
 void OsqpMatrix_Atx(const OsqpMatrix  *A,
                     const OsqpVectorf *x,
                     OsqpVectorf       *y,
                     c_int             sign);
 void CscMatrix_Atx(const CscMatrix  *A,
                    const OsqpVectorf *x,
                    OsqpVectorf       *y,
                    c_int             sign,
                    c_int             skip_diag);
 void CsrMatrix_Atx(const CsrMatrix  *A,
                    const OsqpVectorf *x,
                    OsqpVectorf       *y,
                    c_int             sign,
                    c_int             skip_diag);


# if EMBEDDED != 1

/**
 * Infinity norm of each matrix column
 * @param M	Input matrix
 * @param E     Vector of infinity norms
 *
 */
void OsqpMatrix_inf_norm_cols(const OsqpMatrix *M,OsqpVectorf *E);
void CscMatrix_inf_norm_cols(const CscMatrix *M,OsqpVectorf *E);
void CsrMatrix_inf_norm_cols(const CsrMatrix *M,OsqpVectorf *E);

/**
 * Infinity norm of each matrix row
 * @param M	Input matrix
 * @param E     Vector of infinity norms
 *
 */
void OsqpMatrix_inf_norm_rows(const OsqpMatrix *M,OsqpVectorf *E);
void CscMatrix_inf_norm_rows(const CscMatrix *M,OsqpVectorf *E);
void CsrMatrix_inf_norm_rows(const CsrMatrix *M,OsqpVectorf *E);

# endif // EMBEDDED != 1

/**
 * Compute quadratic form f(x) = 1/2 x' P x
 * @param  P symmetrix matrix (symmetry is TRIU or TRIL)
 * @param  x argument float vector
 * @return   quadratic form value
 */
c_float OsqpMatrix_quad_form(const OsqpMatrix  *P, const OsqpVectorf *x);
c_float CscMatrix_quad_form(const CscMatrix  *P, const OsqpVectorf *x);
c_float CsrMatrix_quad_form(const CsrMatrix  *P, const OsqpVectorf *x);


# ifdef __cplusplus
}
# endif // ifdef __cplusplus

#endif // ifndef LIN_ALG_H
