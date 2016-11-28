#include "lin_alg.h"
#include "util.h" // For debugging
// #include <math.h>


/* VECTOR FUNCTIONS ----------------------------------------------------------*/
/* ||a - b||_2
(TODO: is it needed or only for debug? If it is needed only for debug it should go into the util.c file)*/
c_float vec_norm2_diff(const c_float *a, const c_float *b, c_int l) {
    c_float nmDiff = 0.0, tmp;
    c_int i;
    for (i = 0; i < l; ++i) {
        tmp = (a[i] - b[i]);
        nmDiff += tmp * tmp;
    }
    return c_sqrt(nmDiff);
}

/* a += sc*b */
void vec_add_scaled(c_float *a, const c_float *b, c_int n, c_float sc) {
    c_int i;
    for (i = 0; i < n; ++i) {
        a[i] += sc * b[i];
    }
}

/* ||v||_2^2 */
c_float vec_norm2_sq(const c_float *v, c_int l) {
    c_int i;
    c_float nmsq = 0.0;
    for (i = 0; i < l; ++i) {
        nmsq += v[i] * v[i];
    }
    return nmsq;
}

/* ||v||_2 */
c_float vec_norm2(const c_float *v, c_int l) {
    return c_sqrt(vec_norm2_sq(v, l));
}

// /* ||v||_inf (TODO: delete or keep it? not needed now) */
// c_float vec_normInf(const c_float *a, c_int l) {
//     c_float tmp, max = 0.0;
//     c_int i;
//     for (i = 0; i < l; ++i) {
//         tmp = c_absval(a[i]);
//         if (tmp > max)
//             max = tmp;
//     }
//     return max;
// }

/* set vector to scalar */
void vec_set_scalar(c_float *a, c_float sc, c_int n){
    for (c_int i=0; i<n; i++) {
        a[i] = sc;
    }
}

/* add scalar to vector */
void vec_add_scalar(c_float *a, c_float sc, c_int n){
    for (c_int i=0; i<n; i++) {
        a[i] += sc;
    }
}

/* multiply scalar to vector */
void vec_mult_scalar(c_float *a, c_float sc, c_int n){
    for (c_int i=0; i<n; i++) {
        a[i] *= sc;
    }
}



/* copy vector a into output */
c_float * vec_copy(c_float *a, c_int n) {
    c_float * b;
    b = c_malloc(n * sizeof(c_float));
    memcpy(b, a, n * sizeof(c_float));
    return b;
}


/* copy vector a into preallocated vector b */
void prea_vec_copy(c_float *a, c_float * b, c_int n){
    for (c_int i=0; i<n; i++) {
        b[i] = a[i];
    }
}


/* Vector elementwise reciprocal b = 1./a*/
void vec_ew_recipr(const c_float *a, c_float *b, c_int n){
    c_int i;
    for (i=0; i<n; i++){
        b[i] = 1.0/a[i];
    }
}


/* Inner product a'b */
c_float vec_prod(const c_float *a, const c_float *b, c_int n){
    c_float prod = 0.0;
    c_int i; // Index

    for(i = 0;  i < n; i++){
        prod += a[i] * b[i];
    }

    return prod;
}

/* elementwse product a.*b stored in b*/
void vec_ew_prod(const c_float *a, c_float *b, c_int n){
    for(c_int i = 0;  i < n; i++){
        b[i] *= a[i];
    }
}


/* elementwise sqrt of the vector elements */
void vec_ew_sqrt(c_float *a, c_int n){
    for(c_int i = 0;  i < n; i++){
        a[i] = c_sqrt(a[i]);
    }
}



/* MATRIX FUNCTIONS ----------------------------------------------------------*/
/* Premultiply matrix A by diagonal matrix with diagonal d,
i.e. scale the rows of A by d
*/
void mat_premult_diag(csc *A, const c_float *d){
    int j, i;
    for (j=0; j<A->n; j++){  // Cycle over columns
        for (i=A->p[j]; i<A->p[j+1]; i++){   // Cycle every row in the column
            A->x[i] *= d[A->i[i]];  // Scale by corresponding element of d for row i
        }
    }
}

/* Premultiply matrix A by diagonal matrix with diagonal d,
i.e. scale the columns of A by d
*/
void mat_postmult_diag(csc *A, const c_float *d){
    int j, i;
    for (j=0; j<A->n; j++){  // Cycle over columns j
        for (i=A->p[j]; i<A->p[j+1]; i++){  // Cycle every row i in column j
            A->x[i] *= d[j];  // Scale by corresponding element of d for column j
        }
    }
}


/* Elementwise square matrix M
used in matrix equilibration
*/
void mat_ew_sq(csc * A){
    c_int i;
    for (i=0; i<A->nzmax; i++)
    {
        A->x[i] = A->x[i]*A->x[i];
    }
}


/* Elementwise absolute value of matrix M
used in matrix equilibration
TODO: delete or keep it? We may not need this function.
*/
void mat_ew_abs(csc * A){
    c_int i;
    for (i=0; i<A->nzmax; i++) {
        A->x[i] = c_absval(A->x[i]);
    }
}


/* Matrix-vector multiplication
 *    y  =  A*x  (if plus_eq == 0)
 *    y +=  A*x  (if plus_eq == 1)
*/
void mat_vec(const csc *A, const c_float *x, c_float *y, c_int plus_eq) {
    int i, j;
    if (!plus_eq){
        // y = 0
        for (i=0; i<A->m; i++) {
            y[i] = 0;
        }
    }

    // if A is empty
    if (A->nzmax == 0) {
        return;
    }

    // y +=  A*x
    for (j=0; j<A->n; j++) {
        for (i=A->p[j]; i<A->p[j+1]; i++) {
            y[A->i[i]] += A->x[i] * x[j];
        }
    }
}

/* Matrix-transpose-vector multiplication
 *    y  =  A'*x  (if plus_eq == 0)
 *    y +=  A'*x  (if plus_eq == 1)
 * If skip_diag == 1, then diagonal elements of A are assumed to be zero.
*/
void mat_tpose_vec(const csc *A, const c_float *x, c_float *y,
                   c_int plus_eq, c_int skip_diag) {
    int i, j, k;
    if (!plus_eq){
        // y = 0
        for (i=0; i<A->n; i++) {
            y[i] = 0;
        }
    }

    // if A is empty
    if (A->nzmax == 0) {
        return;
    }

    // y +=  A*x
    if (skip_diag) {
  		  for (j=0; j<A->n; j++) {
            for (k=A->p[j]; k < A->p[j+1]; k++) {
                i = A->i[k];
                y[j] += i==j ? 0 : A->x[k]*x[i];
            }
        }
  	} else {
        for (j=0; j<A->n; j++) {
            for (k=A->p[j]; k < A->p[j+1]; k++) {
                y[j] += A->x[k]*x[A->i[k]];
            }
        }
    }
}


/**
 * Compute quadratic form f(x) = 1/2 x' P x
 * @param  P quadratic matrix in CSC form (only upper triangular)
 * @param  x argument float vector
 * @return   quadratic form value
 */
c_float quad_form(const csc * P, const c_float * x){
    c_float quad_form = 0.;
    c_int i, j, ptr;  // Pointers to iterate over matrix: (i,j) a element pointer

    for (j = 0; j < P->n; j++){ // Iterate over columns
        for (ptr = P->p[j]; ptr < P->p[j+1]; ptr++){  // Iterate over rows
            i = P->i[ptr]; // Row index

            if (i == j){  // Diagonal element
                quad_form += .5*P->x[ptr]*x[i]*x[i];
            }
            else if (i < j) {  // Off-diagonal element
                quad_form += P->x[ptr]*x[i]*x[j];
            }
            else { // Element in lower diagonal part
                #if PRINTLEVEL>0
                c_print("ERROR: quad_form matrix is not upper triangular\n");
                #endif
                return OSQP_NULL;
            }
        }
    }
    return quad_form;
}
