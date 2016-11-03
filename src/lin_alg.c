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
//         tmp = c_abs(a[i]);
//         if (tmp > max)
//             max = tmp;
//     }
//     return max;
// }


/* copy vector b into a
(TODO: if it is only needed for tests remove it and put it in util.h)
*/
void vec_copy(c_float *a, const c_float *b, c_int n) {
    for (c_int i=0; i<n; i++) {
        a[i] = b[i];
    }
}


/* Vector elementwise reciprocal b = 1./a*/
void vec_ew_recipr(const c_float *a, c_float *b, c_int n){
    c_int i;
    for (i=0; i<n; i++){
        b[i] = 1.0/a[i];
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
    for (i=0; i<A->nnz; i++)
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
    for (i=0; i<A->nnz; i++)
    {
        A->x[i] = c_abs(A->x[i]);
    }
}


/* Matrix-vector multiplication
 *    y  =  A*x  (if plus_eq == 0)
 *    y +=  A*x  (if plus_eq == 1)
*/
void mat_vec(const csc *A, const c_float *x, c_float *y, c_int plus_eq){
    int j, i;
    if (!plus_eq){
      // y = 0
      for(i=0; i<A->m; i++){
        y[i] = 0;
      }
    }
    for (j=0; j<A->n; j++){
        for (i=A->p[j]; i<A->p[j+1]; i++){
            y[A->i[i]] += A->x[i] * x[j];
        }
    }
}

/* Vertically concatenate arrays Z = [A' B']'
(uses MALLOC to create inner arrays x, i, p within Z)
*/
// csc * vstack(csc *A, csc *B){
//     c_int j, i;  // row i,  col j
//     c_int z_count=0;
//     csc * Z;
//
//     // Initialize Z variable (concatenate dims horizontally, add also nnz)
//     Z = new_csc_matrix(A->m + B->m, A->n, A->p[A->n] + B->p[B->n]);
//
//     // Assign elements
//     Z->p[0] = 0;
//
//
//     for (j=0; j<A->n; j++){ // Cycle over columns
//         // Shift column pointer to include elements of both matrices
//         Z->p[j] = A->p[j] + B->p[j];
//
//         // Add A elements
//         for (i=A->p[j-1]; i<A->p[j]; i++){ // Add all elements in column j
//             Z->i[z_count] = A->i[i];
//             Z->x[z_count] = A->x[i];
//             z_count++;
//         }
//
//         // Add B elements
//         for (i=B->p[j-1]; i<B->p[j]; i++){ // Add all elements in column j
//             Z->i[z_count] = B->i[i] + A->m; // Shift row idx by height of A
//             Z->x[z_count] = B->x[i];
//             z_count++;
//         }
//     }
//
//     // DEBUG: Print resulting sparse matrix
//     // print_csc_matrix(Z, "Z");
//
//     return Z;
// }
