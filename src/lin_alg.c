#include "lin_alg.h"
// #include <math.h>


/* VECTOR FUNCTIONS ----------------------------------------------------------*/
/* ||a - b||_2 */
c_float vec_norm2_diff(const c_float *a, const c_float *b, c_int l) {
    c_float nmDiff = 0.0, tmp;
    c_int i;
    for (i = 0; i < l; ++i) {
        tmp = (a[i] - b[i]);
        nmDiff += tmp * tmp;
    }
    return c_sqrtf(nmDiff);
}

/* a += sc*b */
void vec_add_scaled(c_float *a, const c_float *b, c_int n, const c_float sc) {
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
    return c_sqrtf(vec_norm2_sq(v, l));
}

/* ||v||_inf */
c_float vec_normInf(const c_float *a, c_int l) {
    c_float tmp, max = 0.0;
    c_int i;
    for (i = 0; i < l; ++i) {
        tmp = c_abs(a[i]);
        if (tmp > max)
            max = tmp;
    }
    return max;
}

/* copy vector b into a */
void vec_copy(c_float *a, c_float *b, c_int n) {
    for (c_int i=0; i<n; i++) {
        a[i] = b[i];
    }
}

/* MATRIX FUNCTIONS ----------------------------------------------------------*/
/* Vertically concatenate arrays Z = [A' B']'
(uses MALLOC to create inner arrays x, i, p within Z)
*/
csc * vstack(csc *A, csc *B){
    c_int j, k;
    c_int z_count=0;
    csc * Z;

    // Initialize Z variable (concatenate dims horizontally, add also nnz)
    Z = new_csc_matrix(A->m + B->m, A->n, A->p[A->n] + B->p[B->n]);

    // Assign elements
    Z->p[0] = 1;
    
    for (j=0; j<A->n; j++){ // Cycle over columns
        // Shift column pointer to include elements of both matrices
        Z->p[j+1] = A->p[j+1] + B->p[j+1];

        // Add A elements
        for (k=A->p[j]; k<A->p[j+1]; k++){ // Add all elements in column j
            Z->i[z_count] = A->i[k-1];
            Z->x[z_count] = A->x[k-1];
            z_count++;
        }

        // Add B elements
        for (k=B->p[j]; k<B->p[j+1]; k++){ // Add all elements in column j
            Z->i[z_count] = B->i[k-1] + A->m; // Shift row idx by height of A
            Z->x[z_count] = B->x[k-1];
            z_count++;
        }
    }

    return Z;
}
