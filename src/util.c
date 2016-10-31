#include "util.h"

/* ================================= DEBUG FUNCTIONS ======================= */
#if PRINTLEVEL > 2

#include <string.h>  // For memcpy function

/* Convert sparse CSC to dense */
c_float * csc_to_dns(csc * M)
{
    c_int i, j=0;  // Predefine row index and column index

    // Initialize matrix of zeros
    c_float * A = (c_float *)c_calloc(M->m * M->n, sizeof(c_float));

    // Allocate elements
    for (c_int idx = 0; idx < M->nnz; idx++)
    {
        // Get row index i (starting from 1)
        i = M->i[idx];

        // Get column index j (increase if necessary) (starting from 1)
		while (M->p[j+1] <= idx) {
			j++;
		}

        // Assign values to A
        A[j*(M->m)+i] = M->x[idx];

    }
    return A;
}

/* Copy sparse CSC matrix B = A. B has been previously created with
new_csc_matrix(...) function
*/
void copy_csc_mat(const csc* A, csc *B){
    memcpy(B->p, A->p, (A->n+1)*sizeof(c_int));
    memcpy(B->i, A->i, (A->nnz)*sizeof(c_int));
    memcpy(B->x, A->x, (A->nnz)*sizeof(c_float));
}


/* Compare sparse matrices */
c_int is_eq_csc(csc *A, csc *B){
    c_int j, i;
    // If number of columns does not coincide, they are not equal.
    if (A->n != B->n) return 0;

    for (j=0; j<A->n; j++){      // Cycle over columns j

        // if column pointer does not coincide, they are not equal
        if (A->p[j] != B->p[j])  return 0;

        for (i=A->p[j]; i<A->p[j+1]; i++){  // Cycle rows i in column j
            if (A->i[i] != B->i[i] ||   // Different row indeces
                c_abs(A->x[i] - B->x[i]) > TESTS_TOL){
                return 0;
            }
        }
    }
    return(1);
}


/* Print a sparse matrix */
void print_csc_matrix(csc* M, char * name)
{
    c_print("%s :\n", name);
    c_int j, i, row_start, row_stop;
    c_int k = 0;
    for(j=0; j<M->n; j++){
        row_start = M->p[j];
        row_stop = M->p[j+1];
        if (row_start == row_stop)
            continue;
        else {
            for(i=row_start; i<row_stop; i++ ){
                c_print("\t[%3u,%3u] = %g\n", M->i[i], j, M->x[k++]);
            }
        }
    }
}


/* Print a dense matrix */
void print_dns_matrix(c_float * M, c_int m, c_int n, char *name)
{
    c_print("%s = \n\t", name);
	for(c_int i=0; i<m; i++){  // Cycle over rows
		for(c_int j=0; j<n; j++){  // Cycle over columns
            if (j < n - 1)
                // c_print("% 14.12e,  ", M[j*m+i]);
                c_print("% 1.6e,  ", M[j*m+i]);

            else
                // c_print("% 14.12e;  ", M[j*m+i]);
                c_print("% 1.6e;  ", M[j*m+i]);
        }
        if (i < m - 1){
            c_print("\n\t");
        }
    }
    c_print("\n");
}

/* Print vector */
void print_vec(c_float * V, c_int n, char *name){
	print_dns_matrix(V, 1, n, name);
}

#endif
