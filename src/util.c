#include "util.h"

/* ================================= DEBUG FUNCTIONS ======================= */
#if PRINTLEVEL > 2

/* Convert sparse CSC to dense */
c_float * csc_to_dns(csc * M)
{
    c_int i, j=1;  // Predefine row index and column index

    // Initialize matrix of zeros
    c_float * A = (c_float *)c_calloc(M->m * M->n, sizeof(c_float));

    // Allocate elements
    for (c_int idx = 0; idx < M->nnz; idx++)
    {
        // Get row index i (starting from 1)
        i = M->i[idx];

        // Get column index j (increase if necessary) (starting from 1)
		while (M->p[j]-1 <= idx) {
			j++;
		}

        // Assign values to A
        A[(j-1)*(M->m)+(i-1)] = M->x[idx];

    }
    return A;
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
                c_print("\t[%3u,%3u] = %g\n", M->i[i-1], j+1, M->x[k++]);
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
