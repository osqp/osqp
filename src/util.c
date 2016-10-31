#include "util.h"

/* ================================= DEBUG FUNCTIONS ======================= */
#if PRINTLEVEL > 2

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
                c_print("% 1.4e,  ", M[j*m+i]);

            else
                // c_print("% 14.12e;  ", M[j*m+i]);
                c_print("% 1.4e;  ", M[j*m+i]);
        }
        if (i < m - 1){
            c_print("\n\t");
        }
    }
    c_print("\n");
}

/* Print vector */
void print_vec(c_float * V, c_int n, char *name){
	print_dns_matrix(V, n, 1, name);
}

#endif
