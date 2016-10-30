#include "stdio.h"
#include "osqp.h"
#include "cs.h"
#include "minunit.h"
#include "lin_alg/matrices.h"

#define LINALG_TOL 1e-08

c_int test_constr_sparse_mat(){
    csc * Asp;  // Sparse matrix allocation
    c_float * Adns;  // Conversion to dense matrix
    c_float norm_diff;  // Norm of the difference of converted matrix

    // Compute sparse matrix from data vectors
    Asp = csc_matrix(m, n, Asp_nnz, Asp_x, Asp_i, Asp_p);

    // Convert sparse to dense
    Adns =  csc_to_dns(Asp);

    // Compute norm of the elementwise difference with
    norm_diff = vec_norm2_diff(Adns, A, m*n);

    // Free memory
    c_free(Asp);  // Do not free with function free_csc_matrix because of vars from file matrices.h
    c_free(Adns);

    return (norm_diff > LINALG_TOL);
}


static char * tests_lin_alg()
{
    /* local variables */
    c_int exitflag = 0, tempflag;  // No errors

    printf("Linear algebra tests\n");
    printf("--------------------\n");

    printf("1) Construct sparse matrix: ");
    tempflag = test_constr_sparse_mat();
    if (!tempflag) c_print("OK!\n");
    exitflag += tempflag;



    // Debug:
    // exitflag = 1;


    // print_dns_matrix(A, m, n, "Afile");
    // print_dns_matrix(Asp_dns, m, n, "Aconverted");

    mu_assert("Error in linear algebra tests", exitflag != 1 );

    return 0;
}
