#include <stdio.h>
#include "osqp.h"
#include "cs.h"
#include "minunit.h"
#include "lin_alg/matrices.h"

#define LINALG_TOL 1e-05

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

c_int test_vec_norms(){
    c_int exitflag = 0;  // Initialize exitflag to 0
    c_float norm2_diff, norm2_sq, norm2, normInf;
    c_float add_scaled[t2_n];

    // Norm of the difference
    norm2_diff = vec_norm2_diff(t2_v1, t2_v2, t2_n);
    if (c_abs(norm2_diff - t2_norm2_diff)>LINALG_TOL) {
        // c_print("norm2_diff = %.4f\n", norm2_diff);
        // c_print("t2_norm2_diff = %.4f\n", t2_norm2_diff);
        // c_print("difference = %.4e\n", c_abs(norm2_diff - t2_norm2_diff));
        c_print("\nError in norm of difference test!");
        exitflag = 1;
    }

    // Add scaled
    vec_copy(add_scaled, t2_v1, t2_n);  // Copy vector v1 in another vector
    vec_add_scaled(add_scaled, t2_v2, t2_n, t2_sc);
    if(vec_norm2_diff(add_scaled, t2_add_scaled, t2_n)>LINALG_TOL) {
        c_print("\nError in add scaled test!");
        exitflag = 1;
    }

    // Norm2 squared
    norm2_sq = vec_norm2_sq(t2_v1, t2_n);
    if (c_abs(norm2_sq - t2_norm2_sq)>LINALG_TOL) {
        c_print("\nError in norm 2 squared test!");
        exitflag = 1;
    }

    // Norm2
    norm2 = vec_norm2(t2_v1, t2_n);
    if (c_abs(norm2 - t2_norm2)>LINALG_TOL) {
        c_print("\nError in norm 2 test!");
        exitflag = 1;
    }

    // NormInf
    normInf = vec_normInf(t2_v1, t2_n);
    if (c_abs(normInf - t2_normInf)>LINALG_TOL) {
        c_print("\nError in norm inf test!");
        exitflag = 1;
    }


    if (exitflag == 1) c_print("\n");  // Print line for aesthetic reasons
    return exitflag;
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

    printf("2) Test vector norms: ");
    tempflag = test_vec_norms();
    if (!tempflag) c_print("OK!\n");
    exitflag += tempflag;


    mu_assert("Error in linear algebra tests", exitflag != 1 );

    return 0;
}
