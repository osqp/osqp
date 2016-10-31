#include <stdio.h>
#include "osqp.h"
#include "cs.h"
#include "util.h"
#include "minunit.h"
#include "lin_alg/matrices.h"


c_int test_constr_sparse_mat(){
    csc * Asp;  // Sparse matrix allocation
    c_float * Adns;  // Conversion to dense matrix
    c_float norm_diff;  // Norm of the difference of converted matrix

    // Compute sparse matrix from data vectors
    Asp = csc_matrix(m, n, Asp_nnz, Asp_x, Asp_i, Asp_p);

    // Convert sparse to dense
    Adns =  csc_to_dns(Asp);

    // DEBUG: Print matrices
    // print_csc_matrix(Asp, "Asp");
    // print_dns_matrix(Adns, m, n, "Adns");
    // print_dns_matrix(A, m, n, "A");

    // Compute norm of the elementwise difference with
    norm_diff = vec_norm2_diff(Adns, A, m*n);

    // Free memory
    c_free(Asp);  // Do not free with function free_csc_matrix because of vars from file matrices.h
    c_free(Adns);

    return (norm_diff > TESTS_TOL);
}

c_int test_vec_operations(){
    c_int exitflag = 0;  // Initialize exitflag to 0
    c_float norm2_diff, norm2_sq, norm2; // normInf;
    c_float add_scaled[t2_n], ew_reciprocal[t2_n];

    // Norm of the difference
    norm2_diff = vec_norm2_diff(t2_v1, t2_v2, t2_n);
    if (c_abs(norm2_diff - t2_norm2_diff)>TESTS_TOL) {
        c_print("\nError in norm of difference test!");
        exitflag = 1;
    }

    // Add scaled
    vec_copy(add_scaled, t2_v1, t2_n);  // Copy vector v1 in another vector
    vec_add_scaled(add_scaled, t2_v2, t2_n, t2_sc);
    if(vec_norm2_diff(add_scaled, t2_add_scaled, t2_n)>TESTS_TOL) {
        c_print("\nError in add scaled test!");
        exitflag = 1;
    }

    // Norm2 squared
    norm2_sq = vec_norm2_sq(t2_v1, t2_n);
    if (c_abs(norm2_sq - t2_norm2_sq)>TESTS_TOL) {
        c_print("\nError in norm 2 squared test!");
        exitflag = 1;
    }

    // Norm2
    norm2 = vec_norm2(t2_v1, t2_n);
    if (c_abs(norm2 - t2_norm2)>TESTS_TOL) {
        c_print("\nError in norm 2 test!");
        exitflag = 1;
    }

    // // NormInf
    // normInf = vec_normInf(t2_v1, t2_n);
    // if (c_abs(normInf - t2_normInf)>TESTS_TOL) {
    //     c_print("\nError in norm inf test!");
    //     exitflag = 1;
    // }

    vec_ew_recipr(t2_v1, ew_reciprocal, t2_n);
    if(vec_norm2_diff(ew_reciprocal, t2_ew_reciprocal, t2_n)>TESTS_TOL) {
        c_print("\nError in elementwise reciprocal test!");
        exitflag = 1;
    }


    if (exitflag == 1) c_print("\n");  // Print line for aesthetic reasons
    return exitflag;
}

// c_int test_mat_concat(){
//     csc * ABcat, * t3_A, * t3_B;
//     c_float * ABcat_dns;
//     c_float norm_diff;
//     c_int exitflag = 0;
//
//     // Construct sparse matrices
//     t3_A = csc_matrix(t3_mA, t3_nA, t3_A_nnz, t3_A_x, t3_A_i, t3_A_p);
//     t3_B = csc_matrix(t3_mB, t3_nA, t3_B_nnz, t3_B_x, t3_B_i, t3_B_p);
//
//     // Stack matrices and store in ABcat
//     ABcat = vstack(t3_A, t3_B);
//
//     // Convert sparse to dense
//     ABcat_dns =  csc_to_dns(ABcat);
//
//
//     // Compute norm of the elementwise difference with
//     norm_diff = vec_norm2_diff(ABcat_dns, t3_AB, ABcat->m*ABcat->n);
//
//
//     // DEBUG: print matrices
//     // print_dns_matrix(t3_AB, t3_mA + t3_mB, t3_nA, "t3_AB");
//     // print_dns_matrix(ABcat_dns, t3_mA + t3_mB, t3_nA, "ABcat_dns");
//
//     if (norm_diff>TESTS_TOL) {
//         c_print("\nError in matrix concatenation test!");
//         exitflag = 1;
//     }
//
//     return exitflag;
//
// }

c_int test_mat_operations(){
    csc *t3_A, *t3_Ad, *t3_dA, *t3_A_ewsq, *t3_A_ewabs; // Matrices from matrices.h
    csc *Ad, *dA, *A_ewsq, *A_ewabs;                    // Matrices used for tests
    c_int exitflag=0;

    // Compute sparse matrix A from vectors stored in matrices.h
    t3_A = csc_matrix(t3_n, t3_n, t3_A_nnz, t3_A_x, t3_A_i, t3_A_p);
    t3_Ad = csc_matrix(t3_n, t3_n, t3_Ad_nnz, t3_Ad_x, t3_Ad_i, t3_Ad_p);
    t3_dA = csc_matrix(t3_n, t3_n, t3_dA_nnz, t3_dA_x, t3_dA_i, t3_dA_p);
    t3_A_ewsq = csc_matrix(t3_n, t3_n, t3_A_ewsq_nnz, t3_A_ewsq_x, t3_A_ewsq_i, t3_A_ewsq_p);
    t3_A_ewabs = csc_matrix(t3_n, t3_n, t3_A_ewabs_nnz, t3_A_ewabs_x, t3_A_ewabs_i, t3_A_ewabs_p);

    // Initialize test matrices
    Ad = new_csc_matrix(t3_n, t3_n, t3_A_nnz);
    dA = new_csc_matrix(t3_n, t3_n, t3_A_nnz);
    A_ewsq = new_csc_matrix(t3_n, t3_n, t3_A_nnz);
    A_ewabs = new_csc_matrix(t3_n, t3_n, t3_A_nnz);

    // Copy values of matrix A in all of test matrices
    copy_csc_mat(t3_A, Ad);
    copy_csc_mat(t3_A, dA);
    copy_csc_mat(t3_A, A_ewsq);
    copy_csc_mat(t3_A, A_ewabs);

    // Premultiply matrix A
    mat_premult_diag(dA, t3_d);

    if (!is_eq_csc(dA, t3_dA)) {
        c_print("\nError in premultiply test!");
        exitflag = 1;
    }

    // Postmultiply matrix A
    mat_postmult_diag(Ad, t3_d);

    if (!is_eq_csc(Ad, t3_Ad)) {
        c_print("\nError in postmultiply test!");
        exitflag = 1;
    }

    // Elementwise square
    mat_ew_sq(A_ewsq);
    if (!is_eq_csc(A_ewsq, t3_A_ewsq)) {
        c_print("\nError in elementwise square test!");
        exitflag = 1;
    }

    // Elementwise absolute value
    mat_ew_abs(A_ewabs);
    if (!is_eq_csc(A_ewabs, t3_A_ewabs)) {
        c_print("\nError in elementwise absolute value test!");
        exitflag = 1;
    }

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

    printf("2) Test vector operations: ");
    tempflag = test_vec_operations();
    if (!tempflag) c_print("OK!\n");
    exitflag += tempflag;

    // printf("3) Test matrix concatenation: ");
    // tempflag = test_mat_concat();
    // if (!tempflag) c_print("OK!\n");
    // exitflag += tempflag;

    printf("3) Test matrix operations: ");
    tempflag = test_mat_operations();
    if (!tempflag) c_print("OK!\n");
    exitflag += tempflag;



    mu_assert("Error in linear algebra tests", exitflag != 1 );

    return 0;
}
