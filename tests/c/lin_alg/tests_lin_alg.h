#include <stdio.h>
#include "osqp.h"
#include "cs.h"
#include "util.h"
#include "minunit.h"

#ifndef LIN_ALG_MATRICES_H
#define LIN_ALG_MATRICES_H
#include "lin_alg/matrices.h"
#endif

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
    c_free(Asp);  // Do not free with function free_csc_matrix
    c_free(Adns); // because of vars from file matrices.h

    return (norm_diff > TESTS_TOL);
}

c_int test_vec_operations(){
    c_int exitflag = 0;  // Initialize exitflag to 0
    c_float norm2_diff, norm2_sq, norm2, vecprod; // normInf;
    c_float ew_reciprocal[t2_n];
    c_float * add_scaled;

    // Norm of the difference
    norm2_diff = vec_norm2_diff(t2_v1, t2_v2, t2_n);
    if (c_absval(norm2_diff - t2_norm2_diff)>TESTS_TOL) {
        c_print("\nError in norm of difference test!");
        exitflag = 1;
    }

    // Add scaled
    add_scaled = vec_copy(t2_v1, t2_n);
    vec_add_scaled(add_scaled, t2_v2, t2_n, t2_sc);
    if(vec_norm2_diff(add_scaled, t2_add_scaled, t2_n)>TESTS_TOL) {
        c_print("\nError in add scaled test!");
        exitflag = 1;
    }

    // Norm2 squared
    norm2_sq = vec_norm2_sq(t2_v1, t2_n);
    if (c_absval(norm2_sq - t2_norm2_sq)>TESTS_TOL) {
        c_print("\nError in norm 2 squared test!");
        exitflag = 1;
    }

    // Norm2
    norm2 = vec_norm2(t2_v1, t2_n);
    if (c_absval(norm2 - t2_norm2)>TESTS_TOL) {
        c_print("\nError in norm 2 test!");
        exitflag = 1;
    }

    // // NormInf
    // normInf = vec_normInf(t2_v1, t2_n);
    // if (c_absval(normInf - t2_normInf)>TESTS_TOL) {
    //     c_print("\nError in norm inf test!");
    //     exitflag = 1;
    // }

    vec_ew_recipr(t2_v1, ew_reciprocal, t2_n);
    if(vec_norm2_diff(ew_reciprocal, t2_ew_reciprocal, t2_n)>TESTS_TOL) {
        c_print("\nError in elementwise reciprocal test!");
        exitflag = 1;
    }


    vecprod = vec_prod(t2_v1, t2_v2, t2_n);
    if(c_absval(vecprod - t2_vec_prod) > TESTS_TOL){
        c_print("\nError in vector product");
        exitflag = 1;
    }

    // cleanup
    c_free(add_scaled);

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

    // // Initialize test matrices
    // Ad = csc_spalloc(t3_n, t3_n, t3_A_nnz, 1, 0);
    // dA = csc_spalloc(t3_n, t3_n, t3_A_nnz, 1, 0);
    // A_ewsq = csc_spalloc(t3_n, t3_n, t3_A_nnz, 1, 0);
    // A_ewabs = csc_spalloc(t3_n, t3_n, t3_A_nnz, 1, 0);
    //
    // // Copy values of matrix A in all of test matrices
    // copy_csc_mat(t3_A, Ad);
    // copy_csc_mat(t3_A, dA);
    // copy_csc_mat(t3_A, A_ewsq);
    // copy_csc_mat(t3_A, A_ewabs);

    // Copy matrices
    Ad = copy_csc_mat(t3_A);
    dA = copy_csc_mat(t3_A);
    A_ewsq = copy_csc_mat(t3_A);
    A_ewabs = copy_csc_mat(t3_A);


    // Premultiply matrix A
    mat_premult_diag(dA, t3_d);

    if (!is_eq_csc(dA, t3_dA, TESTS_TOL)) {
        c_print("\nError in premultiply test!");
        exitflag = 1;
    }

    // Postmultiply matrix A
    mat_postmult_diag(Ad, t3_d);

    // print_csc_matrix(Ad, "Ad");
    // print_csc_matrix(t3_Ad, "t3_Ad");
    if (!is_eq_csc(Ad, t3_Ad, TESTS_TOL)) {
        c_print("\nError in postmultiply test!");
        exitflag = 1;
    }

    // Elementwise square
    mat_ew_sq(A_ewsq);
    if (!is_eq_csc(A_ewsq, t3_A_ewsq, TESTS_TOL)) {
        c_print("\nError in elementwise square test!");
        exitflag = 1;
    }

    // Elementwise absolute value
    mat_ew_abs(A_ewabs);
    if (!is_eq_csc(A_ewabs, t3_A_ewabs, TESTS_TOL)) {
        c_print("\nError in elementwise absolute value test!");
        exitflag = 1;
    }


    // cleanup
    c_free(t3_A);
    c_free(t3_Ad);
    c_free(t3_dA);
    c_free(t3_A_ewsq);
    c_free(t3_A_ewabs);
    csc_spfree(Ad);
    csc_spfree(dA);
    csc_spfree(A_ewsq);
    csc_spfree(A_ewabs);

    return exitflag;
}


c_int test_mat_vec_multiplication(){
    csc *A, *Pu;   // Matrices from matrices.h
    c_float Ax[t4_m], ATy[t4_n], Px[t4_n];
    c_float *Ax_cum, *ATy_cum, *Px_cum;
    c_int exitflag=0;

    // Compute sparse matrices A and P from vectors stored in matrices.h
    A = csc_matrix(t4_m, t4_n, t4_A_nnz, t4_A_x, t4_A_i, t4_A_p);
    Pu = csc_matrix(t4_n, t4_n, t4_Pu_nnz, t4_Pu_x, t4_Pu_i, t4_Pu_p);

    // Matrix-vector multiplication:  y = Ax
    mat_vec(A, t4_x, Ax, 0);
    if(vec_norm2_diff(Ax, t4_Ax, t4_m) > TESTS_TOL){
        c_print("\nError in matrix-vector multiplication!");
        exitflag = 1;
    }

    // Cumulative matrix-vector multiplication:  y += Ax
    Ax_cum = vec_copy(t4_y, t4_m);
    mat_vec(A, t4_x, Ax_cum, 1);
    if(vec_norm2_diff(Ax_cum, t4_Ax_cum, t4_m) > TESTS_TOL){
        c_print("\nError in cumulative matrix-vector multiplication!");
        exitflag = 1;
    }

    // Matrix-transpose-vector multiplication:  x = A'*y
    mat_tpose_vec(A, t4_y, ATy, 0, 0);
    if(vec_norm2_diff(ATy, t4_ATy, t4_n) > TESTS_TOL){
        c_print("\nError in matrix-transpose-vector multiplication!");
        exitflag = 1;
    }

    // Cumulative matrix-transpose-vector multiplication:  x += A'*y
    ATy_cum = vec_copy(t4_x, t4_n);
    mat_tpose_vec(A, t4_y, ATy_cum, 1, 0);
    if(vec_norm2_diff(ATy_cum, t4_ATy_cum, t4_n) > TESTS_TOL){
        c_print("\nError in cumulative matrix-transpose-vector multiplication!");
        exitflag = 1;
    }

    // Symmetric-matrix-vector multiplication (only upper part is stored)
    mat_vec(Pu, t4_x, Px, 0);           // upper traingular part
    mat_tpose_vec(Pu, t4_x, Px, 1, 1);  // lower traingular part (without diagonal)
    if(vec_norm2_diff(Px, t4_Px, t4_n) > TESTS_TOL){
        c_print("\nError in symmetric-matrix-vector multiplication!");
        exitflag = 1;
    }

    // Cumulative symmetric-matrix-vector multiplication
    Px_cum = vec_copy(t4_x, t4_n);
    mat_vec(Pu, t4_x, Px_cum, 1);           // upper traingular part
    mat_tpose_vec(Pu, t4_x, Px_cum, 1, 1);  // lower traingular part (without diagonal)
    if(vec_norm2_diff(Px_cum, t4_Px_cum, t4_n) > TESTS_TOL){
        c_print("\nError in symmetric-matrix-vector multiplication!");
        exitflag = 1;
    }

    // cleanup
    c_free(Ax_cum);
    c_free(ATy_cum);
    c_free(Px_cum);
    c_free(A);
    c_free(Pu);

    return exitflag;
}


c_int test_extract_upper_triangular(){
    c_int exitflag = 0;
    // c_float * Atriudns, * A_t_triudns;
    // Compute sparse matrix A from vectors stored in matrices.h
    csc * A = csc_matrix(t_ut_n, t_ut_n, t_ut_A_nnz, t_ut_A_x, t_ut_A_i, t_ut_A_p);
    csc * A_ut_triu = csc_matrix(t_ut_n, t_ut_n, t_ut_Atriu_nnz, t_ut_Atriu_x, t_ut_Atriu_i, t_ut_Atriu_p);

    csc * Atriu = csc_to_triu(A);

    if (!is_eq_csc(A_ut_triu, Atriu, TESTS_TOL)) {
        c_print("\nError in forming upper triangular matrix!");
        exitflag = 1;
    }

    // DEBUG Print
    // print_csc_matrix(A_ut_triu, "A_ut_triu");
    // print_csc_matrix(Atriu, "Atriu");
    // Atriudns = csc_to_dns(Atriu);
    // A_t_triudns = csc_to_dns(A_ut_triu);
    // print_dns_matrix(Atriudns, t_ut_n, t_ut_n, "Atriudns");
    // print_dns_matrix(A_t_triudns, t_ut_n, t_ut_n, "A_t_triudns");


    // Cleanup
    c_free(A);
    c_free(A_ut_triu);
    csc_spfree(Atriu);

    return exitflag;
}

c_int test_quad_form_upper_triang(){
    c_int exitflag = 0;
    c_float quad_form_t;

    // Get matrices from data
    csc * Atriu = csc_matrix(t_qpform_n, t_qpform_n, t_qpform_Atriu_nnz, t_qpform_Atriu_x, t_qpform_Atriu_i, t_qpform_Atriu_p);

    // Compute quadratic form
    quad_form_t = quad_form(Atriu, t_qpform_x);

    if (c_absval(quad_form_t - t_qpform_value) >  TESTS_TOL) {
        c_print("\nError in computing quadratic form using upper triangular matrix!");
        exitflag = 1;
    }

    // c_print("quadform_t = %.4f\n", quad_form_t);
    // c_print("t_qpform_value = %.4f\n", t_qpform_value);

    // cleanup
    c_free(Atriu);

    return exitflag;
}

static char * tests_lin_alg()
{
    /* local variables */
    c_int exitflag = 0, tempflag;  // No errors

    c_print("Linear algebra tests:\n");

    c_print("   Construct sparse matrix: ");
    tempflag = test_constr_sparse_mat();
    if (!tempflag) c_print("OK!\n");
    exitflag += tempflag;

    c_print("   Vector operations: ");
    tempflag = test_vec_operations();
    if (!tempflag) c_print("OK!\n");
    exitflag += tempflag;


    c_print("   Matrix operations: ");
    tempflag = test_mat_operations();
    if (!tempflag) c_print("OK!\n");
    exitflag += tempflag;

    c_print("   Matrix-vector multiplication: ");
    tempflag = test_mat_vec_multiplication();
    if (!tempflag) c_print("OK!\n");
    exitflag += tempflag;

    c_print("   Extract upper triangular: ");
    tempflag = test_extract_upper_triangular();
    if (!tempflag) c_print("OK!\n");
    exitflag += tempflag;

    c_print("   Compute QP form from upper triangular matrix: ");
    tempflag = test_quad_form_upper_triang();
    if (!tempflag) c_print("OK!\n");
    exitflag += tempflag;

    mu_assert("\nError in linear algebra tests.", exitflag == 0 );

    return 0;
}
