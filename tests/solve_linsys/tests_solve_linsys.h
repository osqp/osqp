#include <stdio.h>
#include "osqp.h"
#include "cs.h"
#include "util.h"
#include "minunit.h"
#include "lin_sys.h"


#ifndef SOLVE_LINSYS_MATRICES_H
#define SOLVE_LINSYS_MATRICES_H
#include "solve_linsys/matrices.h"
#endif


// c_int test_form_KKT(){
//     c_int exitflag = 0;
//     csc *t6_P, *t6_A, *t6_KKT;
//     csc * KKT;
//     // c_float * KKTdns, * t6_KKTdns;
//
//     // Construct sparse matrices from matrices.h
//     t6_KKT = csc_matrix(t6_n+t6_m, t6_n+t6_m, t6_KKT_nnz, t6_KKT_x, t6_KKT_i, t6_KKT_p);
//     t6_P = csc_matrix(t6_n, t6_n, t6_P_nnz, t6_P_x, t6_P_i, t6_P_p);
//     t6_A = csc_matrix(t6_m, t6_n, t6_A_nnz, t6_A_x, t6_A_i, t6_A_p);
//
//
//     // Construct KKT matrix
//     KKT = form_KKT(t6_P, t6_A, t6_rho);
//
//     // DEBUG
//     // KKTdns =  csc_to_dns(KKT);
//     // t6_KKTdns =  csc_to_dns(t6_KKT);
//     // print_dns_matrix(KKTdns, t6_n+t6_m, t6_n+t6_m, "KKTdns");
//     // print_dns_matrix(t6_KKTdns, t6_n+t6_m, t6_n+t6_m, "t6_KKTdns");
//
//
//
//     // Print results // Only for DEBUG
//     // print_csc_matrix(t6_KKT, "t6_KKT");
//
//     // c_print("t6_KKT_n = %i\n", t6_n + t6_m);
//
//     if (!is_eq_csc(KKT, t6_KKT, TESTS_TOL)) {
//         c_print("\nError in forming KKT matrix!");
//         exitflag = 1;
//     }
//
//     // Cleanup
//     c_free(t6_KKT);
//     c_free(t6_P);
//     c_free(t6_A);
//     csc_spfree(KKT);
//
//     return exitflag;
// }


// c_int test_LDL_solve_simple(){
//     c_int exitflag=0;
//     csc *L;   // Matrix from matrices.h
//     Priv * p; // Private structure for storing LDL factorization
//     Settings *settings = (Settings *)c_malloc(sizeof(Settings)); // Settings
//
//     // Convert matrix L from matrices.h to CSC format
//     L = csc_matrix(t5_n, t5_n, t5_L_nnz, t5_L_x, t5_L_i, t5_L_p);
//
//     // Store L, D, P in a private variable
//     p = set_priv(L, t5_Dinv, t5_P);
//
//     // Solve  Ax = b via LDL given factorization
//     solve_lin_sys(settings, p, t5_b);
//
//     if(vec_norm2_diff(t5_b, t5_x, t5_n) > TESTS_TOL){
//         c_print("\nError in the simple LDL linear system solve!");
//         exitflag = 1;
//     }
//
//     // Cleanup
//     c_free(p->bp);
//     c_free(p);
//     c_free(settings);
//     c_free(L);
//
//     return exitflag;
// }
//
//
// c_int test_LDL_solve_random(){
//     c_int exitflag=0;
//     csc *L;   // Matrix from matrices.h
//     Priv * p; // Private structure to form KKT factorization
//     Settings *settings = (Settings *)c_malloc(sizeof(Settings)); // Settings
//
//     // Convert matrix L from matrices.h to CSC format
//     L = csc_matrix(t7_n, t7_n, t7_L_nnz, t7_L_x, t7_L_i, t7_L_p);
//
//     // Store L, D, P in a private variable
//     p = set_priv(L, t7_Dinv, t7_P);
//
//     // Solve  Ax = b via LDL given factorization
//     solve_lin_sys(settings, p, t7_b);
//
//     // // DEBUG
//     // c_print("\n");
//     // print_vec(t7_x, t7_n, "\nx_true");
//     // print_vec(t7_b, t7_n, "x_ldl ");
//     // c_print("\ndiff = %.10f\n", vec_norm2_diff(t7_b, t7_x, t7_n));
//
//     if(vec_norm2_diff(t7_b, t7_x, t7_n) > TESTS_TOL){
//         c_print("\nError in the random LDL linear system solve!");
//         exitflag = 1;
//     }
//
//     // Cleanup
//     c_free(p);
//     c_free(settings);
//
//     return exitflag;
// }


c_int test_solveKKT(){
    c_int exitflag=0;
    Priv * p; // Private structure to form KKT factorization
    Settings *settings = (Settings *)c_malloc(sizeof(Settings)); // Settings
    csc *Pu, *A;  // Pu denotes upper triangular part of P

    // Construct sparse matrices from matrices.h
    Pu = csc_matrix(t8_n, t8_n, t8_Pu_nnz, t8_Pu_x, t8_Pu_i, t8_Pu_p);
    A = csc_matrix(t8_m, t8_n, t8_A_nnz, t8_A_x, t8_A_i, t8_A_p);

    // Form and factorize KKT matrix
    settings->rho = t8_rho;
    settings->sigma = t8_sigma;
    p = init_priv(Pu, A, settings, 0);

    // Solve  KKT x = b via LDL given factorization
    solve_lin_sys(settings, p, t8_rhs);

    if(vec_norm2_diff(t8_rhs, t8_x, t8_m + t8_n) > TESTS_TOL){
        c_print("\nError in forming and solving KKT system!");
        exitflag = 1;
    }

    // Cleanup
    free_priv(p);
    c_free(settings);
    c_free(Pu);
    c_free(A);

    return exitflag;
}


static char * tests_solve_linsys()
{
    /* local variables */
    c_int exitflag = 0, tempflag;  // No errors
    c_print("\n");
    c_print("Test solving linear systems:\n");

    // c_print("0) Form KKT matrix: ");
    // tempflag = test_form_KKT();
    // if (!tempflag) c_print("OK!\n");
    // exitflag += tempflag;

    // c_print("1) Test simple linear system solve via LDL: ");
    // tempflag = test_LDL_solve_simple();
    // if (!tempflag) c_print("OK!\n");
    // exitflag += tempflag;

    // c_print("2) Test random linear system solve via LDL: ");
    // tempflag = test_LDL_solve_random();
    // if (!tempflag) c_print("OK!\n");
    // exitflag += tempflag;

    c_print("   Test forming and solving KKT system: ");
    tempflag = test_solveKKT();
    if (!tempflag) c_print("OK!\n");
    exitflag += tempflag;

    mu_assert("\nError in solving linear systems tests.", exitflag == 0 );

    c_print("\n");
    return 0;
}
