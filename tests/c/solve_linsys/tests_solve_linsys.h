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


c_int test_formKKT(){
    c_int exitflag = 0;
    csc *t6_P, *t6_A, *t6_KKT;
    Priv * p; // Private structure to form KKT factorization
    Settings * settings = c_malloc(sizeof(Settings)); // Settings

    // Construct sparse matrices from matrices.h
    t6_KKT = csc_matrix(t6_n+t6_m, t6_n+t6_m, t6_KKT_nnz, t6_KKT_x, t6_KKT_i, t6_KKT_p);
    t6_P = csc_matrix(t6_n, t6_n, t6_P_nnz, t6_P_x, t6_P_i, t6_P_p);
    t6_A = csc_matrix(t6_n, t6_n, t6_A_nnz, t6_A_x, t6_A_i, t6_A_p);

    // Define settings
    settings->rho = t6_rho;
    p = initPriv(t6_P, t6_A, settings);

    // Print results
    print_csc_matrix(t6_KKT, "t6_KKT");

    return exitflag;
}

c_int test_LDL_solve(){
    csc *L;   // Matrix from matrices.h
    c_float x[t5_n], x_ws[t5_n], diff;
    c_int exitflag=0;

    // Compute sparse matrix A from vectors stored in matrices.h
    L = csc_matrix(t5_n, t5_n, t5_L_nnz, t5_L_x, t5_L_i, t5_L_p);

    // Solve  Ax = b via LDL given factorization
    LDLSolve(x, t5_b, L, t5_D, t5_P, x_ws);
    diff = vec_norm2_diff(x, t5_x, t5_n);
    if(diff > TESTS_TOL){
        c_print("\nError in the LDL linear system solve!");
        exitflag = 1;
    }

    return exitflag;
}


static char * tests_solve_linsys()
{
    /* local variables */
    c_int exitflag = 0, tempflag;  // No errors
    printf("\n");
    printf("Solving linear systems tests\n");
    printf("----------------------------\n");

    printf("0) Form KKT matrix: ");
    tempflag = test_formKKT();
    if (!tempflag) c_print("OK!\n");
    exitflag += tempflag;

    printf("1) Test linear system solve via LDL: ");
    tempflag = test_LDL_solve();
    if (!tempflag) c_print("OK!\n");
    exitflag += tempflag;

    mu_assert("\nError in solving linear systems tests.", exitflag == 0 );

    return 0;
}
