#include "osqp.h"     // OSQP API
#include "cs.h"       // CSC data structure
#include "util.h"     // Utilities for testing
#include "minunit.h"  // Basic testing script header

#ifndef BASIC_QP_MATRICES_H
#define BASIC_QP_MATRICES_H
#include "basic_qp/matrices.h"
#endif



static char * test_basic_qp()
{
    /* local variables */
    c_int exitflag = 0;  // No errors

    // Problem Data
    csc * P, * A;
    c_float * lA, *uA, *lx, *ux;
    c_int n, m;

    // Populate data from matrices.h
    n = basic_qp_n;
    m = basic_qp_m;
    P = csc_matrix(n, n, basic_qp_P_nnz, basic_qp_P_x, basic_qp_P_i, basic_qp_P_p);
    A = csc_matrix(m, n, basic_qp_A_nnz, basic_qp_A_x, basic_qp_A_i, basic_qp_A_p);
    lA = basic_qp_lA;
    uA = basic_qp_uA;
    lx = basic_qp_lx;
    ux = basic_qp_ux;


    printf("\nTest basic QP problem\n");
    printf("---------------------\n");

    // Load problem
    //


    mu_assert("\nError in basic QP test.", exitflag == 0 );

    return 0;
}
