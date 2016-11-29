#include "osqp.h"     // OSQP API
// #include "cs.h"       // CSC data structure
// #include "util.h"     // Utilities for testing
#include "minunit.h"  // Basic testing script header

#ifndef BASIC_QP2_MATRICES_H
#define BASIC_QP2_MATRICES_H
#include "basic_qp2/matrices.h"
#endif



static char * test_basic_qp2()
{
    /* local variables */
    c_int exitflag = 0;  // No errors

    // Problem settings
    Settings * settings = (Settings *)c_malloc(sizeof(Settings));

    // Structures
    Work * work;  // Workspace
    Data * data;  // Data

    // Populate data from matrices.h
    data = (Data *)c_malloc(sizeof(Data));

    data->n = basic_qp2_n;
    data->m = basic_qp2_m;
    data->P = csc_matrix(data->n, data->n, basic_qp2_P_nnz, basic_qp2_P_x, basic_qp2_P_i, basic_qp2_P_p);
    data->q = basic_qp2_q;
    data->A = csc_matrix(data->m, data->n, basic_qp2_A_nnz, basic_qp2_A_x, basic_qp2_A_i, basic_qp2_A_p);
    data->lA = basic_qp2_lA;
    data->uA = basic_qp2_uA;
    // data->lx = basic_qp2_lx;
    // data->ux = basic_qp2_ux;


    c_print("\nTest basic QP problem 2\n");
    c_print("-----------------------\n");

    // Define Solver settings as default
    set_default_settings(settings);
    settings->max_iter = 1000;
    settings->alpha = 1.6;

    // Setup workspace
    work = osqp_setup(data, settings);

    if (!work) {
        c_print("Setup error!\n");
        exitflag = 1;
    }
    else {
    // DEBUG
    // print_csc_matrix(work->data->P, "P");
    // print_vec(work->data->q, work->data->n, "q");
    // print_csc_matrix(work->data->A, "A");
    // print_vec(work->data->lA, work->data->m, "lA");
    // print_vec(work->data->uA, work->data->m, "uA");
    // print_vec(work->data->lx, work->data->n, "lx");
    // print_vec(work->data->ux, work->data->n, "ux");



    // Solve Problem
    osqp_solve(work);

    // Print solution
    #if PRINTLEVEL > 2
    print_vec(work->solution->x, work->data->n, "x");
    print_vec(work->solution->lambda, work->data->m, "lambda");
    #endif

    // Clean workspace
    osqp_cleanup(work);
    }

    mu_assert("\nError in basic QP 2 test.", exitflag == 0 );


    // Cleanup
    c_free(settings);
    c_free(data->A);
    c_free(data->P);
    c_free(data);

    return 0;
}
