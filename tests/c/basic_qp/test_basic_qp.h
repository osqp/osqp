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

    // Problem settings
    Settings * settings = c_malloc(sizeof(Settings));

    // Structures
    Work * work;  // Workspace
    Data * data;  // Data

    // Populate data from matrices.h
    data = c_malloc(sizeof(Data));

    data->n = basic_qp_n;
    data->m = basic_qp_m;
    data->P = csc_matrix(data->n, data->n, basic_qp_P_nnz, basic_qp_P_x, basic_qp_P_i, basic_qp_P_p);
    data->q = basic_qp_q;
    data->A = csc_matrix(data->m, data->n, basic_qp_A_nnz, basic_qp_A_x, basic_qp_A_i, basic_qp_A_p);
    data->lA = basic_qp_lA;
    data->uA = basic_qp_uA;
    data->lx = basic_qp_lx;
    data->ux = basic_qp_ux;
    //TODO: FIX SEGMENTATION FAULT WHEN CREATING lx and ux


    c_print("\nTest basic QP problem\n");
    c_print("---------------------\n");

    // Define Solver settings as default
    set_default_settings(settings);
    settings->max_iter = 200;

    // Setup workspace
    work = osqp_setup(data, settings);

    // Solve Problem
    osqp_solve(work);

    // Clean workspace
    osqp_cleanup(work);

    mu_assert("\nError in basic QP test.", exitflag == 0 );


    // Cleanup
    c_free(settings);

    return 0;
}
