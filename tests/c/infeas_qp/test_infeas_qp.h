#include "osqp.h"     // OSQP API
#include "cs.h"       // CSC data structure
#include "util.h"     // Utilities for testing
#include "minunit.h"  // Basic testing script header

#ifndef INFEAS_QP_MATRICES_H
#define INFEAS_QP_MATRICES_H
#include "infeas_qp/matrices.h"
#endif



static char * test_infeas_qp()
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

    data->n = infeas_qp_n;
    data->m = infeas_qp_m;
    data->P = csc_matrix(data->n, data->n, infeas_qp_P_nnz, infeas_qp_P_x, infeas_qp_P_i, infeas_qp_P_p);
    data->q = infeas_qp_q;
    data->A = csc_matrix(data->m, data->n, infeas_qp_A_nnz, infeas_qp_A_x, infeas_qp_A_i, infeas_qp_A_p);
    data->lA = infeas_qp_lA;
    data->uA = infeas_qp_uA;


    c_print("\nTest infeasible QP problem\n");
    c_print("--------------------------\n");

    // Define Solver settings as default
    set_default_settings(settings);
    settings->max_iter = 200;
    settings->alpha = 1.6;
    settings->polishing = 1;

    // Setup workspace
    work = osqp_setup(data, settings);

    if (!work) {
        c_print("Setup error!\n");
        exitflag = 1;
    } else {
        // Solve Problem
        osqp_solve(work);

        // Print solution
        #if PRINTLEVEL > 2
        print_vec(work->data->lA, work->data->m, "lA");
        print_vec(work->data->uA, work->data->m, "uA");
        print_vec(work->x + work->data->n, work->data->m, "x_s");
        print_vec(work->z + work->data->n, work->data->m, "z_s");
        #endif

        // Clean workspace
        osqp_cleanup(work);
        c_free(data->A);
        c_free(data->P);
        c_free(data);
    }

    mu_assert("\nError in infeasible QP test.", exitflag == 0 );


    // Cleanup
    c_free(settings);

    return 0;
}
