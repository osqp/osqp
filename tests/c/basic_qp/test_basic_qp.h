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
    Settings * settings = (Settings *)c_malloc(sizeof(Settings));

    // Structures
    Work * work;  // Workspace
    Data * data;  // Data

    // Populate data from matrices.h
    data = (Data *)c_malloc(sizeof(Data));

    data->n = basic_qp_n;
    data->m = basic_qp_m;
    data->P = csc_matrix(data->n, data->n, basic_qp_P_nnz, basic_qp_P_x, basic_qp_P_i, basic_qp_P_p);
    data->q = basic_qp_q;
    data->A = csc_matrix(data->m, data->n, basic_qp_A_nnz, basic_qp_A_x, basic_qp_A_i, basic_qp_A_p);
    data->lA = basic_qp_lA;
    data->uA = basic_qp_uA;


    c_print("\nTest basic QP problem 1: ");

    // Define Solver settings as default
    set_default_settings(settings);
    settings->max_iter = 200;
    settings->alpha = 1.6;
    settings->polishing = 0;
    settings->verbose = 0;

    // Setup workspace
    work = osqp_setup(data, settings);

    if (!work) {
        c_print("Setup error!\n");
        exitflag = 1;
    } else {

        // Solve Problem
        osqp_solve(work);

        // Check if problem is infeasible
        if (basic_qp_sol_status == 1) {   // infeasible
            if (work->info->status_val != OSQP_INFEASIBLE) {
                c_print("\nError in solver status!");
                exitflag = 1;
            }
        } else {
            // Compare solver statuses
            if ( !(work->info->status_val == OSQP_SOLVED && basic_qp_sol_status == 0) ) {
                c_print("\nError in solver status!");
                exitflag = 1;
            }
            // Compare primal solutions
            if (vec_norm2_diff(work->solution->x, basic_qp_sol_x, basic_qp_n) /
                vec_norm2(basic_qp_sol_x, basic_qp_n) > 1e-4) {
                c_print("\nError in primal solution!");
                exitflag = 1;
            }
            // Compare dual solutions
            if (vec_norm2_diff(work->solution->lambda, basic_qp_sol_lambda, basic_qp_m) /
                vec_norm2(basic_qp_sol_lambda, basic_qp_m) > 1e-4) {
                c_print("\nError in dual solution!");
                exitflag = 1;
            }
            // Compare objective values
            if (c_absval(work->info->obj_val - basic_qp_sol_obj_value) /
                c_absval(basic_qp_sol_obj_value) > 1e-4) {
                c_print("\nError in objective value!");
                exitflag = 1;
            }
        }

        // Clean workspace
        osqp_cleanup(work);
        c_free(data->A);
        c_free(data->P);
        c_free(data);


    }

    mu_assert("\nError in basic QP test.", exitflag == 0 );
    if (exitflag == 0)
        c_print("OK!\n");


    // Cleanup
    c_free(settings);

    return 0;
}
