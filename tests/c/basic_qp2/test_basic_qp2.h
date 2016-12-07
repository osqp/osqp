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
    data->P = csc_matrix(data->n, data->n, basic_qp2_P_nnz, basic_qp2_P_x,
                         basic_qp2_P_i, basic_qp2_P_p);
    data->q = basic_qp2_q1;
    data->A = csc_matrix(data->m, data->n, basic_qp2_A_nnz, basic_qp2_A_x,
                         basic_qp2_A_i, basic_qp2_A_p);
    data->l = basic_qp2_lA;
    data->u = basic_qp2_uA1;


    c_print("   Test basic QP problem 2: ");

    // Define Solver settings as default
    set_default_settings(settings);
    settings->max_iter = 1000;
    settings->alpha = 1.6;
    settings->warm_start = 1;
    settings->polishing = 0;
    settings->verbose = 0;

    // Setup workspace
    work = osqp_setup(data, settings);

    if (!work) {
        c_print("Setup error!\n");
        exitflag = 1;
    }
    else {
        // Solve Problem first time
        osqp_solve(work);

        // Check if problem is infeasible
        if (basic_qp2_sol_status1 == 1) {   // infeasible
            if (work->info->status_val != OSQP_INFEASIBLE) {
                c_print("\nError in solver status!");
                exitflag = 1;
            }
        } else {
            // Compare solver statuses
            if ( !(work->info->status_val == OSQP_SOLVED && basic_qp2_sol_status1 == 0) ) {
                c_print("\nError in solver status!");
                exitflag = 1;
            }
            // Compare primal solutions
            if (vec_norm2_diff(work->solution->x, basic_qp2_sol_x1, basic_qp2_n) /
                vec_norm2(basic_qp2_sol_x1, basic_qp2_n) > 1e-4) {
                c_print("\nError in primal solution!");
                exitflag = 1;
            }
            // Compare dual solutions
            if (vec_norm2_diff(work->solution->y, basic_qp2_sol_lambda1, basic_qp2_m) /
                vec_norm2(basic_qp2_sol_lambda1, basic_qp2_m) > 1e-4) {
                c_print("\nError in dual solution!");
                exitflag = 1;
            }
            // Compare objective values
            if (c_absval(work->info->obj_val - basic_qp2_sol_obj_value1) /
                c_absval(basic_qp2_sol_obj_value1) > 1e-4) {
                c_print("\nError in objective value!");
                exitflag = 1;
            }
        }

        // ====================================================================


        // Set polishing to 0
        osqp_update_polishing(work, 0);
        osqp_update_max_iter(work, 200);
        // Solve Problem second time (warm start, reuse factorization, no polishing)
        osqp_solve(work);

        // Check if problem is infeasible
        if (basic_qp2_sol_status1 == 1) {   // infeasible
            if (work->info->status_val != OSQP_INFEASIBLE) {
                c_print("\nError in solver status!");
                exitflag = 1;
            }
        } else {
            // Compare solver statuses
            if ( !(work->info->status_val == OSQP_SOLVED && basic_qp2_sol_status1 == 0) ) {
                c_print("\nError in solver status!");
                exitflag = 1;
            }
            // Compare primal solutions
            if (vec_norm2_diff(work->solution->x, basic_qp2_sol_x1, basic_qp2_n) /
                vec_norm2(basic_qp2_sol_x1, basic_qp2_n) > 1e-4) {
                c_print("\nError in primal solution!");
                exitflag = 1;
            }
            // Compare dual solutions
            if (vec_norm2_diff(work->solution->y, basic_qp2_sol_lambda1, basic_qp2_m) /
                vec_norm2(basic_qp2_sol_lambda1, basic_qp2_m) > 1e-4) {
                c_print("\nError in dual solution!");
                exitflag = 1;
            }
            // Compare objective values
            if (c_absval(work->info->obj_val - basic_qp2_sol_obj_value1) /
                c_absval(basic_qp2_sol_obj_value1) > 1e-4) {
                c_print("\nError in objective value!");
                exitflag = 1;
            }
        }


        // ====================================================================

        // Modify linear cost and upper bound
        osqp_update_lin_cost(work, basic_qp2_q2);
        osqp_update_upper_bound(work, basic_qp2_uA2);
        // Solve Problem third time (with different data now)
        osqp_solve(work);

        // Check if problem is infeasible
        if (basic_qp2_sol_status2 == 1) {   // infeasible
            if (work->info->status_val != OSQP_INFEASIBLE) {
                c_print("\nError in solver status!");
                exitflag = 1;
            }
        } else {
            // Compare solver statuses
            if ( !(work->info->status_val == OSQP_SOLVED && basic_qp2_sol_status2 == 0) ) {
                c_print("\nError in solver status!");
                exitflag = 1;
            }
            // Compare primal solutions
            if (vec_norm2_diff(work->solution->x, basic_qp2_sol_x2, basic_qp2_n) /
                vec_norm2(basic_qp2_sol_x2, basic_qp2_n) > 1e-4) {
                c_print("\nError in primal solution!");
                exitflag = 1;
            }
            // Compare dual solutions
            if (vec_norm2_diff(work->solution->y, basic_qp2_sol_lambda2, basic_qp2_m) /
                vec_norm2(basic_qp2_sol_lambda2, basic_qp2_m) > 1e-4) {
                c_print("\nError in dual solution!");
                exitflag = 1;
            }
            // Compare objective values
            if (c_absval(work->info->obj_val - basic_qp2_sol_obj_value2) /
                c_absval(basic_qp2_sol_obj_value2) > 1e-4) {
                c_print("\nError in objective value!");
                exitflag = 1;
            }
        }

        // Clean workspace
        osqp_cleanup(work);
    }

    mu_assert("\nError in basic QP 2 test.", exitflag == 0);
    if (exitflag == 0)
        c_print("OK!\n");

    // Cleanup
    c_free(settings);
    c_free(data->A);
    c_free(data->P);
    c_free(data);

    return 0;
}
