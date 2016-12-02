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


    c_print("Test basic QP problem 1: ");

    // Define Solver settings as default
    set_default_settings(settings);
    settings->max_iter = 200;
    settings->alpha = 1.6;
    settings->polishing = 1;
    settings->scaling = 0;
    settings->verbose = 0;
    settings->warm_start = 0;

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

        // ====================================================================
        //    UPDATE DATA
        // ====================================================================

        // UPDATE LINEAR COST
        osqp_update_lin_cost(work, basic_qp_q_new);
        if (vec_norm2_diff(work->data->q, basic_qp_q_new, basic_qp_n) > TESTS_TOL) {
            c_print("\nError in updating linear cost!");
            exitflag = 1;
        }

        // UPDATE BOUNDS
        if (osqp_update_bounds(work, basic_qp_lA_new, basic_qp_uA_new)) {
            c_print("\nError in bounds ordering!");
            exitflag = 1;
        } else {
            if (vec_norm2_diff(work->data->lA, basic_qp_lA_new, basic_qp_m) > TESTS_TOL) {
              c_print("\nError in updating bounds!");
              exitflag = 1;
            }
            if (vec_norm2_diff(work->data->uA, basic_qp_uA_new, basic_qp_m) > TESTS_TOL) {
              c_print("\nError in updating bounds!");
              exitflag = 1;
            }
        }

        // UPDATE LOWER BOUND
        if (osqp_update_lower_bound(work, basic_qp_lA)) {
            c_print("\nError in bounds ordering!");
            exitflag = 1;
        } else {
            if (vec_norm2_diff(work->data->lA, basic_qp_lA, basic_qp_m) > TESTS_TOL) {
                c_print("\nError in updating lower bound!");
                exitflag = 1;
            }
        }

        // UPDATE UPPER BOUND
        if (osqp_update_upper_bound(work, basic_qp_uA)) {
            c_print("\nError in bounds ordering!");
            exitflag = 1;
        } else {
            if (vec_norm2_diff(work->data->uA, basic_qp_uA, basic_qp_m) > TESTS_TOL) {
                c_print("\nError in updating upper bound!");
                exitflag = 1;
            }
        }

        // ====================================================================


        // ====================================================================
        //    UPDATE SETTINGS
        // ====================================================================

        // UPDATE MAXIMUM ITERATION NUMBER
        if (osqp_update_max_iter(work, 77)) {
            c_print("\nError in max_iter value!");
            exitflag = 1;
        } else {
            if (work->settings->max_iter != 77) {
                c_print("\nError in updating max_iter!");
                exitflag = 1;
            }
        }

        // UPDATE ABSOLUTE TOLERANCE
        if (osqp_update_eps_abs(work, 7.67e-10)) {
            c_print("\nError in absolute tolerance value!");
            exitflag = 1;
        } else {
            if (work->settings->eps_abs != 7.67e-10) {
                c_print("\nError in updating absolute tolerance!");
                exitflag = 1;
            }
        }

        // UPDATE RELATIVE TOLERANCE
        if (osqp_update_eps_rel(work, 5.61e-3)) {
            c_print("\nError in relative tolerance value!");
            exitflag = 1;
        } else {
            if (work->settings->eps_rel != 5.61e-3) {
                c_print("\nError in updating relative tolerance!");
                exitflag = 1;
            }
        }

        // UPDATE RELAXATION PARAMETER
        if (osqp_update_alpha(work, 0.17)) {
            c_print("\nError in relaxation parameter value!");
            exitflag = 1;
        } else {
            if (work->settings->alpha != 0.17) {
                c_print("\nError in updating relaxation parameter!");
                exitflag = 1;
            }
        }

        // UPDATE REGULARIZATION PARAMETER IN POLISHING
        if (osqp_update_delta(work, 2.2e-9)) {
            c_print("\nError in regularization parameter value!");
            exitflag = 1;
        } else {
            if (work->settings->delta != 2.2e-9) {
                c_print("\nError in updating regularization parameter!");
                exitflag = 1;
            }
        }

        // UPDATE POLISHING
        if (osqp_update_polishing(work, 0)) {
            c_print("\nError in polishing value!");
            exitflag = 1;
        } else {
            if (work->settings->polishing != 0 ||
                work->info->polish_time != 0.0) {
                c_print("\nError in updating polishing!");
                exitflag = 1;
            }
        }

        // UPDATE NUMBER OF ITERATIVE REFINEMENT STEPS IN POLISHING
        if (osqp_update_pol_refine_iter(work, 14)) {
            c_print("\nError in pol_refine_iter value!");
            exitflag = 1;
        } else {
            if (work->settings->pol_refine_iter != 14) {
                c_print("\nError in updating iterative refinement steps!");
                exitflag = 1;
            }
        }

        // UPDATE VERBOSE
        if (osqp_update_verbose(work, 1)) {
            c_print("\nError in verbose value!");
            exitflag = 1;
        } else {
            if (work->settings->verbose != 1) {
                c_print("\nError in updating verbose setting!");
                exitflag = 1;
            }
        }

        // UPDATE WARM STARTING
        if (osqp_update_warm_start(work, 1)) {
            c_print("\nError in verbose value!");
            exitflag = 1;
        } else {
            if (work->settings->warm_start != 1) {
                c_print("\nError in updating warm starting!");
                exitflag = 1;
            }
        }


        // ====================================================================

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
