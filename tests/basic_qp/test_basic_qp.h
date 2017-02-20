#include "osqp.h"     // OSQP API
#include "cs.h"       // CSC data structure
#include "util.h"     // Utilities for testing
#include "minunit.h"  // Basic testing script header

#include "basic_qp/basic_qp.h"


static char * test_basic_qp()
{
    /* local variables */
    c_int exitflag = 0;  // No errors

    // Problem settings
    Settings * settings = (Settings *)c_malloc(sizeof(Settings));

    // Structures
    Work * work;  // Workspace
    Data * data;  // Data
    basic_qp_sols_data *  sols_data = generate_problem_basic_qp_sols_data();

    // Populate data
    data = generate_problem_basic_qp();
    sols_data = generate_problem_basic_qp_sols_data();

    c_print("QP Tests:\n");

    c_print("   Test basic QP problem 1: ");

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

        // Compare solver statuses
        if ( work->info->status_val != sols_data->status_test ) {
            c_print("\nError in solver status!");
            exitflag = 1;
        }
        // Compare primal solutions
        if (vec_norm2_diff(work->solution->x, sols_data->x_test, data->n) > 1e-4) {
            c_print("\nError in primal solution!");
            exitflag = 1;
        }
        // Compare dual solutions
        if (vec_norm2_diff(work->solution->y, sols_data->y_test, data->m) > 1e-4) {
            print_vec(work->solution->y, data->m, "y_solution");
            print_vec(sols_data->y_test, data->m, "y_test");
            c_print("\nError in dual solution!");
            exitflag = 1;
        }
        // Compare objective values
        if (c_absval(work->info->obj_val - sols_data->obj_value_test) > 1e-4) {
            c_print("\nError in objective value!");
            exitflag = 1;
        }

        // // ====================================================================
        // //    UPDATE DATA
        // // ====================================================================
        //
        // // UPDATE LINEAR COST
        // osqp_update_lin_cost(work, basic_qp_q_new);
        // if (vec_norm2_diff(work->data->q, basic_qp_q_new, basic_qp_n) > TESTS_TOL) {
        //     c_print("\nError in updating linear cost!");
        //     exitflag = 1;
        // }
        //
        // // UPDATE BOUNDS
        // if (osqp_update_bounds(work, basic_qp_l_new, basic_qp_u_new)) {
        //     c_print("\nError in bounds ordering!");
        //     exitflag = 1;
        // } else {
        //     if (vec_norm2_diff(work->data->l, basic_qp_l_new, basic_qp_m) > TESTS_TOL) {
        //       c_print("\nError in updating bounds!");
        //       exitflag = 1;
        //     }
        //     if (vec_norm2_diff(work->data->u, basic_qp_u_new, basic_qp_m) > TESTS_TOL) {
        //       c_print("\nError in updating bounds!");
        //       exitflag = 1;
        //     }
        // }
        //
        // // UPDATE LOWER BOUND
        // if (osqp_update_lower_bound(work, basic_qp_lA)) {
        //     c_print("\nError in bounds ordering!");
        //     exitflag = 1;
        // } else {
        //     if (vec_norm2_diff(work->data->l, basic_qp_lA, basic_qp_m) > TESTS_TOL) {
        //         c_print("\nError in updating lower bound!");
        //         exitflag = 1;
        //     }
        // }
        //
        // // UPDATE UPPER BOUND
        // if (osqp_update_upper_bound(work, basic_qp_uA)) {
        //     c_print("\nError in bounds ordering!");
        //     exitflag = 1;
        // } else {
        //     if (vec_norm2_diff(work->data->u, basic_qp_uA, basic_qp_m) > TESTS_TOL) {
        //         c_print("\nError in updating upper bound!");
        //         exitflag = 1;
        //     }
        // }
        //
        // // ====================================================================
        //
        //
        // // ====================================================================
        // //    UPDATE SETTINGS
        // // ====================================================================
        //
        // // UPDATE MAXIMUM ITERATION NUMBER
        // if (osqp_update_max_iter(work, 77)) {
        //     c_print("\nError in max_iter value!");
        //     exitflag = 1;
        // } else {
        //     if (work->settings->max_iter != 77) {
        //         c_print("\nError in updating max_iter!");
        //         exitflag = 1;
        //     }
        // }
        //
        // // UPDATE ABSOLUTE TOLERANCE
        // c_float new_eps_abs = 0.002;
        // if (osqp_update_eps_abs(work, new_eps_abs)) {
        //     c_print("\nError in absolute tolerance value!");
        //     exitflag = 1;
        // } else {
        //     if ( c_absval(work->settings->eps_abs - new_eps_abs) > TESTS_TOL  ) {
        //         c_print("\nError in updating absolute tolerance!");
        //         exitflag = 1;
        //     }
        // }
        //
        // // UPDATE RELATIVE TOLERANCE
        // c_float new_eps_rel = 5.61e-3;
        // if (osqp_update_eps_rel(work, new_eps_rel)) {
        //     c_print("\nError in relative tolerance value!");
        //     exitflag = 1;
        // } else {
        //     if ( c_absval(work->settings->eps_rel - new_eps_rel) > TESTS_TOL) {
        //         c_print("\nError in updating relative tolerance!");
        //         exitflag = 1;
        //     }
        // }
        //
        // // UPDATE RELAXATION PARAMETER
        // c_float new_alpha = 0.17;
        // if (osqp_update_alpha(work, new_alpha)) {
        //     c_print("\nError in relaxation parameter value!");
        //     exitflag = 1;
        // } else {
        //     if ( c_absval(work->settings->alpha - new_alpha)> TESTS_TOL) {
        //         c_print("\nError in updating relaxation parameter!");
        //         exitflag = 1;
        //     }
        // }
        //
        // // UPDATE REGULARIZATION PARAMETER IN POLISHING
        // c_float new_delta = 2.2e-04;
        // if (osqp_update_delta(work, new_delta)) {
        //     c_print("\nError in regularization parameter value!");
        //     exitflag = 1;
        // } else {
        //     if ( c_absval(work->settings->delta - new_delta)>TESTS_TOL) {
        //         c_print("\nError in updating regularization parameter!");
        //         exitflag = 1;
        //     }
        // }
        //
        // // UPDATE POLISHING
        // if (osqp_update_polishing(work, 0)) {
        //     c_print("\nError in polishing value!");
        //     exitflag = 1;
        // } else {
        //     if (work->settings->polishing != 0 ||
        //         work->info->polish_time != 0.0) {
        //         c_print("\nError in updating polishing!");
        //         exitflag = 1;
        //     }
        // }
        //
        // // UPDATE NUMBER OF ITERATIVE REFINEMENT STEPS IN POLISHING
        // if (osqp_update_pol_refine_iter(work, 14)) {
        //     c_print("\nError in pol_refine_iter value!");
        //     exitflag = 1;
        // } else {
        //     if (work->settings->pol_refine_iter != 14) {
        //         c_print("\nError in updating iterative refinement steps!");
        //         exitflag = 1;
        //     }
        // }
        //
        // // UPDATE VERBOSE
        // if (osqp_update_verbose(work, 1)) {
        //     c_print("\nError in verbose value!");
        //     exitflag = 1;
        // } else {
        //     if (work->settings->verbose != 1) {
        //         c_print("\nError in updating verbose setting!");
        //         exitflag = 1;
        //     }
        // }
        //
        // // UPDATE WARM STARTING
        // if (osqp_update_warm_start(work, 1)) {
        //     c_print("\nError in verbose value!");
        //     exitflag = 1;
        // } else {
        //     if (work->settings->warm_start != 1) {
        //         c_print("\nError in updating warm starting!");
        //         exitflag = 1;
        //     }
        // }


        // ====================================================================

        // Clean workspace
        osqp_cleanup(work);
        clean_problem_basic_qp(data);
        clean_problem_basic_qp_sols_data(sols_data);

    }

    mu_assert("\nError in basic QP test.", exitflag == 0 );
    if (exitflag == 0)
        c_print("OK!\n");


    // Cleanup
    c_free(settings);

    return 0;
}
