#include <catch2/catch.hpp>

#include "osqp_api.h"    /* OSQP API wrapper (public + some private) */
#include "osqp_tester.h" /* Tester helpers */
#include "test_utils.h"  /* Testing Helper functions */

#include "derivative_adjoint_data.h"


TEST_CASE_METHOD(derivative_adjoint_test_fixture, "Adjoint derivative: basic test", "[derivative],[adjoint]")
{
    OSQPInt exitflag;

    // Setup workspace
    exitflag = osqp_setup(&tmpSolver, data->P, data->q,
                          data->A, data->l, data->u,
                          data->m, data->n, settings.get());
    solver.reset(tmpSolver);

    // Setup correct
    mu_assert("Setup error!", exitflag == 0);

    // Solve Problem first time
    osqp_solve(solver.get());

    // Compare solver statuses
    mu_assert("Error in solver status!",
        solver->info->status_val == sols_data->status_test);

    // Compare primal solutions
    mu_assert("Error in primal solution!",
              vec_norm_inf_diff(solver->solution->x, sols_data->x_test,
              data->n) < TESTS_TOL);

    // Compare dual solutions
    mu_assert("Error in dual solution!",
              vec_norm_inf_diff(solver->solution->y, sols_data->y_test,
              data->m) < TESTS_TOL);

    // Compare objective values
    mu_assert("Error in objective value!",
              c_absval(solver->info->obj_val - sols_data->obj_value_test) <
              TESTS_TOL);


    SECTION("Both dx and dy") {
        // Run the derivative computation
        OSQPInt retval = osqp_adjoint_derivative_compute(solver.get(), sols_data->dx_1, sols_data->dy_1);

        mu_assert("Error computing derivatives",
                  retval == OSQP_NO_ERROR);

        // Get the vectors
        std::unique_ptr<OSQPFloat[]> dq(new OSQPFloat[data->n]);
        std::unique_ptr<OSQPFloat[]> dl(new OSQPFloat[data->m]);
        std::unique_ptr<OSQPFloat[]> du(new OSQPFloat[data->m]);

        retval = osqp_adjoint_derivative_get_vec(solver.get(), dq.get(), dl.get(), du.get());

        mu_assert("Error getting vector derivatives",
                  retval == OSQP_NO_ERROR);

        // Get the matrices
        OSQPCscMatrix_ptr dP{(OSQPCscMatrix*) malloc(sizeof(OSQPCscMatrix))};
        OSQPCscMatrix_ptr dA{(OSQPCscMatrix*) malloc(sizeof(OSQPCscMatrix))};

        std::unique_ptr<OSQPFloat[]> dPx(new OSQPFloat[data->P->nzmax]);
        std::unique_ptr<OSQPFloat[]> dAx(new OSQPFloat[data->A->nzmax]);

        // The matrices here share the same row/column pointers and just differ by the data
        csc_set_data(dP.get(), data->n, data->n, data->P->nzmax, dPx.get(), data->P->i, data->P->p);
        csc_set_data(dA.get(), data->m, data->n, data->A->nzmax, dAx.get(), data->A->i, data->A->p);

        retval = osqp_adjoint_derivative_get_mat(solver.get(), dP.get(), dA.get());

        mu_assert("Error getting matrix derivatives",
                  retval == OSQP_NO_ERROR);
    }

    SECTION("Just dx") {
        // Run the derivative computation
        OSQPInt retval = osqp_adjoint_derivative_compute(solver.get(), sols_data->dx_1, sols_data->dy_zeros);

        mu_assert("Error computing derivatives",
                  retval == OSQP_NO_ERROR);

        // Get the vectors
        std::unique_ptr<OSQPFloat[]> dq(new OSQPFloat[data->n]);
        std::unique_ptr<OSQPFloat[]> dl(new OSQPFloat[data->m]);
        std::unique_ptr<OSQPFloat[]> du(new OSQPFloat[data->m]);

        retval = osqp_adjoint_derivative_get_vec(solver.get(), dq.get(), dl.get(), du.get());

        mu_assert("Error getting vector derivatives",
                  retval == OSQP_NO_ERROR);

        // Get the matrices
        OSQPCscMatrix_ptr dP{(OSQPCscMatrix*) malloc(sizeof(OSQPCscMatrix))};
        OSQPCscMatrix_ptr dA{(OSQPCscMatrix*) malloc(sizeof(OSQPCscMatrix))};

        std::unique_ptr<OSQPFloat[]> dPx(new OSQPFloat[data->P->nzmax]);
        std::unique_ptr<OSQPFloat[]> dAx(new OSQPFloat[data->A->nzmax]);

        // The matrices here share the same row/column pointers and just differ by the data
        csc_set_data(dP.get(), data->n, data->n, data->P->nzmax, dPx.get(), data->P->i, data->P->p);
        csc_set_data(dA.get(), data->m, data->n, data->A->nzmax, dAx.get(), data->A->i, data->A->p);

        retval = osqp_adjoint_derivative_get_mat(solver.get(), dP.get(), dA.get());

        mu_assert("Error getting matrix derivatives",
                  retval == OSQP_NO_ERROR);
    }

    SECTION("Just dy") {
        // Run the derivative computation
        OSQPInt retval = osqp_adjoint_derivative_compute(solver.get(), sols_data->dx_zeros, sols_data->dy_1);

        mu_assert("Error computing derivatives",
                    retval == OSQP_NO_ERROR);

        // Get the vectors
        std::unique_ptr<OSQPFloat[]> dq(new OSQPFloat[data->n]);
        std::unique_ptr<OSQPFloat[]> dl(new OSQPFloat[data->m]);
        std::unique_ptr<OSQPFloat[]> du(new OSQPFloat[data->m]);

        retval = osqp_adjoint_derivative_get_vec(solver.get(), dq.get(), dl.get(), du.get());

        mu_assert("Error getting vector derivatives",
                  retval == OSQP_NO_ERROR);

        // Get the matrices
        OSQPCscMatrix_ptr dP{(OSQPCscMatrix*) malloc(sizeof(OSQPCscMatrix))};
        OSQPCscMatrix_ptr dA{(OSQPCscMatrix*) malloc(sizeof(OSQPCscMatrix))};

        std::unique_ptr<OSQPFloat[]> dPx(new OSQPFloat[data->P->nzmax]);
        std::unique_ptr<OSQPFloat[]> dAx(new OSQPFloat[data->A->nzmax]);

        // The matrices here share the same row/column pointers and just differ by the data
        csc_set_data(dP.get(), data->n, data->n, data->P->nzmax, dPx.get(), data->P->i, data->P->p);
        csc_set_data(dA.get(), data->m, data->n, data->A->nzmax, dAx.get(), data->A->i, data->A->p);

        retval = osqp_adjoint_derivative_get_mat(solver.get(), dP.get(), dA.get());

        mu_assert("Error getting matrix derivatives",
                  retval == OSQP_NO_ERROR);
    }
}


TEST_CASE_METHOD(derivative_adjoint_test_fixture, "Adjoint derivative: Not setup", "[derivative],[adjoint]")
{
    OSQPInt exitflag;

    // Run the derivative computation
    OSQPInt retval = osqp_adjoint_derivative_compute(solver.get(), sols_data->dx_1, sols_data->dy_zeros);

    mu_assert("Error when workspace not initialized",
                retval == OSQP_WORKSPACE_NOT_INIT_ERROR);

    // Get the vectors
    std::unique_ptr<OSQPFloat[]> dq(new OSQPFloat[data->n]);
    std::unique_ptr<OSQPFloat[]> dl(new OSQPFloat[data->m]);
    std::unique_ptr<OSQPFloat[]> du(new OSQPFloat[data->m]);

    retval = osqp_adjoint_derivative_get_vec(solver.get(), dq.get(), dl.get(), du.get());

    mu_assert("Error when workspace not initialized",
              retval == OSQP_WORKSPACE_NOT_INIT_ERROR);

    // Get the matrices
    OSQPCscMatrix_ptr dP{(OSQPCscMatrix*) malloc(sizeof(OSQPCscMatrix))};
    OSQPCscMatrix_ptr dA{(OSQPCscMatrix*) malloc(sizeof(OSQPCscMatrix))};

    std::unique_ptr<OSQPFloat[]> dPx(new OSQPFloat[data->P->nzmax]);
    std::unique_ptr<OSQPFloat[]> dAx(new OSQPFloat[data->A->nzmax]);

    // The matrices here share the same row/column pointers and just differ by the data
    csc_set_data(dP.get(), data->n, data->n, data->P->nzmax, dPx.get(), data->P->i, data->P->p);
    csc_set_data(dA.get(), data->m, data->n, data->A->nzmax, dAx.get(), data->A->i, data->A->p);

    retval = osqp_adjoint_derivative_get_mat(solver.get(), dP.get(), dA.get());

    mu_assert("Error when workspace not initialized",
              retval == OSQP_WORKSPACE_NOT_INIT_ERROR);
}
