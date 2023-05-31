#include <catch2/catch.hpp>

#include "osqp_api.h"    /* OSQP API wrapper (public + some private) */
#include "osqp_tester.h" /* Tester helpers */
#include "test_utils.h"  /* Testing Helper functions */

#include "update_matrices_data.h"

#ifndef OSQP_ALGEBRA_CUDA

#include "kkt.h"

TEST_CASE("Test updating KKT matrix", "[kkt],[update]")
{

  update_matrices_sols_data* data;

  OSQPFloat      sigma;
  OSQPFloat*     rho_inv_vec_val;
  OSQPVectorf*   rho_vec;
  OSQPVectorf*   rho_inv_vec;
  OSQPInt        m;
  OSQPInt*       PtoKKT;
  OSQPInt*       AtoKKT;
  OSQPCscMatrix* KKT;

  // Load problem data
  data = generate_problem_update_matrices_sols_data();

  // Define rho_vec and sigma to form KKT
  sigma       = data->test_form_KKT_sigma;
  m           = data->test_form_KKT_A->m;
  rho_vec     = OSQPVectorf_malloc(m);
  rho_inv_vec = OSQPVectorf_malloc(m);
  OSQPVectorf_set_scalar(rho_vec, data->test_form_KKT_rho);
  OSQPVectorf_ew_reciprocal(rho_inv_vec,rho_vec);

  // Copy value of rho_inv_vec to a bare array
  rho_inv_vec_val = (OSQPFloat *) c_malloc(m * sizeof(OSQPFloat));
  OSQPVectorf_to_raw(rho_inv_vec_val, rho_inv_vec);

  // Allocate vectors of indices
  PtoKKT = (OSQPInt*) c_malloc((data->test_form_KKT_Pu->p[data->test_form_KKT_Pu->n]) *
                               sizeof(OSQPInt));
  AtoKKT = (OSQPInt*) c_malloc((data->test_form_KKT_A->p[data->test_form_KKT_A->n]) *
                               sizeof(OSQPInt));

  // Form KKT matrix storing the index vectors
  KKT = form_KKT(data->test_form_KKT_Pu,
                 data->test_form_KKT_A,
                 0,                     //CSC format
                 sigma,
                 rho_inv_vec_val,
                 1.0,                   // dummy
                 PtoKKT,
                 AtoKKT,
                 OSQP_NULL);

  // Assert if KKT matrix is the same as predicted one
  mu_assert("Update matrices: error in forming KKT matrix!",
            csc_is_eq(KKT, data->test_form_KKT_KKTu, TESTS_TOL));

  // Update KKT matrix with new P and new A
  update_KKT_A(KKT,
              data->test_form_KKT_A_new,
              data->test_form_KKT_A_new_idx,
              data->test_form_KKT_A_new_n,
              AtoKKT);

  update_KKT_P(KKT,
               data->test_form_KKT_Pu_new,
               data->test_form_KKT_Pu_new_idx,
               data->test_form_KKT_Pu_new_n,
               PtoKKT, sigma, 0);

  // Assert if KKT matrix is the same as predicted one
  mu_assert("Update matrices: error in updating KKT matrix!",
            csc_is_eq(KKT, data->test_form_KKT_KKTu_new, TESTS_TOL));


  // Cleanup
  clean_problem_update_matrices_sols_data(data);
  c_free(rho_inv_vec_val);
  csc_spfree(KKT);
  OSQPVectorf_free(rho_vec);
  OSQPVectorf_free(rho_inv_vec);
  c_free(AtoKKT);
  c_free(PtoKKT);
}

#endif /* ifndef OSQP_ALGEBRA_CUDA */


TEST_CASE_METHOD(OSQPTestFixture, "Test updating P and A", "[update]")
{
  OSQPInt exitflag;

  // Populate data
  update_matrices_sols_data_ptr data{generate_problem_update_matrices_sols_data()};

  OSQPInt nnzP = data->test_solve_Pu_new->p[data->test_solve_Pu->n];
  OSQPInt nnzA = data->test_solve_A->p[data->test_solve_A->n];

  // Define Solver settings
  settings->max_iter = 1000;

  /* Test all possible linear system solvers in this test case */
  settings->linsys_solver = GENERATE(filter(&isLinsysSupported, values({OSQP_DIRECT_SOLVER, OSQP_INDIRECT_SOLVER})));

  CAPTURE(settings->linsys_solver);

  // Setup solver
  exitflag = osqp_setup(&tmpSolver, data->test_solve_Pu, data->test_solve_q,
                        data->test_solve_A, data->test_solve_l, data->test_solve_u,
                        data->test_solve_A->m, data->test_solve_Pu->n, settings.get());
  solver.reset(tmpSolver);

  // Setup correct
  mu_assert("Update matrices: original problem, setup error!", exitflag == 0);

  // Solve Problem
  osqp_solve(solver.get());

  // Compare solver statuses
  mu_assert("Update matrices: original problem, error in solver status!",
            solver->info->status_val == data->test_solve_status);

  // Compare primal solutions
  mu_assert("Update matrices: original problem, error in primal solution!",
            vec_norm_inf_diff(solver->solution->x, data->test_solve_x,
                              data->n) < TESTS_TOL);

  // Compare dual solutions
  mu_assert("Update matrices: original problem, error in dual solution!",
            vec_norm_inf_diff(solver->solution->y, data->test_solve_y,
                              data->m) < TESTS_TOL);

  SECTION( "Matrix Updates: Update P (vector of indices)" ) {
    std::unique_ptr<OSQPInt[]> Px_new_idx(new OSQPInt[nnzP]);

    // Generate indices going from beginning to end of P
    for (OSQPInt i = 0; i < nnzP; i++) {
      Px_new_idx[i] = i;
    }

    osqp_update_data_mat(solver.get(),
                         data->test_solve_Pu_new->x, Px_new_idx.get(), nnzP,
                         NULL, NULL, 0);

    // Solve Problem
    osqp_solve(solver.get());

    // Compare solver statuses
    mu_assert("Update matrices: problem with updating P, error in solver status!",
              solver->info->status_val == data->test_solve_P_new_status);

    // Compare primal solutions
    mu_assert("Update matrices: problem with updating P, error in primal solution!",
              vec_norm_inf_diff(solver->solution->x, data->test_solve_P_new_x,
                                data->n) < TESTS_TOL);

    // Compare dual solutions
    mu_assert("Update matrices: problem with updating P, error in dual solution!",
              vec_norm_inf_diff(solver->solution->y, data->test_solve_P_new_y,
                                data->m) < TESTS_TOL);
  }

  SECTION( "Matrix Updates: Update P (all indices)" ) {
    osqp_update_data_mat(solver.get(),
                         data->test_solve_Pu_new->x, OSQP_NULL, nnzP,
                         NULL, NULL, 0);

    // Solve Problem
    osqp_solve(solver.get());

    // Compare solver statuses
    mu_assert("Update matrices: problem with updating P (all indices), error in solver status!",
              solver->info->status_val == data->test_solve_P_new_status);

    // Compare primal solutions
    mu_assert("Update matrices: problem with updating P (all indices), error in primal solution!",
              vec_norm_inf_diff(solver->solution->x, data->test_solve_P_new_x,
                                data->n) < TESTS_TOL);

    // Compare dual solutions
    mu_assert("Update matrices: problem with updating P (all indices), error in dual solution!",
              vec_norm_inf_diff(solver->solution->y, data->test_solve_P_new_y,
                                data->m) < TESTS_TOL);
  }

  SECTION( "Matrix Updates: Update A (vector of indices)" ) {
    std::unique_ptr<OSQPInt[]> Ax_new_idx(new OSQPInt[nnzA]);

    // Generate indices going from beginning to end of A
    for (OSQPInt i = 0; i < nnzA; i++) {
      Ax_new_idx[i] = i;
    }

    osqp_update_data_mat(solver.get(),
                         NULL, NULL, 0,
                         data->test_solve_A_new->x, Ax_new_idx.get(), nnzA);

    // Solve Problem
    osqp_solve(solver.get());

    // Compare solver statuses
    mu_assert("Update matrices: problem with updating A, error in solver status!",
              solver->info->status_val == data->test_solve_A_new_status);

    // Compare primal solutions
    mu_assert("Update matrices: problem with updating A, error in primal solution!",
              vec_norm_inf_diff(solver->solution->x, data->test_solve_A_new_x,
                                data->n) < TESTS_TOL);

    // Compare dual solutions
    mu_assert("Update matrices: problem with updating A, error in dual solution!",
              vec_norm_inf_diff(solver->solution->y, data->test_solve_A_new_y,
                                data->m) < TESTS_TOL);
  }

  SECTION( "Matrix Updates: Update A (All indices)" ) {
    // Update A (all indices)
    osqp_update_data_mat(solver.get(),
                         NULL, NULL, 0,
                         data->test_solve_A_new->x, OSQP_NULL, nnzA);

    // Solve Problem
    osqp_solve(solver.get());

    // Compare solver statuses
    mu_assert("Update matrices: problem with updating A (all indices), error in solver status!",
              solver->info->status_val == data->test_solve_A_new_status);

    // Compare primal solutions
    mu_assert("Update matrices: problem with updating A (all indices), error in primal solution!",
              vec_norm_inf_diff(solver->solution->x, data->test_solve_A_new_x,
                                data->n) < TESTS_TOL);

    // Compare dual solutions
    mu_assert("Update matrices: problem with updating A (all indices), error in dual solution!",
              vec_norm_inf_diff(solver->solution->y, data->test_solve_A_new_y,
                                data->m) < TESTS_TOL);
  }

  SECTION( "Matrix Updates: Update P and A (specified indices)" ) {
    std::unique_ptr<OSQPInt[]> Px_new_idx(new OSQPInt[nnzP]);
    std::unique_ptr<OSQPInt[]> Ax_new_idx(new OSQPInt[nnzA]);

    // Generate indices going from beginning to end of P
    for (OSQPInt i = 0; i < nnzP; i++) {
      Px_new_idx[i] = i;
    }

    // Generate indices going from beginning to end of A
    for (OSQPInt i = 0; i < nnzA; i++) {
      Ax_new_idx[i] = i;
    }

    // Update P and A
    osqp_update_data_mat(solver.get(),
                         data->test_solve_Pu_new->x, Px_new_idx.get(), nnzP,
                         data->test_solve_A_new->x, Ax_new_idx.get(), nnzA);

    // Solve Problem
    osqp_solve(solver.get());

    // Compare solver statuses
    mu_assert(
      "Update matrices: problem with updating P and A, error in solver status!",
      solver->info->status_val == data->test_solve_P_A_new_status);

    // Compare primal solutions
    mu_assert(
      "Update matrices: problem with updating P and A, error in primal solution!",
      vec_norm_inf_diff(solver->solution->x, data->test_solve_P_A_new_x,
                        data->n) < TESTS_TOL);

    // Compare dual solutions
    mu_assert(
      "Update matrices: problem with updating P and A, error in dual solution!",
      vec_norm_inf_diff(solver->solution->y, data->test_solve_P_A_new_y,
                        data->m) < TESTS_TOL);
  }

  SECTION( "Matrix Updates: Update P and A (all indicies" ) {
    // Update P and A (all indices)
    osqp_update_data_mat(solver.get(),
                         data->test_solve_Pu_new->x, OSQP_NULL, nnzP,
                         data->test_solve_A_new->x, OSQP_NULL, nnzA);

    // Solve Problem
    osqp_solve(solver.get());

    // Compare solver statuses
    mu_assert(
      "Update matrices: problem with updating P and A (all indices), error in solver status!",
      solver->info->status_val == data->test_solve_P_A_new_status);

    // Compare primal solutions
    mu_assert(
      "Update matrices: problem with updating P and A (all indices), error in primal solution!",
      vec_norm_inf_diff(solver->solution->x, data->test_solve_P_A_new_x,
                        data->n) < TESTS_TOL);

    // Compare dual solutions
    mu_assert(
      "Update matrices: problem with updating P and A (all indices), error in dual solution!",
      vec_norm_inf_diff(solver->solution->y, data->test_solve_P_A_new_y,
                        data->m) < TESTS_TOL);
  }
}
