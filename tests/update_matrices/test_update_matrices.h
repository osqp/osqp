#include <stdio.h>
#include "osqp.h"
#include "osqp_tester.h"

#include "update_matrices/data.h"

#ifndef OSQP_ALGEBRA_CUDA

#ifdef __cplusplus
extern "C" {
#endif
  #include "kkt.h"
#ifdef __cplusplus
}
#endif

void test_form_KKT() {

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


void test_update() {
  OSQPInt i, nnzP, nnzA;
  update_matrices_sols_data *data;

  OSQPTestData* problem;
  OSQPSolver*   solver;
  OSQPSettings* settings;
  OSQPInt       exitflag;

  // Update matrix P
  OSQPInt* Px_new_idx;

  // Update matrix A
  OSQPInt* Ax_new_idx;

  // Load problem data
  data = generate_problem_update_matrices_sols_data();

  // Generate first problem data
  problem    = (OSQPTestData*) c_malloc(sizeof(OSQPTestData));
  problem->P = data->test_solve_Pu;
  problem->q = data->test_solve_q;
  problem->A = data->test_solve_A;
  problem->l = data->test_solve_l;
  problem->u = data->test_solve_u;
  problem->n = data->test_solve_Pu->n;
  problem->m = data->test_solve_A->m;


  // Define Solver settings as default
  // Problem settings
  settings = (OSQPSettings *)c_malloc(sizeof(OSQPSettings));
  osqp_set_default_settings(settings);
  settings->max_iter = 1000;
  settings->alpha    = 1.6;
  settings->verbose  = 1;
  settings->eps_abs  = 1e-05;
  settings->eps_rel  = 1e-05;

  // Setup solver
  exitflag = osqp_setup(&solver,problem->P,problem->q,
                      problem->A,problem->l,problem->u,
                      problem->m,problem->n, settings);

  // Setup correct
  mu_assert("Update matrices: original problem, setup error!", exitflag == 0);

  // Solve Problem
  osqp_solve(solver);

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


  // Update P
  nnzP       = data->test_solve_Pu_new->p[data->test_solve_Pu->n];
  Px_new_idx = (OSQPInt*) c_malloc(nnzP * sizeof(OSQPInt));

  // Generate indices going from beginning to end of P
  for (i = 0; i < nnzP; i++) {
    Px_new_idx[i] = i;
  }

  osqp_update_data_mat(solver,
                       data->test_solve_Pu_new->x, NULL, nnzP,
                       NULL, NULL, 0);

  // Solve Problem
  osqp_solve(solver);

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

  // Cleanup and setup solver
  osqp_cleanup(solver);
  exitflag = osqp_setup(&solver,problem->P,problem->q,
                      problem->A,problem->l,problem->u,
                      problem->m,problem->n, settings);


  // Update P (all indices)
  osqp_update_data_mat(solver,
                       data->test_solve_Pu_new->x, OSQP_NULL, nnzP,
                       NULL, NULL, 0);

  // Solve Problem
  osqp_solve(solver);

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

  // Cleanup and setup solver
  osqp_cleanup(solver);
  exitflag = osqp_setup(&solver,problem->P,problem->q,
                      problem->A,problem->l,problem->u,
                      problem->m,problem->n, settings);


  // Update A
  nnzA       = data->test_solve_A->p[data->test_solve_A->n];
  Ax_new_idx = (OSQPInt*) c_malloc(nnzA * sizeof(OSQPInt));

  // Generate indices going from beginning to end of A
  for (i = 0; i < nnzA; i++) {
    Ax_new_idx[i] = i;
  }

  osqp_update_data_mat(solver,
                       NULL, NULL, 0,
                       data->test_solve_A_new->x, Ax_new_idx, nnzA);

  // Solve Problem
  osqp_solve(solver);

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

  // Cleanup and setup solver
  osqp_cleanup(solver);
  exitflag = osqp_setup(&solver,problem->P,problem->q,
                      problem->A,problem->l,problem->u,
                      problem->m,problem->n, settings);


  // Update A (all indices)
  osqp_update_data_mat(solver,
                       NULL, NULL, 0,
                       data->test_solve_A_new->x, OSQP_NULL, nnzA);

  // Solve Problem
  osqp_solve(solver);

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


  // Cleanup and setup solver
  osqp_cleanup(solver);
  exitflag = osqp_setup(&solver,problem->P,problem->q,
                      problem->A,problem->l,problem->u,
                      problem->m,problem->n, settings);

  // Update P and A
  osqp_update_data_mat(solver,
                       data->test_solve_Pu_new->x, Px_new_idx, nnzP,
                       data->test_solve_A_new->x, Ax_new_idx, nnzA);

  // Solve Problem
  osqp_solve(solver);

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

  // Cleanup and setup solver
  osqp_cleanup(solver);
  exitflag = osqp_setup(&solver,problem->P,problem->q,
                      problem->A,problem->l,problem->u,
                      problem->m,problem->n, settings);


  // Update P and A (all indices)
  osqp_update_data_mat(solver,
                       data->test_solve_Pu_new->x, OSQP_NULL, nnzP,
                       data->test_solve_A_new->x, OSQP_NULL, nnzA);

  // Solve Problem
  osqp_solve(solver);

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


  // Cleanup problems
  osqp_cleanup(solver);
  clean_problem_update_matrices_sols_data(data);
  c_free(problem);
  c_free(settings);
  c_free(Ax_new_idx);
  c_free(Px_new_idx);
}

#ifdef OSQP_ALGEBRA_MKL
void test_update_pardiso() {
  OSQPInt i, nnzP, nnzA, exitflag;
  update_matrices_sols_data* data;
  OSQPTestData* problem;
  OSQPSolver*   solver;
  OSQPSettings* settings;

  // Update matrix P
  OSQPInt* Px_new_idx;

  // Update matrix A
  OSQPInt* Ax_new_idx;

  // Load problem data
  data = generate_problem_update_matrices_sols_data();

  // Generate first problem data
  problem    = (OSQPTestData*)c_malloc(sizeof(OSQPTestData));
  problem->P = data->test_solve_Pu;
  problem->q = data->test_solve_q;
  problem->A = data->test_solve_A;
  problem->l = data->test_solve_l;
  problem->u = data->test_solve_u;
  problem->n = data->test_solve_Pu->n;
  problem->m = data->test_solve_A->m;


  // Define Solver settings as default
  // Problem settings
  settings = (OSQPSettings *)c_malloc(sizeof(OSQPSettings));
  osqp_set_default_settings(settings);
  settings->max_iter      = 1000;
  settings->alpha         = 1.6;
  settings->verbose       = 1;
  settings->linsys_solver = OSQP_DIRECT_SOLVER;

  // Setup solver
  exitflag = osqp_setup(&solver,problem->P,problem->q,
                      problem->A,problem->l,problem->u,
                      problem->m,problem->n, settings);

  // Setup correct
  mu_assert("Update matrices: original problem, setup error!", exitflag == 0);

  // Solve Problem
  osqp_solve(solver);

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


  // Update P
  nnzP       = data->test_solve_Pu->p[data->test_solve_Pu->n];
  Px_new_idx = (OSQPInt*)c_malloc(nnzP * sizeof(OSQPInt)); // Generate indices going from
                                                           // beginning to end of P

  for (i = 0; i < nnzP; i++) {
    Px_new_idx[i] = i;
  }

  osqp_update_data_mat(solver,
                       data->test_solve_Pu_new->x, Px_new_idx, nnzP,
                       NULL, NULL, 0);

  // Solve Problem
  osqp_solve(solver);

  // Compare solver statuses
  mu_assert("Update matrices: problem with P updated, error in solver status!",
            solver->info->status_val == data->test_solve_P_new_status);

  // Compare primal solutions
  mu_assert("Update matrices: problem with P updated, error in primal solution!",
            vec_norm_inf_diff(solver->solution->x, data->test_solve_P_new_x,
                              data->n) < TESTS_TOL);

  // Compare dual solutions
  mu_assert("Update matrices: problem with P updated, error in dual solution!",
            vec_norm_inf_diff(solver->solution->y, data->test_solve_P_new_y,
                              data->m) < TESTS_TOL);


  // Update A
  nnzA       = data->test_solve_A->p[data->test_solve_A->n];
  Ax_new_idx = (OSQPInt*)c_malloc(nnzA * sizeof(OSQPInt)); // Generate indices going from
                                                           // beginning to end of P

  for (i = 0; i < nnzA; i++) {
    Ax_new_idx[i] = i;
  }

  // Cleanup and setup solver
  osqp_cleanup(solver);
  exitflag = osqp_setup(&solver,problem->P,problem->q,
                      problem->A,problem->l,problem->u,
                      problem->m,problem->n, settings);

  osqp_update_data_mat(solver,
                       NULL, NULL, 0,
                       data->test_solve_A_new->x, Ax_new_idx, nnzA);

  // Solve Problem
  osqp_solve(solver);

  // Compare solver statuses
  mu_assert("Update matrices: problem with A updated, error in solver status!",
            solver->info->status_val == data->test_solve_A_new_status);

  // Compare primal solutions
  mu_assert("Update matrices: problem with A updated, error in primal solution!",
            vec_norm_inf_diff(solver->solution->x, data->test_solve_A_new_x,
                              data->n) < TESTS_TOL);

  // Compare dual solutions
  mu_assert("Update matrices: problem with A updated, error in dual solution!",
            vec_norm_inf_diff(solver->solution->y, data->test_solve_A_new_y,
                              data->m) < TESTS_TOL);


  // Cleanup and setup solver
  osqp_cleanup(solver);
  exitflag = osqp_setup(&solver,problem->P,problem->q,
                      problem->A,problem->l,problem->u,
                      problem->m,problem->n, settings);

  osqp_update_data_mat(solver,
                       data->test_solve_Pu_new->x, Px_new_idx, nnzP,
                       data->test_solve_A_new->x, Ax_new_idx, nnzA);

  // Solve Problem
  osqp_solve(solver);

  // Compare solver statuses
  mu_assert(
    "Update matrices: problem with P and A updated, error in solver status!",
    solver->info->status_val == data->test_solve_P_A_new_status);

  // Compare primal solutions
  mu_assert(
    "Update matrices: problem with P and A updated, error in primal solution!",
    vec_norm_inf_diff(solver->solution->x, data->test_solve_P_A_new_x,
                      data->n) < TESTS_TOL);

  // Compare dual solutions
  mu_assert(
    "Update matrices: problem with P and A updated, error in dual solution!",
    vec_norm_inf_diff(solver->solution->y, data->test_solve_P_A_new_y,
                      data->m) < TESTS_TOL);


  // Cleanup problems
  osqp_cleanup(solver);
  clean_problem_update_matrices_sols_data(data);
  c_free(problem);
  c_free(settings);
  c_free(Ax_new_idx);
  c_free(Px_new_idx);
}
#endif
