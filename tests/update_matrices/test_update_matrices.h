#include <stdio.h>
#include "osqp.h"
#include "cs.h"
#include "util.h"
#include "minunit.h"
#include "kkt.h"
#include "lin_sys.h"


#include "update_matrices/data.h"


static const char* test_form_KKT() {
  update_matrices_sols_data *data;
  c_float sigma, *rho_vec, *rho_inv_vec;
  c_int   m, *PtoKKT, *AtoKKT, *Pdiag_idx, Pdiag_n;
  csc    *KKT;

  // Load problem data
  data = generate_problem_update_matrices_sols_data();

  // Define rho_vec and sigma to form KKT
  sigma       = data->test_form_KKT_sigma;
  m           = data->test_form_KKT_A->m;
  rho_vec     = (c_float*) c_calloc(m, sizeof(c_float));
  rho_inv_vec = (c_float*) c_calloc(m, sizeof(c_float));
  vec_add_scalar(rho_vec, data->test_form_KKT_rho, m);
  vec_ew_recipr(rho_vec, rho_inv_vec, m);

  // Allocate vectors of indices
  PtoKKT = (c_int*) c_malloc((data->test_form_KKT_Pu->p[data->test_form_KKT_Pu->n]) *
                    sizeof(c_int));
  AtoKKT = (c_int*) c_malloc((data->test_form_KKT_A->p[data->test_form_KKT_A->n]) *
                    sizeof(c_int));

  // Form KKT matrix storing the index vectors
  KKT = form_KKT(data->test_form_KKT_Pu,
                 data->test_form_KKT_A,
                 0,
                 sigma,
                 rho_inv_vec,
                 PtoKKT,
                 AtoKKT,
                 &Pdiag_idx,
                 &Pdiag_n,
                 OSQP_NULL);

  // Assert if KKT matrix is the same as predicted one
  mu_assert("Update matrices: error in forming KKT matrix!",
            is_eq_csc(KKT, data->test_form_KKT_KKTu, TESTS_TOL));

  // Update KKT matrix with new P and new A
  update_KKT_P(KKT, data->test_form_KKT_Pu_new, PtoKKT, sigma, Pdiag_idx,
               Pdiag_n);
  update_KKT_A(KKT, data->test_form_KKT_A_new, AtoKKT);


  // Assert if KKT matrix is the same as predicted one
  mu_assert("Update matrices: error in updating KKT matrix!",
            is_eq_csc(KKT, data->test_form_KKT_KKTu_new, TESTS_TOL));


  // Cleanup
  clean_problem_update_matrices_sols_data(data);
  c_free(Pdiag_idx);
  csc_spfree(KKT);
  c_free(rho_vec);
  c_free(rho_inv_vec);
  c_free(AtoKKT);
  c_free(PtoKKT);
  return 0;
}

static const char* test_update() {
  c_int i, nnzP, nnzA;
  update_matrices_sols_data *data;
  OSQPData *problem;
  OSQPWorkspace *work;
  OSQPSettings  *settings;
  c_int exitflag;

  // Update matrix P
  c_int *Px_new_idx;

  // Update matrix A
  c_int *Ax_new_idx;

  // Load problem data
  data = generate_problem_update_matrices_sols_data();

  // Generate first problem data
  problem    = (OSQPData*) c_malloc(sizeof(OSQPData));
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

  // Setup workspace
  exitflag = osqp_setup(&work, problem, settings);

  // Setup correct
  mu_assert("Update matrices: original problem, setup error!", exitflag == 0);

  // Solve Problem
  osqp_solve(work);

  // Compare solver statuses
  mu_assert("Update matrices: original problem, error in solver status!",
            work->info->status_val == data->test_solve_status);

  // Compare primal solutions
  mu_assert("Update matrices: original problem, error in primal solution!",
            vec_norm_inf_diff(work->solution->x, data->test_solve_x,
                              data->n) < TESTS_TOL);

  // Compare dual solutions
  mu_assert("Update matrices: original problem, error in dual solution!",
            vec_norm_inf_diff(work->solution->y, data->test_solve_y,
                              data->m) < TESTS_TOL);


  // Update P
  nnzP       = data->test_solve_Pu->p[data->test_solve_Pu->n];
  Px_new_idx = (c_int*) c_malloc(nnzP * sizeof(c_int));

  // Generate indices going from beginning to end of P
  for (i = 0; i < nnzP; i++) {
    Px_new_idx[i] = i;
  }

  osqp_update_P(work, data->test_solve_Pu_new->x, Px_new_idx, nnzP);

  // Solve Problem
  osqp_solve(work);

  // Compare solver statuses
  mu_assert("Update matrices: problem with updating P, error in solver status!",
            work->info->status_val == data->test_solve_P_new_status);

  // Compare primal solutions
  mu_assert("Update matrices: problem with updating P, error in primal solution!",
            vec_norm_inf_diff(work->solution->x, data->test_solve_P_new_x,
                              data->n) < TESTS_TOL);

  // Compare dual solutions
  mu_assert("Update matrices: problem with updating P, error in dual solution!",
            vec_norm_inf_diff(work->solution->y, data->test_solve_P_new_y,
                              data->m) < TESTS_TOL);

  // Cleanup and setup workspace
  osqp_cleanup(work);
  exitflag = osqp_setup(&work, problem, settings);


  // Update P (all indices)
  osqp_update_P(work, data->test_solve_Pu_new->x, OSQP_NULL, nnzP);

  // Solve Problem
  osqp_solve(work);

  // Compare solver statuses
  mu_assert("Update matrices: problem with updating P (all indices), error in solver status!",
            work->info->status_val == data->test_solve_P_new_status);

  // Compare primal solutions
  mu_assert("Update matrices: problem with updating P (all indices), error in primal solution!",
            vec_norm_inf_diff(work->solution->x, data->test_solve_P_new_x,
                              data->n) < TESTS_TOL);

  // Compare dual solutions
  mu_assert("Update matrices: problem with updating P (all indices), error in dual solution!",
            vec_norm_inf_diff(work->solution->y, data->test_solve_P_new_y,
                              data->m) < TESTS_TOL);

  // Cleanup and setup workspace
  osqp_cleanup(work);
  exitflag = osqp_setup(&work, problem, settings);


  // Update A
  nnzA       = data->test_solve_A->p[data->test_solve_A->n];
  Ax_new_idx = (c_int*) c_malloc(nnzA * sizeof(c_int));

  // Generate indices going from beginning to end of A
  for (i = 0; i < nnzA; i++) {
    Ax_new_idx[i] = i;
  }

  osqp_update_A(work, data->test_solve_A_new->x, Ax_new_idx, nnzA);

  // Solve Problem
  osqp_solve(work);

  // Compare solver statuses
  mu_assert("Update matrices: problem with updating A, error in solver status!",
            work->info->status_val == data->test_solve_A_new_status);

  // Compare primal solutions
  mu_assert("Update matrices: problem with updating A, error in primal solution!",
            vec_norm_inf_diff(work->solution->x, data->test_solve_A_new_x,
                              data->n) < TESTS_TOL);

  // Compare dual solutions
  mu_assert("Update matrices: problem with updating A, error in dual solution!",
            vec_norm_inf_diff(work->solution->y, data->test_solve_A_new_y,
                              data->m) < TESTS_TOL);

  // Cleanup and setup workspace
  osqp_cleanup(work);
  exitflag = osqp_setup(&work, problem, settings);


  // Update A (all indices)
  osqp_update_A(work, data->test_solve_A_new->x, OSQP_NULL, nnzA);

  // Solve Problem
  osqp_solve(work);

  // Compare solver statuses
  mu_assert("Update matrices: problem with updating A (all indices), error in solver status!",
            work->info->status_val == data->test_solve_A_new_status);

  // Compare primal solutions
  mu_assert("Update matrices: problem with updating A (all indices), error in primal solution!",
            vec_norm_inf_diff(work->solution->x, data->test_solve_A_new_x,
                              data->n) < TESTS_TOL);

  // Compare dual solutions
  mu_assert("Update matrices: problem with updating A (all indices), error in dual solution!",
            vec_norm_inf_diff(work->solution->y, data->test_solve_A_new_y,
                              data->m) < TESTS_TOL);


  // Cleanup and setup workspace
  osqp_cleanup(work);
  exitflag = osqp_setup(&work, problem, settings);

  // Update P and A
  osqp_update_P_A(work, data->test_solve_Pu_new->x, Px_new_idx, nnzP,
                  data->test_solve_A_new->x, Ax_new_idx, nnzA);

  // Solve Problem
  osqp_solve(work);

  // Compare solver statuses
  mu_assert(
    "Update matrices: problem with updating P and A, error in solver status!",
    work->info->status_val == data->test_solve_P_A_new_status);

  // Compare primal solutions
  mu_assert(
    "Update matrices: problem with updating P and A, error in primal solution!",
    vec_norm_inf_diff(work->solution->x, data->test_solve_P_A_new_x,
                      data->n) < TESTS_TOL);

  // Compare dual solutions
  mu_assert(
    "Update matrices: problem with updating P and A, error in dual solution!",
    vec_norm_inf_diff(work->solution->y, data->test_solve_P_A_new_y,
                      data->m) < TESTS_TOL);

  // Cleanup and setup workspace
  osqp_cleanup(work);
  exitflag = osqp_setup(&work, problem, settings);


  // Update P and A (all indices)
  osqp_update_P_A(work, data->test_solve_Pu_new->x, OSQP_NULL, nnzP,
                  data->test_solve_A_new->x, OSQP_NULL, nnzA);

  // Solve Problem
  osqp_solve(work);

  // Compare solver statuses
  mu_assert(
    "Update matrices: problem with updating P and A (all indices), error in solver status!",
    work->info->status_val == data->test_solve_P_A_new_status);

  // Compare primal solutions
  mu_assert(
    "Update matrices: problem with updating P and A (all indices), error in primal solution!",
    vec_norm_inf_diff(work->solution->x, data->test_solve_P_A_new_x,
                      data->n) < TESTS_TOL);

  // Compare dual solutions
  mu_assert(
    "Update matrices: problem with updating P and A (all indices), error in dual solution!",
    vec_norm_inf_diff(work->solution->y, data->test_solve_P_A_new_y,
                      data->m) < TESTS_TOL);


  // Cleanup problems
  osqp_cleanup(work);
  clean_problem_update_matrices_sols_data(data);
  c_free(problem);
  c_free(settings);
  c_free(Ax_new_idx);
  c_free(Px_new_idx);

  return 0;
}

#ifdef ENABLE_MKL_PARDISO
static char* test_update_pardiso() {
  c_int i, nnzP, nnzA, exitflag;
  update_matrices_sols_data *data;
  OSQPData *problem;
  OSQPWorkspace *work;
  OSQPSettings  *settings;

  // Update matrix P
  c_int *Px_new_idx;

  // Update matrix A
  c_int *Ax_new_idx;

  // Load problem data
  data = generate_problem_update_matrices_sols_data();

  // Generate first problem data
  problem    = c_malloc(sizeof(OSQPData));
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
  settings->linsys_solver = MKL_PARDISO_SOLVER;

  // Setup workspace
  exitflag = osqp_setup(&work, problem, settings);

  // Setup correct
  mu_assert("Update matrices: original problem, setup error!", exitflag == 0);

  // Solve Problem
  osqp_solve(work);

  // Compare solver statuses
  mu_assert("Update matrices: original problem, error in solver status!",
            work->info->status_val == data->test_solve_status);

  // Compare primal solutions
  mu_assert("Update matrices: original problem, error in primal solution!",
            vec_norm_inf_diff(work->solution->x, data->test_solve_x,
                              data->n) < TESTS_TOL);

  // Compare dual solutions
  mu_assert("Update matrices: original problem, error in dual solution!",
            vec_norm_inf_diff(work->solution->y, data->test_solve_y,
                              data->m) < TESTS_TOL);


  // Update P
  nnzP       = data->test_solve_Pu->p[data->test_solve_Pu->n];
  Px_new_idx = c_malloc(nnzP * sizeof(c_int)); // Generate indices going from
                                               // beginning to end of P

  for (i = 0; i < nnzP; i++) {
    Px_new_idx[i] = i;
  }

  osqp_update_P(work, data->test_solve_Pu_new->x, Px_new_idx, nnzP);

  // Solve Problem
  osqp_solve(work);

  // Compare solver statuses
  mu_assert("Update matrices: problem with P updated, error in solver status!",
            work->info->status_val == data->test_solve_P_new_status);

  // Compare primal solutions
  mu_assert("Update matrices: problem with P updated, error in primal solution!",
            vec_norm_inf_diff(work->solution->x, data->test_solve_P_new_x,
                              data->n) < TESTS_TOL);

  // Compare dual solutions
  mu_assert("Update matrices: problem with P updated, error in dual solution!",
            vec_norm_inf_diff(work->solution->y, data->test_solve_P_new_y,
                              data->m) < TESTS_TOL);


  // Update A
  nnzA       = data->test_solve_A->p[data->test_solve_A->n];
  Ax_new_idx = c_malloc(nnzA * sizeof(c_int)); // Generate indices going from
                                               // beginning to end of P

  for (i = 0; i < nnzA; i++) {
    Ax_new_idx[i] = i;
  }

  // Cleanup and setup workspace
  osqp_cleanup(work);
  exitflag = osqp_setup(&work, problem, settings);

  osqp_update_A(work, data->test_solve_A_new->x, Ax_new_idx, nnzA);

  // Solve Problem
  osqp_solve(work);

  // Compare solver statuses
  mu_assert("Update matrices: problem with A updated, error in solver status!",
            work->info->status_val == data->test_solve_A_new_status);

  // Compare primal solutions
  mu_assert("Update matrices: problem with A updated, error in primal solution!",
            vec_norm_inf_diff(work->solution->x, data->test_solve_A_new_x,
                              data->n) < TESTS_TOL);

  // Compare dual solutions
  mu_assert("Update matrices: problem with A updated, error in dual solution!",
            vec_norm_inf_diff(work->solution->y, data->test_solve_A_new_y,
                              data->m) < TESTS_TOL);


  // Cleanup and setup workspace
  osqp_cleanup(work);
  exitflag = osqp_setup(&work, problem, settings);

  osqp_update_P_A(work, data->test_solve_Pu_new->x, Px_new_idx, nnzP,
                  data->test_solve_A_new->x, Ax_new_idx, nnzA);

  // Solve Problem
  osqp_solve(work);

  // Compare solver statuses
  mu_assert(
    "Update matrices: problem with P and A updated, error in solver status!",
    work->info->status_val == data->test_solve_P_A_new_status);

  // Compare primal solutions
  mu_assert(
    "Update matrices: problem with P and A updated, error in primal solution!",
    vec_norm_inf_diff(work->solution->x, data->test_solve_P_A_new_x,
                      data->n) < TESTS_TOL);

  // Compare dual solutions
  mu_assert(
    "Update matrices: problem with P and A updated, error in dual solution!",
    vec_norm_inf_diff(work->solution->y, data->test_solve_P_A_new_y,
                      data->m) < TESTS_TOL);


  // Cleanup problems
  osqp_cleanup(work);
  clean_problem_update_matrices_sols_data(data);
  c_free(problem);
  c_free(settings);
  c_free(Ax_new_idx);
  c_free(Px_new_idx);

  return 0;
}
#endif

static const char* test_update_matrices()
{
  mu_run_test(test_form_KKT);
  mu_run_test(test_update);

#ifdef ENABLE_MKL_PARDISO
  mu_run_test(test_update_pardiso);
#endif

  return 0;
}
