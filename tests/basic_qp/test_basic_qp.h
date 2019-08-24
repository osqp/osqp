#include "osqp.h"    // OSQP API
#include "auxil.h"   // Needed for cold_start()
#include "cs.h"      // CSC data structure
#include "util.h"    // Utilities for testing
#include "minunit.h" // Basic testing script header

#include "basic_qp/data.h"


static const char* test_basic_qp_solve()
{
  c_int exitflag, tmp_int;
  c_float tmp_float;
  csc *tmp_mat, *P_tmp;

  // Problem settings
  OSQPSettings *settings = (OSQPSettings *)c_malloc(sizeof(OSQPSettings));

  // Structures
  OSQPWorkspace *work; // Workspace
  OSQPData *data;      // Data
  basic_qp_sols_data *sols_data;

  // Populate data
  data = generate_problem_basic_qp();
  sols_data = generate_problem_basic_qp_sols_data();

  // Define Solver settings as default
  osqp_set_default_settings(settings);
  settings->max_iter   = 2000;
  settings->alpha      = 1.6;
  settings->polish     = 1;
  settings->scaling    = 0;
  settings->verbose    = 1;
  settings->warm_start = 0;

  // Setup workspace
  exitflag = osqp_setup(&work, data, settings);

  // Setup correct
  mu_assert("Basic QP test solve: Setup error!", exitflag == 0);


  // Solve Problem
  osqp_solve(work);

  // Compare solver statuses
  mu_assert("Basic QP test solve: Error in solver status!",
	    work->info->status_val == sols_data->status_test);

  // Compare primal solutions
  mu_assert("Basic QP test solve: Error in primal solution!",
	    vec_norm_inf_diff(work->solution->x, sols_data->x_test,
			      data->n) < TESTS_TOL);

  // Compare dual solutions
  mu_assert("Basic QP test solve: Error in dual solution!",
	    vec_norm_inf_diff(work->solution->y, sols_data->y_test,
			      data->m) < TESTS_TOL);


  // Compare objective values
  mu_assert("Basic QP test solve: Error in objective value!",
	    c_absval(work->info->obj_val - sols_data->obj_value_test) <
	    TESTS_TOL);

  // Try to set wrong settings
  mu_assert("Basic QP test solve: Wrong value of rho not caught!",
	    osqp_update_rho(work, -0.1) == 1);

  mu_assert("Basic QP test solve: Wrong value of max_iter not caught!",
	    osqp_update_max_iter(work, -1) == 1);

  mu_assert("Basic QP test solve: Wrong value of eps_abs not caught!",
	    osqp_update_eps_abs(work, -1.) == 1);

  mu_assert("Basic QP test solve: Wrong value of eps_rel not caught!",
	    osqp_update_eps_rel(work, -1.) == 1);

  mu_assert("Basic QP test solve: Wrong value of eps_prim_inf not caught!",
	    osqp_update_eps_prim_inf(work, -0.1) == 1);

  mu_assert("Basic QP test solve: Wrong value of eps_dual_inf not caught!",
	    osqp_update_eps_dual_inf(work, -0.1) == 1);

  mu_assert("Basic QP test solve: Wrong value of alpha not caught!",
	    osqp_update_alpha(work, 2.0) == 1);

  mu_assert("Basic QP test solve: Wrong value of warm_start not caught!",
	    osqp_update_warm_start(work, -1) == 1);

  mu_assert("Basic QP test solve: Wrong value of scaled_termination not caught!",
	    osqp_update_scaled_termination(work, 2) == 1);

  mu_assert("Basic QP test solve: Wrong value of check_termination not caught!",
	    osqp_update_check_termination(work, -1) == 1);

  mu_assert("Basic QP test solve: Wrong value of delta not caught!",
	    osqp_update_delta(work, 0.) == 1);

  mu_assert("Basic QP test solve: Wrong value of polish not caught!",
	    osqp_update_polish(work, 2) == 1);

  mu_assert("Basic QP test solve: Wrong value of polish_refine_iter not caught!",
	    osqp_update_polish_refine_iter(work, -1) == 1);

  mu_assert("Basic QP test solve: Wrong value of verbose not caught!",
	    osqp_update_verbose(work, 2) == 1);

  // Clean workspace
  osqp_cleanup(work);

  /* =============================
       SETUP WITH WRONG SETTINGS
     ============================= */

  // Setup workspace with empty settings
  exitflag = osqp_setup(&work, data, OSQP_NULL);
  mu_assert("Basic QP test solve: Setup should result in error due to empty settings",
            exitflag == OSQP_SETTINGS_VALIDATION_ERROR);

  // Setup workspace with a wrong number of scaling iterations
  tmp_int = settings->scaling;
  settings->scaling = -1;
  exitflag = osqp_setup(&work, data, settings);
  mu_assert("Basic QP test solve: Setup should result in error due to a negative number of scaling iterations",
            exitflag == OSQP_SETTINGS_VALIDATION_ERROR);
  settings->scaling = tmp_int;

  // Setup workspace with wrong settings->adaptive_rho
  tmp_int = settings->adaptive_rho;
  settings->adaptive_rho = 2;
  exitflag = osqp_setup(&work, data, settings);
  mu_assert("Basic QP test solve: Setup should result in error due to non-boolean settings->adaptive_rho",
            exitflag == OSQP_SETTINGS_VALIDATION_ERROR);
  settings->adaptive_rho = tmp_int;

  // Setup workspace with wrong settings->adaptive_rho_interval
  tmp_int = settings->adaptive_rho_interval;
  settings->adaptive_rho_interval = -1;
  exitflag = osqp_setup(&work, data, settings);
  mu_assert("Basic QP test solve: Setup should result in error due to negative settings->adaptive_rho_interval",
            exitflag == OSQP_SETTINGS_VALIDATION_ERROR);
  settings->adaptive_rho_interval = tmp_int;

#ifdef PROFILING
  // Setup workspace with wrong settings->adaptive_rho_fraction
  tmp_float = settings->adaptive_rho_fraction;
  settings->adaptive_rho_fraction = -1.5;
  exitflag = osqp_setup(&work, data, settings);
  mu_assert("Basic QP test solve: Setup should result in error due to non-positive settings->adaptive_rho_fraction",
            exitflag == OSQP_SETTINGS_VALIDATION_ERROR);
  settings->adaptive_rho_fraction = tmp_float;
#endif

  // Setup workspace with wrong settings->adaptive_rho_tolerance
  tmp_float = settings->adaptive_rho_tolerance;
  settings->adaptive_rho_tolerance = 0.5;
  exitflag = osqp_setup(&work, data, settings);
  mu_assert("Basic QP test solve: Setup should result in error due to wrong settings->adaptive_rho_tolerance",
            exitflag == OSQP_SETTINGS_VALIDATION_ERROR);
  settings->adaptive_rho_tolerance = tmp_float;

  // Setup workspace with wrong settings->polish_refine_iter
  tmp_int = settings->polish_refine_iter;
  settings->polish_refine_iter = -3;
  exitflag = osqp_setup(&work, data, settings);
  mu_assert("Basic QP test solve: Setup should result in error due to negative settings->polish_refine_iter",
            exitflag == OSQP_SETTINGS_VALIDATION_ERROR);
  settings->polish_refine_iter = tmp_int;

  // Setup workspace with wrong settings->rho
  tmp_float = settings->rho;
  settings->rho = 0.0;
  exitflag = osqp_setup(&work, data, settings);
  mu_assert("Basic QP test solve: Setup should result in error due to non-positive settings->rho",
            exitflag == OSQP_SETTINGS_VALIDATION_ERROR);
  settings->rho = tmp_float;

  // Setup workspace with wrong settings->sigma
  tmp_float = settings->sigma;
  settings->sigma = -0.1;
  exitflag = osqp_setup(&work, data, settings);
  mu_assert("Basic QP test solve: Setup should result in error due to non-positive settings->sigma",
            exitflag == OSQP_SETTINGS_VALIDATION_ERROR);
  settings->sigma = tmp_float;

  // Setup workspace with wrong settings->delta
  tmp_float = settings->delta;
  settings->delta = -1.1;
  exitflag = osqp_setup(&work, data, settings);
  mu_assert("Basic QP test solve: Setup should result in error due to non-positive settings->delta",
            exitflag == OSQP_SETTINGS_VALIDATION_ERROR);
  settings->delta = tmp_float;

  // Setup workspace with wrong settings->max_iter
  tmp_int = settings->max_iter;
  settings->max_iter = 0;
  exitflag = osqp_setup(&work, data, settings);
  mu_assert("Basic QP test solve: Setup should result in error due to non-positive settings->max_iter",
            exitflag == OSQP_SETTINGS_VALIDATION_ERROR);
  settings->max_iter = tmp_int;

  // Setup workspace with wrong settings->eps_abs
  tmp_float = settings->eps_abs;
  settings->eps_abs = -1.1;
  exitflag = osqp_setup(&work, data, settings);
  mu_assert("Basic QP test solve: Setup should result in error due to negative settings->eps_abs",
            exitflag == OSQP_SETTINGS_VALIDATION_ERROR);
  settings->eps_abs = tmp_float;

  // Setup workspace with wrong settings->eps_rel
  tmp_float = settings->eps_rel;
  settings->eps_rel = -0.1;
  exitflag = osqp_setup(&work, data, settings);
  mu_assert("Basic QP test solve: Setup should result in error due to negative settings->eps_rel",
            exitflag == OSQP_SETTINGS_VALIDATION_ERROR);
  settings->eps_rel = tmp_float;

  // Setup workspace with wrong settings->eps_prim_inf
  tmp_float = settings->eps_prim_inf;
  settings->eps_prim_inf = -0.1;
  exitflag = osqp_setup(&work, data, settings);
  mu_assert("Basic QP test solve: Setup should result in error due to non-positive settings->eps_prim_inf",
            exitflag == OSQP_SETTINGS_VALIDATION_ERROR);
  settings->eps_prim_inf = tmp_float;

  // Setup workspace with wrong settings->eps_dual_inf
  tmp_float = settings->eps_dual_inf;
  settings->eps_dual_inf = 0.0;
  exitflag = osqp_setup(&work, data, settings);
  mu_assert("Basic QP test solve: Setup should result in error due to non-positive settings->eps_dual_inf",
            exitflag == OSQP_SETTINGS_VALIDATION_ERROR);
  settings->eps_dual_inf = tmp_float;

  // Setup workspace with wrong settings->alpha
  tmp_float = settings->alpha;
  settings->alpha = 2.0;
  exitflag = osqp_setup(&work, data, settings);
  mu_assert("Basic QP test solve: Setup should result in error due to wrong settings->alpha",
            exitflag == OSQP_SETTINGS_VALIDATION_ERROR);
  settings->alpha = tmp_float;

  // Setup workspace with wrong settings->linsys_solver
  tmp_int = settings->linsys_solver;
  settings->linsys_solver = 5;
  exitflag = osqp_setup(&work, data, settings);
  mu_assert("Basic QP test solve: Setup should result in error due to wrong settings->linsys_solver",
            exitflag == OSQP_SETTINGS_VALIDATION_ERROR);
  settings->linsys_solver = tmp_int;

  // Setup workspace with wrong settings->verbose
  tmp_int = settings->verbose;
  settings->verbose = 2;
  exitflag = osqp_setup(&work, data, settings);
  mu_assert("Basic QP test solve: Setup should result in error due to non-boolean settings->verbose",
            exitflag == OSQP_SETTINGS_VALIDATION_ERROR);
  settings->verbose = tmp_int;

  // Setup workspace with wrong settings->scaled_termination
  tmp_int = settings->scaled_termination;
  settings->scaled_termination = 2;
  exitflag = osqp_setup(&work, data, settings);
  mu_assert("Basic QP test solve: Setup should result in error due to non-boolean settings->scaled_termination",
            exitflag == OSQP_SETTINGS_VALIDATION_ERROR);
  settings->scaled_termination = tmp_int;

  // Setup workspace with wrong settings->check_termination
  tmp_int = settings->check_termination;
  settings->check_termination = -1;
  exitflag = osqp_setup(&work, data, settings);
  mu_assert("Basic QP test solve: Setup should result in error due to non-boolean settings->check_termination",
            exitflag == OSQP_SETTINGS_VALIDATION_ERROR);
  settings->check_termination = tmp_int;

  // Setup workspace with wrong settings->warm_start
  tmp_int = settings->warm_start;
  settings->warm_start = 5;
  exitflag = osqp_setup(&work, data, settings);
  mu_assert("Basic QP test solve: Setup should result in error due to non-boolean settings->warm_start",
            exitflag == OSQP_SETTINGS_VALIDATION_ERROR);
  settings->warm_start = tmp_int;

#ifdef PROFILING
  // Setup workspace with wrong settings->time_limit
  tmp_float = settings->time_limit;
  settings->time_limit = -0.2;
  exitflag = osqp_setup(&work, data, settings);
  mu_assert("Basic QP test solve: Setup should result in error due to wrong settings->time_limit",
            exitflag == OSQP_SETTINGS_VALIDATION_ERROR);
  settings->time_limit = tmp_float;
#endif


  /* =========================
       SETUP WITH WRONG DATA
     ========================= */

  // Setup workspace with empty data
  exitflag = osqp_setup(&work, OSQP_NULL, settings);
  mu_assert("Basic QP test solve: Setup should result in error due to empty data",
            exitflag == OSQP_DATA_VALIDATION_ERROR);

  // Setup workspace with wrong data->m
  tmp_int = data->m;
  data->m = data->m - 1;
  exitflag = osqp_setup(&work, data, settings);
  mu_assert("Basic QP test solve: Setup should result in error due to wrong data->m",
            exitflag == OSQP_DATA_VALIDATION_ERROR);
  data->m = tmp_int;

  // Setup workspace with wrong data->n
  tmp_int = data->n;
  data->n = data->n + 1;
  exitflag = osqp_setup(&work, data, settings);
  mu_assert("Basic QP test solve: Setup should result in error due to wrong data->n",
            exitflag == OSQP_DATA_VALIDATION_ERROR);

  // Setup workspace with zero data->n
  data->n = 0;
  exitflag = osqp_setup(&work, data, settings);
  mu_assert("Basic QP test solve: Setup should result in error due to zero data->n",
            exitflag == OSQP_DATA_VALIDATION_ERROR);
  data->n = tmp_int;

  // Setup workspace with wrong P->m
  tmp_int = data->P->m;
  data->P->m = data->n + 1;
  exitflag = osqp_setup(&work, data, settings);
  mu_assert("Basic QP test solve: Setup should result in error due to wrong P->m",
            exitflag == OSQP_DATA_VALIDATION_ERROR);
  data->P->m = tmp_int;

  // Setup workspace with wrong P->n
  tmp_int = data->P->n;
  data->P->n = data->n + 1;
  exitflag = osqp_setup(&work, data, settings);
  mu_assert("Basic QP test solve: Setup should result in error due to wrong P->n",
            exitflag == OSQP_DATA_VALIDATION_ERROR);
  data->P->n = tmp_int;

  // Setup workspace with non-upper-triangular P
  tmp_mat = data->P;

  // Construct non-upper-triangular P
  P_tmp = (csc*) c_malloc(sizeof(csc));
  P_tmp->m = 2;
  P_tmp->n = 2;
  P_tmp->nz = -1;
  P_tmp->nzmax = 4;
  P_tmp->x = (c_float*) c_malloc(4 * sizeof(c_float));
  P_tmp->x[0] = 4.0;
  P_tmp->x[1] = 1.0;
  P_tmp->x[2] = 1.0;
  P_tmp->x[3] = 2.0;
  P_tmp->i = (c_int*) c_malloc(4 * sizeof(c_int));
  P_tmp->i[0] = 0;
  P_tmp->i[1] = 1;
  P_tmp->i[2] = 0;
  P_tmp->i[3] = 1;
  P_tmp->p = (c_int*) c_malloc((2 + 1) * sizeof(c_int));
  P_tmp->p[0] = 0;
  P_tmp->p[1] = 2;
  P_tmp->p[2] = 4;

  data->P = P_tmp;
  exitflag = osqp_setup(&work, data, settings);
  mu_assert("Basic QP test solve: Setup should result in error due to non-triu structure of P",
            exitflag == OSQP_DATA_VALIDATION_ERROR);
  data->P = tmp_mat;

  // Setup workspace with non-consistent bounds
  data->l[0] = data->u[0] + 1.0;
  exitflag = osqp_setup(&work, data, settings);
  mu_assert("Basic QP test solve: Setup should result in error due to non-consistent bounds",
            exitflag == OSQP_DATA_VALIDATION_ERROR);


  // Cleanup data
  clean_problem_basic_qp(data);
  clean_problem_basic_qp_sols_data(sols_data);

  // Cleanup
  c_free(settings);
  c_free(P_tmp->x);
  c_free(P_tmp->i);
  c_free(P_tmp->p);
  c_free(P_tmp);

  return 0;
}

#ifdef ENABLE_MKL_PARDISO
static char* test_basic_qp_solve_pardiso()
{
  c_int exitflag;

  // Problem settings
  OSQPSettings *settings = (OSQPSettings *)c_malloc(sizeof(OSQPSettings));

  // Structures
  OSQPWorkspace *work; // Workspace
  OSQPData *data;      // Data
  basic_qp_sols_data *sols_data;

  // Populate data
  data = generate_problem_basic_qp();
  sols_data = generate_problem_basic_qp_sols_data();


  // Define Solver settings as default
  osqp_set_default_settings(settings);
  settings->max_iter      = 2000;
  settings->alpha         = 1.6;
  settings->polish        = 1;
  settings->scaling       = 0;
  settings->verbose       = 1;
  settings->warm_start    = 0;
  settings->linsys_solver = MKL_PARDISO_SOLVER;

  // Setup workspace
  exitflag = osqp_setup(&work, data, settings);

  // Setup correct
  mu_assert("Basic QP test solve Pardiso: Setup error!", exitflag == 0);

  // Solve Problem
  osqp_solve(work);

  // Compare solver statuses
  mu_assert("Basic QP test solve Pardiso: Error in solver status!",
            work->info->status_val == sols_data->status_test);

  // Compare primal solutions
  mu_assert("Basic QP test solve Pardiso: Error in primal solution!",
            vec_norm_inf_diff(work->solution->x, sols_data->x_test,
                              data->n) < TESTS_TOL);

  // Compare dual solutions
  mu_assert("Basic QP test solve Pardiso: Error in dual solution!",
            vec_norm_inf_diff(work->solution->y, sols_data->y_test,
                              data->m) < TESTS_TOL);


  // Compare objective values
  mu_assert("Basic QP test solve Pardiso: Error in objective value!",
            c_absval(work->info->obj_val - sols_data->obj_value_test) <
            TESTS_TOL);

  // Clean workspace
  osqp_cleanup(work);


  // Cleanup data
  clean_problem_basic_qp(data);
  clean_problem_basic_qp_sols_data(sols_data);

  // Cleanup
  c_free(settings);

  return 0;
}
#endif

static const char* test_basic_qp_update()
{
  c_int exitflag;

  // Problem settings
  OSQPSettings *settings = (OSQPSettings *)c_malloc(sizeof(OSQPSettings));

  // Structures
  OSQPWorkspace *work; // Workspace
  OSQPData *data;      // Data
  basic_qp_sols_data *sols_data;

  // Populate data
  data = generate_problem_basic_qp();
  sols_data = generate_problem_basic_qp_sols_data();


  // Define Solver settings as default
  osqp_set_default_settings(settings);
  settings->max_iter   = 200;
  settings->alpha      = 1.6;
  settings->polish     = 1;
  settings->scaling    = 0;
  settings->verbose    = 1;
  settings->warm_start = 0;

  // Setup workspace
  exitflag = osqp_setup(&work, data, settings);

  // Setup correct
  mu_assert("Basic QP test update: Setup error!", exitflag == 0);


  // ====================================================================
  //  Update data
  // ====================================================================

  // Update linear cost
  osqp_update_lin_cost(work, sols_data->q_new);
  mu_assert("Basic QP test update: Error in updating linear cost!",
            vec_norm_inf_diff(work->data->q, sols_data->q_new,
                              data->n) < TESTS_TOL);

  // UPDATE BOUND
  // Try to update with non-consistent values
  mu_assert("Basic QP test update: Error in bounds update ordering not caught!",
            osqp_update_bounds(work, sols_data->u_new, sols_data->l_new) == 1);

  // Now update with correct values
  mu_assert("Basic QP test update: Error in bounds update ordering!",
            osqp_update_bounds(work, sols_data->l_new, sols_data->u_new) == 0);

  mu_assert("Basic QP test update: Error in bounds update, lower bound!",
            vec_norm_inf_diff(work->data->l, sols_data->l_new,
                              data->m) < TESTS_TOL);

  mu_assert("Basic QP test update: Error in bounds update, upper bound!",
            vec_norm_inf_diff(work->data->u, sols_data->u_new,
                              data->m) < TESTS_TOL);

  // Return original values
  osqp_update_bounds(work, data->l, data->u);


  // UPDATE LOWER BOUND
  // Try to update with non-consistent values
  mu_assert(
    "Basic QP test update: Error in lower bound update ordering not caught!",
    osqp_update_lower_bound(work, sols_data->u_new) == 1);

  // Now update with correct values
  mu_assert("Basic QP test update: Error in lower bound update ordering!",
            osqp_update_lower_bound(work, sols_data->l_new) == 0);

  mu_assert("Basic QP test update: Error in updating lower bound!",
            vec_norm_inf_diff(work->data->l, sols_data->l_new,
                              data->m) < TESTS_TOL);

  // Return original values
  osqp_update_lower_bound(work, data->l);


  // UPDATE UPPER BOUND
  // Try to update with non-consistent values
  mu_assert(
    "Basic QP test update: Error in upper bound update: ordering not caught!",
    osqp_update_upper_bound(work, sols_data->l_new) == 1);

  // Now update with correct values
  mu_assert("Basic QP test update: Error in upper bound update: ordering!",
            osqp_update_upper_bound(work, sols_data->u_new) == 0);

  mu_assert("Basic QP test update: Error in updating upper bound!",
            vec_norm_inf_diff(work->data->u, sols_data->u_new,
                              data->m) < TESTS_TOL);


  // Clean workspace
  osqp_cleanup(work);


  // Cleanup data
  clean_problem_basic_qp(data);
  clean_problem_basic_qp_sols_data(sols_data);

  // Cleanup
  c_free(settings);

  return 0;
}

static const char* test_basic_qp_check_termination()
{
  c_int exitflag;

  // Problem settings
  OSQPSettings *settings = (OSQPSettings *)c_malloc(sizeof(OSQPSettings));

  // Structures
  OSQPWorkspace *work; // Workspace
  OSQPData *data;      // Data
  basic_qp_sols_data *sols_data;

  // Populate data
  data = generate_problem_basic_qp();
  sols_data = generate_problem_basic_qp_sols_data();


  // Define Solver settings as default
  osqp_set_default_settings(settings);
  settings->max_iter          = 200;
  settings->alpha             = 1.6;
  settings->polish            = 0;
  settings->scaling           = 0;
  settings->verbose           = 1;
  settings->check_termination = 0;
  settings->warm_start        = 0;

  // Setup workspace
  exitflag = osqp_setup(&work, data, settings);

  // Setup correct
  mu_assert("Basic QP test solve: Setup error!", exitflag == 0);

  // Solve Problem
  osqp_solve(work);

  // Check if iter == max_iter
  mu_assert(
    "Basic QP test check termination: Error in number of iterations taken!",
    work->info->iter == work->settings->max_iter);

  // Compare solver statuses
  mu_assert("Basic QP test check termination: Error in solver status!",
            work->info->status_val == sols_data->status_test);

  // Compare primal solutions
  mu_assert("Basic QP test check termination: Error in primal solution!",
            vec_norm_inf_diff(work->solution->x, sols_data->x_test,
                              data->n) < TESTS_TOL);

  // Compare dual solutions
  // print_vec(work->solution->y, data->m, "y_sol");
  // print_vec(sols_data->y_test, data->m, "y_test");
  mu_assert("Basic QP test check termination: Error in dual solution!",
            vec_norm_inf_diff(work->solution->y, sols_data->y_test,
                              data->m) < TESTS_TOL);

  // Compare objective values
  mu_assert("Basic QP test check termination: Error in objective value!",
            c_absval(work->info->obj_val - sols_data->obj_value_test) <
            TESTS_TOL);

  // Clean workspace
  osqp_cleanup(work);

  // Cleanup data
  clean_problem_basic_qp(data);
  clean_problem_basic_qp_sols_data(sols_data);

  // Cleanup
  c_free(settings);

  return 0;
}

static const char* test_basic_qp_update_rho()
{
  c_int extiflag;

  // Problem settings
  OSQPSettings *settings = (OSQPSettings *)c_malloc(sizeof(OSQPSettings));

  // Structures
  OSQPWorkspace *work; // Workspace
  OSQPData *data;      // Data
  basic_qp_sols_data *sols_data;

  // Exitflag
  c_int exitflag;

  // rho to use
  c_float rho;

  // Define number of iterations to compare
  c_int n_iter_new_solver, n_iter_update_rho;

  // Populate data
  data = generate_problem_basic_qp();
  sols_data = generate_problem_basic_qp_sols_data();


  // Define Solver settings as default
  rho = 0.7;
  osqp_set_default_settings(settings);
  settings->rho               = rho;
  settings->adaptive_rho      = 0; // Disable adaptive rho for this test
  settings->eps_abs           = 5e-05;
  settings->eps_rel           = 5e-05;
  settings->check_termination = 1;

  // Setup workspace
  exitflag = osqp_setup(&work, data, settings);

  // Setup correct
  mu_assert("Basic QP test update rho: Setup error!", exitflag == 0);

  // Solve Problem
  osqp_solve(work);

  // Store number of iterations
  n_iter_new_solver = work->info->iter;

  // Compare solver statuses
  mu_assert("Update rho test solve: Error in solver status!",
            work->info->status_val == sols_data->status_test);

  // Compare primal solutions
  mu_assert("Update rho test solve: Error in primal solution!",
            vec_norm_inf_diff(work->solution->x, sols_data->x_test,
                              data->n)/vec_norm_inf(sols_data->x_test, data->n) < TESTS_TOL);

  // Compare dual solutions
  mu_assert("Update rho test solve: Error in dual solution!",
            vec_norm_inf_diff(work->solution->y, sols_data->y_test,
                              data->m)/vec_norm_inf(sols_data->y_test, data->m) < TESTS_TOL);

  // Compare objective values
  mu_assert("Update rho test solve: Error in objective value!",
            c_absval(work->info->obj_val - sols_data->obj_value_test) <
            TESTS_TOL);

  // Clean workspace
  osqp_cleanup(work);


  // Create new problem with different rho and update it
  osqp_set_default_settings(settings);
  settings->rho               = 0.1;
  settings->adaptive_rho      = 0;
  settings->check_termination = 1;
  settings->eps_abs           = 5e-05;
  settings->eps_rel           = 5e-05;

  // Setup workspace
  exitflag = osqp_setup(&work, data, settings);

  // Setup correct
  mu_assert("Basic QP test update rho: Setup error!", exitflag == 0);

  // Update rho
  exitflag = osqp_update_rho(work, rho);
  mu_assert("Basic QP test update rho: Error update rho!", exitflag == 0);

  // Solve Problem
  osqp_solve(work);

  // Compare solver statuses
  mu_assert("Basic QP test update rho: Error in solver status!",
            work->info->status_val == sols_data->status_test);

  // Compare primal solutions
  mu_assert("Basic QP test update rho: Error in primal solution!",
            vec_norm_inf_diff(work->solution->x, sols_data->x_test,
                              data->n)/vec_norm_inf(sols_data->x_test, data->n) < TESTS_TOL);

  // Compare dual solutions
  mu_assert("Basic QP test update rho: Error in dual solution!",
            vec_norm_inf_diff(work->solution->y, sols_data->y_test,
                              data->m)/vec_norm_inf(sols_data->y_test, data->m)< TESTS_TOL);

  // Compare objective values
  mu_assert("Basic QP test update rho: Error in objective value!",
            c_absval(work->info->obj_val - sols_data->obj_value_test) <
            TESTS_TOL);

  // Get number of iterations
  n_iter_update_rho = work->info->iter;

  // Assert same number of iterations
  mu_assert("Basic QP test update rho: Error in number of iterations!",
            n_iter_new_solver == n_iter_update_rho);

  // Cleanup solver
  osqp_cleanup(work);

  // Cleanup data
  clean_problem_basic_qp(data);
  clean_problem_basic_qp_sols_data(sols_data);

  // Cleanup
  c_free(settings);

  return 0;
}

#ifdef PROFILING
static const char* test_basic_qp_time_limit()
{
  c_int exitflag;

  // Problem settings
  OSQPSettings *settings = (OSQPSettings *)c_malloc(sizeof(OSQPSettings));

  // Structures
  OSQPWorkspace *work; // Workspace
  OSQPData *data;      // Data
  basic_qp_sols_data *sols_data;

  // Populate data
  data = generate_problem_basic_qp();
  sols_data = generate_problem_basic_qp_sols_data();

  // Define Solver settings as default
  osqp_set_default_settings(settings);
  settings->rho = 20;
  settings->adaptive_rho = 0;

  // Check default time limit
  mu_assert("Basic QP test time limit: Default not correct", settings->time_limit == 0);

  // Setup workspace
  exitflag = osqp_setup(&work, data, settings);

  // Setup correct
  mu_assert("Basic QP test time limit: Setup error!", exitflag == 0);

  // Solve Problem
  osqp_solve(work);

  // Compare solver statuses
  mu_assert("Basic QP test time limit: Error in no time limit solver status!",
	    work->info->status_val == sols_data->status_test);

  // Update time limit
# ifdef PRINTING
  osqp_update_time_limit(work, 1e-5);
  osqp_update_eps_rel(work, 1e-09);
  osqp_update_eps_abs(work, 1e-09);
# else
  // Not printing makes the code run a lot faster, so we need to make it work harder
  // to fail by time limit exceeded
  osqp_update_time_limit(work, 1e-7);
  osqp_update_eps_rel(work, 1e-12);
  osqp_update_eps_abs(work, 1e-12);
# endif
  osqp_update_max_iter(work, (c_int)2e9);
  osqp_update_check_termination(work, 0);

  // Solve Problem
  cold_start(work);
  osqp_solve(work);

  // Compare solver statuses
  mu_assert("Basic QP test time limit: Error in timed out solver status!",
	    work->info->status_val == OSQP_TIME_LIMIT_REACHED);

  // Cleanup solver
  osqp_cleanup(work);

  // Cleanup data
  clean_problem_basic_qp(data);
  clean_problem_basic_qp_sols_data(sols_data);

  // Cleanup
  c_free(settings);

  return 0;
}
#endif // PROFILING


static const char* test_basic_qp_warm_start()
{
  c_int exitflag, iter;

  // Cold started variables
  c_float x0[2] = { 0.0, 0.0, };
  c_float y0[4] = { 0.0, 0.0, 0.0, 0.0, };

  // Optimal solution
  c_float xopt[2] = { 0.3, 0.7, };
  c_float yopt[4] = {-2.9, 0.0, 0.2, 0.0, };

  // Problem settings
  OSQPSettings *settings = (OSQPSettings *)c_malloc(sizeof(OSQPSettings));

  // Structures
  OSQPWorkspace *work; // Workspace
  OSQPData *data;      // Data
  basic_qp_sols_data *sols_data;

  // Populate data
  data = generate_problem_basic_qp();
  sols_data = generate_problem_basic_qp_sols_data();

  // Define Solver settings as default
  osqp_set_default_settings(settings);
  settings->check_termination = 1;

  // Setup workspace
  exitflag = osqp_setup(&work, data, settings);

  // Solve Problem
  osqp_solve(work);
  iter = work->info->iter;

  // Cold start and solve again
  osqp_warm_start(work, x0, y0);
  osqp_solve(work);

  // Check if the number of iterations is the same
  mu_assert("Basic QP test warm start: Cold start error!", work->info->iter == iter);

  // Warm start from the solution and solve again
  osqp_warm_start_x(work, xopt);
  osqp_warm_start_y(work, yopt);
  osqp_solve(work);

  // Check that the number of iterations equals 1
  mu_assert("Basic QP test warm start: Warm start error!", work->info->iter == 1);

  // Cleanup solver
  osqp_cleanup(work);

  // Cleanup data
  clean_problem_basic_qp(data);
  clean_problem_basic_qp_sols_data(sols_data);

  // Cleanup
  c_free(settings);

  return 0;
}


static const char* test_basic_qp()
{
  mu_run_test(test_basic_qp_solve);
#ifdef ENABLE_MKL_PARDISO
  mu_run_test(test_basic_qp_solve_pardiso);
#endif
  mu_run_test(test_basic_qp_update);
  mu_run_test(test_basic_qp_check_termination);
  mu_run_test(test_basic_qp_update_rho);
#ifdef PROFILING
  mu_run_test(test_basic_qp_time_limit);
#endif
  mu_run_test(test_basic_qp_warm_start);

  return 0;
}
