#include <catch2/catch.hpp>

#include "osqp_api.h"    /* OSQP API wrapper (public + some private) */
#include "osqp_tester.h" /* Tester helpers */
#include "test_utils.h"  /* Testing Helper functions */

#include "basic_qp_data.h"


TEST_CASE_METHOD(basic_qp_test_fixture, "Basic QP: Solve", "[solve][qp]")
{
  OSQPInt exitflag;

  // Test-specific options
  settings->polishing     = 1;
  settings->scaling       = 0;
  settings->warm_starting = 0;

  /* Test all possible linear system solvers in this test case */
  settings->linsys_solver = GENERATE(filter(&isLinsysSupported, values({OSQP_DIRECT_SOLVER, OSQP_INDIRECT_SOLVER})));

  CAPTURE(settings->linsys_solver);

  // Setup solver
  exitflag = osqp_setup(&tmpSolver, data->P, data->q,
                        data->A, data->l, data->u,
                        data->m, data->n, settings.get());
  solver.reset(tmpSolver);

  // Setup correct
  mu_assert("Basic QP test solve: Setup error!", exitflag == 0);

  // Solve Problem
  osqp_solve(solver.get());

  // Compare solver statuses
  mu_assert("Basic QP test solve: Error in solver status!",
      solver->info->status_val == sols_data->status_test);

  // Compare primal solutions
  mu_assert("Basic QP test solve: Error in primal solution!",
      vec_norm_inf_diff(solver->solution->x, sols_data->x_test,
            data->n) < TESTS_TOL);

  // Compare dual solutions
  mu_assert("Basic QP test solve: Error in dual solution!",
      vec_norm_inf_diff(solver->solution->y, sols_data->y_test,
            data->m) < TESTS_TOL);


  // Compare objective values
  mu_assert("Basic QP test solve: Error in objective value!",
      c_absval(solver->info->obj_val - sols_data->obj_value_test) <
      TESTS_TOL);
}

TEST_CASE_METHOD(basic_qp_test_fixture, "Basic QP: Settings", "[solve][qp]")
{
  OSQPInt        exitflag;
  OSQPInt        tmp_int;
  OSQPFloat      tmp_float;
  OSQPCscMatrix* tmp_mat;
  OSQPCscMatrix* P_tmp;

  // Define Solver settings as default
  settings->max_iter      = 2000;
  settings->alpha         = 1.6;
  settings->polishing     = 1;
  settings->scaling       = 0;
  settings->verbose       = 1;
  settings->warm_starting = 0;

  // Setup solver
  exitflag = osqp_setup(&tmpSolver, data->P, data->q,
                        data->A, data->l, data->u,
                        data->m, data->n, settings.get());
  solver.reset(tmpSolver);

  // Setup correct
  mu_assert("Basic QP test solve: Setup error!", exitflag == 0);


  // Solve Problem
  osqp_solve(solver.get());

  // Compare solver statuses
  mu_assert("Basic QP test solve: Error in solver status!",
	    solver->info->status_val == sols_data->status_test);

  // Compare primal solutions
  mu_assert("Basic QP test solve: Error in primal solution!",
	    vec_norm_inf_diff(solver->solution->x, sols_data->x_test,
			      data->n) < TESTS_TOL);

  // Compare dual solutions
  mu_assert("Basic QP test solve: Error in dual solution!",
	    vec_norm_inf_diff(solver->solution->y, sols_data->y_test,
			      data->m) < TESTS_TOL);


  // Compare objective values
  mu_assert("Basic QP test solve: Error in objective value!",
	    c_absval(solver->info->obj_val - sols_data->obj_value_test) <
	    TESTS_TOL);

  // Try to set wrong settings
  mu_assert("Basic QP test solve: Wrong value of rho not caught!",
	    osqp_update_rho(solver.get(), -0.1) == 1);

  settings->max_iter = -1;
  mu_assert("Basic QP test solve: Wrong value of max_iter not caught!",
	    osqp_update_settings(solver.get(), settings.get()) > 0);
  settings->max_iter = 2000;

  settings->eps_abs = -1.;
  mu_assert("Basic QP test solve: Wrong value of eps_abs not caught!",
	    osqp_update_settings(solver.get(), settings.get()) > 0);
  settings->eps_abs = OSQP_EPS_ABS;

  settings->eps_rel = -1.;
  mu_assert("Basic QP test solve: Wrong value of eps_rel not caught!",
	    osqp_update_settings(solver.get(), settings.get()) > 0);
  settings->eps_rel = OSQP_EPS_REL;

  settings->eps_prim_inf = -0.1;
  mu_assert("Basic QP test solve: Wrong value of eps_prim_inf not caught!",
	    osqp_update_settings(solver.get(), settings.get()) > 0);
  settings->eps_prim_inf = OSQP_EPS_PRIM_INF;

  settings->eps_dual_inf = -0.1;
  mu_assert("Basic QP test solve: Wrong value of eps_dual_inf not caught!",
	    osqp_update_settings(solver.get(), settings.get()) > 0);
  settings->eps_dual_inf = OSQP_EPS_DUAL_INF;

  settings->alpha = 2.0;
  mu_assert("Basic QP test solve: Wrong value of alpha not caught!",
	    osqp_update_settings(solver.get(), settings.get()) > 0);
  settings->alpha = OSQP_ALPHA;

  settings->warm_starting = -1;
  mu_assert("Basic QP test solve: Wrong value of warm_starting not caught!",
	    osqp_update_settings(solver.get(), settings.get()) > 0);
  settings->warm_starting = 0;

  settings->scaled_termination = 2;
  mu_assert("Basic QP test solve: Wrong value of scaled_termination not caught!",
	    osqp_update_settings(solver.get(), settings.get()) > 0);
  settings->scaled_termination = OSQP_SCALED_TERMINATION;

  settings->check_termination = -1;
  mu_assert("Basic QP test solve: Wrong value of check_termination not caught!",
	    osqp_update_settings(solver.get(), settings.get()) > 0);
  settings->check_termination = OSQP_CHECK_TERMINATION;

  settings->delta = 0.0;
  mu_assert("Basic QP test solve: Wrong value of delta not caught!",
	    osqp_update_settings(solver.get(), settings.get()) > 0);
  settings->delta = OSQP_DELTA;

  settings->polishing = 2;
  mu_assert("Basic QP test solve: Wrong value of polishing not caught!",
	    osqp_update_settings(solver.get(), settings.get()) > 0);
  settings->polishing = 1;

  settings->polish_refine_iter = -1;
  mu_assert("Basic QP test solve: Wrong value of polish_refine_iter not caught!",
	    osqp_update_settings(solver.get(), settings.get()) > 0);
  settings->polish_refine_iter = OSQP_POLISH_REFINE_ITER;

  settings->verbose = 2;
  mu_assert("Basic QP test solve: Wrong value of verbose not caught!",
	    osqp_update_settings(solver.get(), settings.get()) > 0);
  settings->verbose = 1;

  /* =============================
       SETUP WITH WRONG SETTINGS
     ============================= */
  tmpSolver = nullptr;

  // Setup solver with empty settings
  exitflag = osqp_setup(&tmpSolver, data->P, data->q,
                        data->A, data->l, data->u,
                        data->m, data->n, OSQP_NULL);
  solver.reset(tmpSolver);

  mu_assert("Basic QP test solve: Setup should result in error due to empty settings",
            exitflag == OSQP_SETTINGS_VALIDATION_ERROR);

  // Setup solver with a wrong number of scaling iterations
  tmp_int = settings->scaling;
  settings->scaling = -1;
  exitflag = osqp_setup(&tmpSolver, data->P, data->q,
                        data->A, data->l, data->u,
                        data->m, data->n, settings.get());
  solver.reset(tmpSolver);
  mu_assert("Basic QP test solve: Setup should result in error due to a negative number of scaling iterations",
            exitflag == OSQP_SETTINGS_VALIDATION_ERROR);
  settings->scaling = tmp_int;

  // Setup solver with wrong settings->adaptive_rho
  tmp_int = settings->adaptive_rho;
  settings->adaptive_rho = 2;
  exitflag = osqp_setup(&tmpSolver, data->P, data->q,
                        data->A, data->l, data->u,
                        data->m, data->n, settings.get());
  solver.reset(tmpSolver);
  mu_assert("Basic QP test solve: Setup should result in error due to non-boolean settings->adaptive_rho",
            exitflag == OSQP_SETTINGS_VALIDATION_ERROR);
  settings->adaptive_rho = tmp_int;

  // Setup solver with wrong settings->adaptive_rho_interval
  tmp_int = settings->adaptive_rho_interval;
  settings->adaptive_rho_interval = -1;
  exitflag = osqp_setup(&tmpSolver, data->P, data->q,
                        data->A, data->l, data->u,
                        data->m, data->n, settings.get());
  solver.reset(tmpSolver);
  mu_assert("Basic QP test solve: Setup should result in error due to negative settings->adaptive_rho_interval",
            exitflag == OSQP_SETTINGS_VALIDATION_ERROR);
  settings->adaptive_rho_interval = tmp_int;

#ifdef OSQP_ENABLE_PROFILING
  // Setup solver with wrong settings->adaptive_rho_fraction
  tmp_float = settings->adaptive_rho_fraction;
  settings->adaptive_rho_fraction = -1.5;
  // Setup solver
  exitflag = osqp_setup(&tmpSolver, data->P, data->q,
                        data->A, data->l, data->u,
                        data->m, data->n, settings.get());
  solver.reset(tmpSolver);
  mu_assert("Basic QP test solve: Setup should result in error due to non-positive settings->adaptive_rho_fraction",
            exitflag == OSQP_SETTINGS_VALIDATION_ERROR);
  settings->adaptive_rho_fraction = tmp_float;
#endif

  // Setup solver with wrong settings->adaptive_rho_tolerance
  tmp_float = settings->adaptive_rho_tolerance;
  settings->adaptive_rho_tolerance = 0.5;
  exitflag = osqp_setup(&tmpSolver, data->P, data->q,
                        data->A, data->l, data->u,
                        data->m, data->n, settings.get());
  solver.reset(tmpSolver);
  mu_assert("Basic QP test solve: Setup should result in error due to wrong settings->adaptive_rho_tolerance",
            exitflag == OSQP_SETTINGS_VALIDATION_ERROR);
  settings->adaptive_rho_tolerance = tmp_float;

  // Setup solver with wrong settings->polish_refine_iter
  tmp_int = settings->polish_refine_iter;
  settings->polish_refine_iter = -3;
  exitflag = osqp_setup(&tmpSolver, data->P, data->q,
                        data->A, data->l, data->u,
                        data->m, data->n, settings.get());
  solver.reset(tmpSolver);
  mu_assert("Basic QP test solve: Setup should result in error due to negative settings->polish_refine_iter",
            exitflag == OSQP_SETTINGS_VALIDATION_ERROR);
  settings->polish_refine_iter = tmp_int;

  // Setup solver with wrong settings->rho
  tmp_float = settings->rho;
  settings->rho = 0.0;
  exitflag = osqp_setup(&tmpSolver, data->P, data->q,
                        data->A, data->l, data->u,
                        data->m, data->n, settings.get());
  solver.reset(tmpSolver);
  mu_assert("Basic QP test solve: Setup should result in error due to non-positive settings->rho",
            exitflag == OSQP_SETTINGS_VALIDATION_ERROR);
  settings->rho = tmp_float;

  // Setup solver with wrong settings->sigma
  tmp_float = settings->sigma;
  settings->sigma = -0.1;
  exitflag = osqp_setup(&tmpSolver, data->P, data->q,
                        data->A, data->l, data->u,
                        data->m, data->n, settings.get());
  solver.reset(tmpSolver);
  mu_assert("Basic QP test solve: Setup should result in error due to non-positive settings->sigma",
            exitflag == OSQP_SETTINGS_VALIDATION_ERROR);
  settings->sigma = tmp_float;

  // Setup solver with wrong settings->delta
  tmp_float = settings->delta;
  settings->delta = -1.1;
  exitflag = osqp_setup(&tmpSolver, data->P, data->q,
                        data->A, data->l, data->u,
                        data->m, data->n, settings.get());
  solver.reset(tmpSolver);
  mu_assert("Basic QP test solve: Setup should result in error due to non-positive settings->delta",
            exitflag == OSQP_SETTINGS_VALIDATION_ERROR);
  settings->delta = tmp_float;

  // Setup solver with wrong settings->max_iter
  tmp_int = settings->max_iter;
  settings->max_iter = 0;
  exitflag = osqp_setup(&tmpSolver, data->P, data->q,
                        data->A, data->l, data->u,
                        data->m, data->n, settings.get());
  solver.reset(tmpSolver);
  mu_assert("Basic QP test solve: Setup should result in error due to non-positive settings->max_iter",
            exitflag == OSQP_SETTINGS_VALIDATION_ERROR);
  settings->max_iter = tmp_int;

  // Setup solver with wrong settings->eps_abs
  tmp_float = settings->eps_abs;
  settings->eps_abs = -1.1;
  exitflag = osqp_setup(&tmpSolver, data->P, data->q,
                        data->A, data->l, data->u,
                        data->m, data->n, settings.get());
  solver.reset(tmpSolver);
  mu_assert("Basic QP test solve: Setup should result in error due to negative settings->eps_abs",
            exitflag == OSQP_SETTINGS_VALIDATION_ERROR);
  settings->eps_abs = tmp_float;

  // Setup solver with wrong settings->eps_rel
  tmp_float = settings->eps_rel;
  settings->eps_rel = -0.1;
  exitflag = osqp_setup(&tmpSolver, data->P, data->q,
                        data->A, data->l, data->u,
                        data->m, data->n, settings.get());
  solver.reset(tmpSolver);
  mu_assert("Basic QP test solve: Setup should result in error due to negative settings->eps_rel",
            exitflag == OSQP_SETTINGS_VALIDATION_ERROR);
  settings->eps_rel = tmp_float;

  // Setup solver with wrong settings->eps_prim_inf
  tmp_float = settings->eps_prim_inf;
  settings->eps_prim_inf = -0.1;
  exitflag = osqp_setup(&tmpSolver, data->P, data->q,
                        data->A, data->l, data->u,
                        data->m, data->n, settings.get());
  solver.reset(tmpSolver);
  mu_assert("Basic QP test solve: Setup should result in error due to non-positive settings->eps_prim_inf",
            exitflag == OSQP_SETTINGS_VALIDATION_ERROR);
  settings->eps_prim_inf = tmp_float;

  // Setup solver with wrong settings->eps_dual_inf
  tmp_float = settings->eps_dual_inf;
  settings->eps_dual_inf = 0.0;
  exitflag = osqp_setup(&tmpSolver, data->P, data->q,
                        data->A, data->l, data->u,
                        data->m, data->n, settings.get());
  solver.reset(tmpSolver);
  mu_assert("Basic QP test solve: Setup should result in error due to non-positive settings->eps_dual_inf",
            exitflag == OSQP_SETTINGS_VALIDATION_ERROR);
  settings->eps_dual_inf = tmp_float;

  // Setup solver with wrong settings->alpha
  tmp_float = settings->alpha;
  settings->alpha = 2.0;
  exitflag = osqp_setup(&tmpSolver, data->P, data->q,
                        data->A, data->l, data->u,
                        data->m, data->n, settings.get());
  solver.reset(tmpSolver);
  mu_assert("Basic QP test solve: Setup should result in error due to wrong settings->alpha",
            exitflag == OSQP_SETTINGS_VALIDATION_ERROR);
  settings->alpha = tmp_float;

  // Setup solver with wrong settings->linsys_solver
  enum osqp_linsys_solver_type tmp_solver_type = settings->linsys_solver;
  settings->linsys_solver = OSQP_UNKNOWN_SOLVER;
  exitflag = osqp_setup(&tmpSolver, data->P, data->q,
                        data->A, data->l, data->u,
                        data->m, data->n, settings.get());
  solver.reset(tmpSolver);
  mu_assert("Basic QP test solve: Setup should result in error due to wrong settings->linsys_solver",
            exitflag == OSQP_SETTINGS_VALIDATION_ERROR);
  settings->linsys_solver = tmp_solver_type;

  // Setup solver with wrong settings->verbose
  tmp_int = settings->verbose;
  settings->verbose = 2;
  exitflag = osqp_setup(&tmpSolver, data->P, data->q,
                        data->A, data->l, data->u,
                        data->m, data->n, settings.get());
  solver.reset(tmpSolver);
  mu_assert("Basic QP test solve: Setup should result in error due to non-boolean settings->verbose",
            exitflag == OSQP_SETTINGS_VALIDATION_ERROR);
  settings->verbose = tmp_int;

  // Setup solver with wrong settings->scaled_termination
  tmp_int = settings->scaled_termination;
  settings->scaled_termination = 2;
  exitflag = osqp_setup(&tmpSolver, data->P, data->q,
                        data->A, data->l, data->u,
                        data->m, data->n, settings.get());
  solver.reset(tmpSolver);
  mu_assert("Basic QP test solve: Setup should result in error due to non-boolean settings->scaled_termination",
            exitflag == OSQP_SETTINGS_VALIDATION_ERROR);
  settings->scaled_termination = tmp_int;

  // Setup solver with wrong settings->check_termination
  tmp_int = settings->check_termination;
  settings->check_termination = -1;
  exitflag = osqp_setup(&tmpSolver, data->P, data->q,
                        data->A, data->l, data->u,
                        data->m, data->n, settings.get());
  solver.reset(tmpSolver);
  mu_assert("Basic QP test solve: Setup should result in error due to non-boolean settings->check_termination",
            exitflag == OSQP_SETTINGS_VALIDATION_ERROR);
  settings->check_termination = tmp_int;

  // Setup solver with wrong settings->warm_starting
  tmp_int = settings->warm_starting;
  settings->warm_starting = 5;
  exitflag = osqp_setup(&tmpSolver, data->P, data->q,
                        data->A, data->l, data->u,
                        data->m, data->n, settings.get());
  solver.reset(tmpSolver);
  mu_assert("Basic QP test solve: Setup should result in error due to non-boolean settings->warm_starting",
            exitflag == OSQP_SETTINGS_VALIDATION_ERROR);
  settings->warm_starting = tmp_int;

#ifdef OSQP_ENABLE_PROFILING
  // Setup solver with wrong settings->time_limit
  tmp_float = settings->time_limit;
  settings->time_limit = -0.2;
  exitflag = osqp_setup(&tmpSolver, data->P, data->q,
                        data->A, data->l, data->u,
                        data->m, data->n, settings.get());
  solver.reset(tmpSolver);
  mu_assert("Basic QP test solve: Setup should result in error due to wrong settings->time_limit",
            exitflag == OSQP_SETTINGS_VALIDATION_ERROR);
  settings->time_limit = tmp_float;
#endif


  /* =========================
       SETUP WITH WRONG DATA
     ========================= */

  // Setup solver with empty data
  exitflag = osqp_setup(&tmpSolver, OSQP_NULL, OSQP_NULL,
                        OSQP_NULL, OSQP_NULL, OSQP_NULL,
                        data->m, data->n, settings.get());
  solver.reset(tmpSolver);
  mu_assert("Basic QP test solve: Setup should result in error due to empty data",
            exitflag == OSQP_DATA_VALIDATION_ERROR);

  // Setup solver with wrong data->m
  tmp_int = data->m;
  data->m = data->m - 1;
  exitflag = osqp_setup(&tmpSolver, data->P, data->q,
                        data->A, data->l, data->u,
                        data->m, data->n, settings.get());
  solver.reset(tmpSolver);
  mu_assert("Basic QP test solve: Setup should result in error due to wrong data->m",
            exitflag == OSQP_DATA_VALIDATION_ERROR);
  data->m = tmp_int;

  // Setup solver with wrong data->n
  tmp_int = data->n;
  data->n = data->n + 1;
  exitflag = osqp_setup(&tmpSolver, data->P, data->q,
                        data->A, data->l, data->u,
                        data->m, data->n, settings.get());
  solver.reset(tmpSolver);
  mu_assert("Basic QP test solve: Setup should result in error due to wrong data->n",
            exitflag == OSQP_DATA_VALIDATION_ERROR);

  // Setup solver with zero data->n
  data->n = 0;
  exitflag = osqp_setup(&tmpSolver, data->P, data->q,
                        data->A, data->l, data->u,
                        data->m, data->n, settings.get());
  solver.reset(tmpSolver);
  mu_assert("Basic QP test solve: Setup should result in error due to zero data->n",
            exitflag == OSQP_DATA_VALIDATION_ERROR);
  data->n = tmp_int;

  // Setup solver with wrong P->m
  tmp_int = data->P->m;
  data->P->m = data->n + 1;
  exitflag = osqp_setup(&tmpSolver, data->P, data->q,
                        data->A, data->l, data->u,
                        data->m, data->n, settings.get());
  solver.reset(tmpSolver);
  mu_assert("Basic QP test solve: Setup should result in error due to wrong P->m",
            exitflag == OSQP_DATA_VALIDATION_ERROR);
  data->P->m = tmp_int;

  // Setup solver with wrong P->n
  tmp_int = data->P->n;
  data->P->n = data->n + 1;
  exitflag = osqp_setup(&tmpSolver, data->P, data->q,
                        data->A, data->l, data->u,
                        data->m, data->n, settings.get());
  solver.reset(tmpSolver);
  mu_assert("Basic QP test solve: Setup should result in error due to wrong P->n",
            exitflag == OSQP_DATA_VALIDATION_ERROR);
  data->P->n = tmp_int;

  // Setup solver with non-upper-triangular P
  P_tmp = (OSQPCscMatrix*) c_malloc(sizeof(OSQPCscMatrix));
  P_tmp->m = 2;
  P_tmp->n = 2;
  P_tmp->nz = -1;
  P_tmp->nzmax = 4;
  P_tmp->x = (OSQPFloat*) c_malloc(4 * sizeof(OSQPFloat));
  P_tmp->x[0] = 4.0;
  P_tmp->x[1] = 1.0;
  P_tmp->x[2] = 1.0;
  P_tmp->x[3] = 2.0;
  P_tmp->i = (OSQPInt*) c_malloc(4 * sizeof(OSQPInt));
  P_tmp->i[0] = 0;
  P_tmp->i[1] = 1;
  P_tmp->i[2] = 0;
  P_tmp->i[3] = 1;
  P_tmp->p = (OSQPInt*) c_malloc((2 + 1) * sizeof(OSQPInt));
  P_tmp->p[0] = 0;
  P_tmp->p[1] = 2;
  P_tmp->p[2] = 4;

  exitflag = osqp_setup(&tmpSolver, P_tmp, data->q,
                        data->A, data->l, data->u,
                        data->m, data->n, settings.get());
  solver.reset(tmpSolver);
  mu_assert("Basic QP test solve: Setup should result in error due to non-triu structure of P",
            exitflag == OSQP_DATA_VALIDATION_ERROR);

  // Setup solver with non-consistent bounds
  data->l[0] = data->u[0] + 1.0;
  exitflag = osqp_setup(&tmpSolver, data->P, data->q,
                        data->A, data->l, data->u,
                        data->m, data->n, settings.get());
  solver.reset(tmpSolver);
  mu_assert("Basic QP test solve: Setup should result in error due to non-consistent bounds",
            exitflag == OSQP_DATA_VALIDATION_ERROR);

  // Cleanup
  c_free(P_tmp->x);
  c_free(P_tmp->i);
  c_free(P_tmp->p);
  c_free(P_tmp);
}

TEST_CASE_METHOD(basic_qp_test_fixture, "Basic QP: Data update", "[solve][qp][data][update]")
{
  OSQPInt      exitflag;
  OSQPVectorf* q_new;
  OSQPVectorf* l_new;
  OSQPVectorf* u_new;

  // Define Solver settings as default
  settings->max_iter      = 200;
  settings->polishing     = 1;
  settings->scaling       = 0;
  settings->warm_starting = 0;

  // Setup solver
  exitflag = osqp_setup(&tmpSolver, data->P, data->q,
                        data->A, data->l, data->u,
                        data->m, data->n, settings.get());
  solver.reset(tmpSolver);

  // Setup correct
  mu_assert("Basic QP test update: Setup error!", exitflag == 0);


  // ====================================================================
  //  Update data
  // ====================================================================
  SECTION( "QP Updates: linear cost" ) {
    OSQPInt    retval   = OSQP_NO_ERROR;
    OSQPFloat* new_data = sols_data->q_new;

    OSQPVectorf_ptr new_vec{OSQPVectorf_new(new_data, data->n)};

    retval = osqp_update_data_vec(solver.get(), new_data, NULL, NULL);

    mu_assert("QP vector updates: Error in return flag updating linear cost",
              retval == OSQP_NO_ERROR);

    mu_assert("QP vector updates: Error in solver data after updating linear cost",
              OSQPVectorf_norm_inf_diff(solver->work->data->q, new_vec.get()) < TESTS_TOL);
  }

  SECTION( "QP Updates: lower bounds" ) {
    OSQPInt    retval   = OSQP_NO_ERROR;
    OSQPFloat* new_data = sols_data->l_new;

    OSQPVectorf_ptr new_vec{OSQPVectorf_new(new_data, data->m)};

    retval = osqp_update_data_vec(solver.get(), NULL, new_data, data->u);

    mu_assert("QP vector updates: Error in return flag updating lower bounds",
              retval == OSQP_NO_ERROR);

    mu_assert("QP vector updates: Error in solver data after updating lower bounds",
              OSQPVectorf_norm_inf_diff(solver->work->data->l, new_vec.get()) < TESTS_TOL);
  }

  SECTION( "QP Updates: upper bounds" ) {
    OSQPInt    retval   = OSQP_NO_ERROR;
    OSQPFloat* new_data = sols_data->u_new;

    OSQPVectorf_ptr new_vec{OSQPVectorf_new(new_data, data->m)};

    retval = osqp_update_data_vec(solver.get(), NULL, data->l, new_data);

    mu_assert("QP vector updates: Error in return flag updating upper bounds",
              retval == OSQP_NO_ERROR);

    mu_assert("QP vector updates: Error in solver data after updating upper bounds",
              OSQPVectorf_norm_inf_diff(solver->work->data->u, new_vec.get()) < TESTS_TOL);
  }

  SECTION( "QP Updates: lower and upper bounds consistency validation" ) {
    OSQPInt exp_flag = OSQP_NO_ERROR;
    OSQPInt retval   = OSQP_NO_ERROR;

    OSQPFloat* l_cur = data->l;
    OSQPFloat* l_new = sols_data->l_new;
    OSQPFloat* u_cur = data->u;
    OSQPFloat* u_new = sols_data->u_new;
    OSQPFloat* nullv = OSQP_NULL;

    OSQPFloat* new_l_data = nullptr;
    OSQPFloat* exp_l_data = nullptr;
    OSQPFloat* new_u_data = nullptr;
    OSQPFloat* exp_u_data = nullptr;

    std::initializer_list<std::tuple<OSQPFloat**, OSQPFloat**, OSQPFloat**, OSQPFloat**, OSQPInt>> testcases =
      { /* first (second) is new (expected) upper data, third (fourth) is new (expected) lower data, fifth is return code from update func */
        /* Flipped upper and lower (validation error) */
        std::make_tuple( &u_new, &l_cur, &l_new, &u_cur, OSQP_DATA_VALIDATION_ERROR ),
        /* Write lower bound with incorrect data */
        std::make_tuple( &u_new, &l_cur, &nullv, &u_cur, OSQP_DATA_VALIDATION_ERROR ),
        /* Write upper bound with incorrect data */
        std::make_tuple( &nullv, &l_cur, &l_new, &u_cur, OSQP_DATA_VALIDATION_ERROR ),
        /* Correct data */
        std::make_tuple( &l_new, &l_new, &u_new, &u_new, OSQP_NO_ERROR )
      };

    auto test_case = GENERATE_REF( table<OSQPFloat**, OSQPFloat**, OSQPFloat**, OSQPFloat**, OSQPInt>(testcases) );

    new_l_data = *std::get<0>(test_case);
    exp_l_data = *std::get<1>(test_case);
    new_u_data = *std::get<2>(test_case);
    exp_u_data = *std::get<3>(test_case);
    exp_flag = std::get<4>(test_case);

    OSQPVectorf_ptr exp_l_vec{OSQPVectorf_new(exp_l_data, data->m)};
    OSQPVectorf_ptr exp_u_vec{OSQPVectorf_new(exp_u_data, data->m)};

    retval = osqp_update_data_vec(solver.get(), NULL, new_l_data, new_u_data);

    mu_assert("QP vector updates: Error in return flag updating upper both bounds",
              retval == exp_flag);

    mu_assert("QP vector updates: Error in solver upper bound data after updating both bounds",
              OSQPVectorf_norm_inf_diff(solver->work->data->u, exp_u_vec.get()) < TESTS_TOL);

    mu_assert("QP vector updates: Error in solver lower bound data after updating both bounds",
              OSQPVectorf_norm_inf_diff(solver->work->data->l, exp_l_vec.get()) < TESTS_TOL);
  }
}

TEST_CASE_METHOD(basic_qp_test_fixture, "Basic QP: Termination", "[solve][qp]")
{
  OSQPInt exitflag;

  // Problem-specific settings
  osqp_set_default_settings(settings.get());
  settings->max_iter          = 200;
  settings->alpha             = 1.6;
  settings->polishing         = 0;
  settings->scaling           = 0;
  settings->verbose           = 1;
  settings->check_termination = 0;
  settings->warm_starting     = 0;

  /* Test all possible linear system solvers in this test case */
  settings->linsys_solver = GENERATE(filter(&isLinsysSupported, values({OSQP_DIRECT_SOLVER, OSQP_INDIRECT_SOLVER})));

  CAPTURE(settings->linsys_solver, settings->polishing);

  // Setup solver
  exitflag = osqp_setup(&tmpSolver, data->P, data->q,
                        data->A, data->l, data->u,
                        data->m, data->n, settings.get());
  solver.reset(tmpSolver);

  // Setup correct
  mu_assert("Basic QP test solve: Setup error!", exitflag == 0);

  // Solve Problem
  osqp_solve(solver.get());

  // Check if iter == max_iter
  mu_assert(
    "Basic QP test check termination: Error in number of iterations taken!",
    solver->info->iter == solver->settings->max_iter);

  // Compare solver statuses
  mu_assert("Basic QP test check termination: Error in solver status!",
            solver->info->status_val == sols_data->status_test);

  // Compare primal solutions
  mu_assert("Basic QP test check termination: Error in primal solution!",
            vec_norm_inf_diff(solver->solution->x, sols_data->x_test,
                              data->n) < TESTS_TOL);

  // Compare dual solutions
  // print_vec(work->solution->y, data->m, "y_sol");
  // print_vec(sols_data->y_test, data->m, "y_test");
  mu_assert("Basic QP test check termination: Error in dual solution!",
            vec_norm_inf_diff(solver->solution->y, sols_data->y_test,
                              data->m) < TESTS_TOL);

  // Compare objective values
  mu_assert("Basic QP test check termination: Error in objective value!",
            c_absval(solver->info->obj_val - sols_data->obj_value_test) <
            TESTS_TOL);
}

TEST_CASE_METHOD(basic_qp_test_fixture, "Basic QP: Update rho", "[update][qp]")
{
  // Exitflag
  OSQPInt exitflag;

  // rho to use
  OSQPFloat rho;

  /* Test all possible linear system solvers in this test case */
  osqp_linsys_solver_type linsys = GENERATE(filter(&isLinsysSupported, values({OSQP_DIRECT_SOLVER, OSQP_INDIRECT_SOLVER})));

  // Define number of iterations to compare
  OSQPInt n_iter_new_solver;
  OSQPInt n_iter_update_rho;

  // Define Solver settings as default
  rho = 0.7;
  osqp_set_default_settings(settings.get());
  settings->rho               = rho;
  settings->adaptive_rho      = 0; // Disable adaptive rho for this test
  settings->eps_abs           = 5e-05;
  settings->eps_rel           = 5e-05;
  settings->check_termination = 1;
  settings->linsys_solver     = linsys;

  // Setup solver
  exitflag = osqp_setup(&tmpSolver, data->P, data->q,
                        data->A, data->l, data->u,
                        data->m, data->n, settings.get());
  solver.reset(tmpSolver);

  // Setup correct
  mu_assert("Basic QP test update rho: Setup error!", exitflag == 0);

  // Solve Problem
  osqp_solve(solver.get());

  // Store number of iterations
  n_iter_new_solver = solver->info->iter;

  // Compare solver statuses
  mu_assert("Update rho test solve: Error in solver status!",
            solver->info->status_val == sols_data->status_test);

  // Compare primal solutions
  mu_assert("Update rho test solve: Error in primal solution!",
            vec_norm_inf_diff(solver->solution->x, sols_data->x_test,
                              data->n)/vec_norm_inf(sols_data->x_test, data->n) < TESTS_TOL);

  // Compare dual solutions
  mu_assert("Update rho test solve: Error in dual solution!",
            vec_norm_inf_diff(solver->solution->y, sols_data->y_test,
                              data->m)/vec_norm_inf(sols_data->y_test, data->m) < TESTS_TOL);

  // Compare objective values
  mu_assert("Update rho test solve: Error in objective value!",
            c_absval(solver->info->obj_val - sols_data->obj_value_test) <
            TESTS_TOL);

  // Create new problem with different rho and update it
  osqp_set_default_settings(settings.get());
  settings->rho               = 0.1;
  settings->adaptive_rho      = 0;
  settings->check_termination = 1;
  settings->eps_abs           = 5e-05;
  settings->eps_rel           = 5e-05;
  settings->linsys_solver     = linsys;

  // Setup solver
  solver.reset(nullptr);  // TODO (CUDA): Needed until CUDA supports multiple instances
  exitflag = osqp_setup(&tmpSolver, data->P, data->q,
                        data->A, data->l, data->u,
                        data->m, data->n, settings.get());
  solver.reset(tmpSolver);

  // Setup correct
  mu_assert("Basic QP test update rho: Setup error!", exitflag == 0);

  // Update rho
  exitflag = osqp_update_rho(solver.get(), rho);
  mu_assert("Basic QP test update rho: Error update rho!", exitflag == 0);

  // Solve Problem
  osqp_solve(solver.get());

  // Compare solver statuses
  mu_assert("Basic QP test update rho: Error in solver status!",
            solver->info->status_val == sols_data->status_test);

  // Compare primal solutions
  mu_assert("Basic QP test update rho: Error in primal solution!",
            vec_norm_inf_diff(solver->solution->x, sols_data->x_test,
                              data->n)/vec_norm_inf(sols_data->x_test, data->n) < TESTS_TOL);

  // Compare dual solutions
  mu_assert("Basic QP test update rho: Error in dual solution!",
            vec_norm_inf_diff(solver->solution->y, sols_data->y_test,
                              data->m)/vec_norm_inf(sols_data->y_test, data->m)< TESTS_TOL);

  // Compare objective values
  mu_assert("Basic QP test update rho: Error in objective value!",
            c_absval(solver->info->obj_val - sols_data->obj_value_test) <
            TESTS_TOL);

  // Get number of iterations
  n_iter_update_rho = solver->info->iter;

  // Assert same number of iterations
  mu_assert("Basic QP test update rho: Error in number of iterations!",
            n_iter_new_solver == n_iter_update_rho);
}

#ifdef OSQP_ENABLE_PROFILING
TEST_CASE_METHOD(basic_qp_test_fixture, "Basic QP: Time limit", "[solve][qp]")
{
  OSQPInt exitflag;

  // Define Solver settings as default
  osqp_set_default_settings(settings.get());
  settings->rho = 20;
  settings->adaptive_rho = 0;

  // Check default time limit
  mu_assert("Basic QP test time limit: Default not correct",
            settings->time_limit == OSQP_TIME_LIMIT);

  // Setup solver
  exitflag = osqp_setup(&tmpSolver, data->P, data->q,
                        data->A, data->l, data->u,
                        data->m, data->n, settings.get());
  solver.reset(tmpSolver);

  // Setup correct
  mu_assert("Basic QP test time limit: Setup error!", exitflag == 0);

  // Solve Problem
  osqp_solve(solver.get());

  // Compare solver statuses
  mu_assert("Basic QP test time limit: Error in no time limit solver status!",
	    solver->info->status_val == sols_data->status_test);

  // Update time limit
# ifdef OSQP_ENABLE_PRINTING
  settings->time_limit = 1e-5;
  settings->eps_rel = 1e-9;
  settings->eps_abs = 1e-9;
# else
  // Not printing makes the code run a lot faster, so we need to make it work harder
  // to fail by time limit exceeded
  settings->time_limit = 1e-7;
  settings->eps_rel = 1e-12;
  settings->eps_abs = 1e-12;
# endif
  settings->max_iter = (OSQPInt)2e9;
  settings->check_termination = 0;
  osqp_update_settings(solver.get(), settings.get());

  // Solve Problem
  osqp_cold_start(solver.get());
  osqp_solve(solver.get());

  // Compare solver statuses
  mu_assert("Basic QP test time limit: Error in timed out solver status!",
	    solver->info->status_val == OSQP_TIME_LIMIT_REACHED);
}
#endif // OSQP_ENABLE_PROFILING


TEST_CASE_METHOD(basic_qp_test_fixture, "Basic QP: Warm start", "[solve][qp][warm-start]")
{
  OSQPInt exitflag;
  OSQPInt iter;

  // Cold started variables
  OSQPFloat x0[2] = { 0.0, 0.0, };
  OSQPFloat y0[4] = { 0.0, 0.0, 0.0, 0.0, };

  // Optimal solution
  OSQPFloat xopt[2] = { 0.3, 0.7, };
  OSQPFloat yopt[4] = {-2.9, 0.0, 0.2, 0.0, };

  // Setup problem-specific setting
  settings->check_termination = 1;
  settings->adaptive_rho = 0;

  // Setup solver
  exitflag = osqp_setup(&tmpSolver, data->P, data->q,
                        data->A, data->l, data->u,
                        data->m, data->n, settings.get());
  solver.reset(tmpSolver);

  // Setup correct
  mu_assert("Basic QP test warm start: Setup error!", exitflag == 0);

  // Solve Problem initially
  osqp_solve(solver.get());
  iter = solver->info->iter;

  SECTION( "Cold start from original start point" ) {
    // Cold start and solve again
    osqp_warm_start(solver.get(), x0, y0);
    osqp_solve(solver.get());

    // Check if the number of iterations is the same
    mu_assert("Basic QP test warm start: Cold start error!", solver->info->iter == iter);
  }

  SECTION( "Warm start from solution" ) {
    // Warm start from the solution and solve again
    osqp_warm_start(solver.get(), xopt, OSQP_NULL);
    osqp_warm_start(solver.get(), OSQP_NULL, yopt);
    osqp_solve(solver.get());

    // Check that the number of iterations equals 1
    mu_assert("Basic QP test warm start: Warm start error!", solver->info->iter == 1);
  }
}
