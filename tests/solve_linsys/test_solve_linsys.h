#include <stdio.h>
#include "osqp.h"
#include "util.h"
#include "osqp_tester.h"
#include "lin_sys.h"


#include "solve_linsys/data.h"


void test_solveKKT() {

  c_int m, n, exitflag = 0;
  c_float pri_res, dua_res;
  OSQPVectorf *rho_vec, *rhs, *ref;
  OSQPMatrix *Pu, *A;
  LinSysSolver *s;  // Private structure to form KKT factorization
  OSQPSettings *settings = (OSQPSettings *)c_malloc(sizeof(OSQPSettings)); // Settings
  solve_linsys_sols_data *data = generate_problem_solve_linsys_sols_data();

  // Settings
  settings->rho           = data->test_solve_KKT_rho;
  settings->sigma         = data->test_solve_KKT_sigma;
  settings->linsys_solver = LINSYS_SOLVER;

  // Set rho_vec
  m       = data->test_solve_KKT_A->m;
  n       = data->test_solve_KKT_Pu->n;
  rho_vec = OSQPVectorf_malloc(m);
  OSQPVectorf_set_scalar(rho_vec,settings->rho);

  //data Matrices
  Pu = OSQPMatrix_new_from_csc(data->test_solve_KKT_Pu,1);
  A  = OSQPMatrix_new_from_csc(data->test_solve_KKT_A, 0);

  // Set residuals to small values to enforce accurate solution by indirect solvers
  pri_res = 1e-7;
  dua_res = 1e-7;

  // Form and factorize KKT matrix
  exitflag = init_linsys_solver(&s, Pu, A, rho_vec, settings, &pri_res, &dua_res, 0);

  // Solve  KKT x = b via LDL given factorization
  rhs = OSQPVectorf_new(data->test_solve_KKT_rhs, m+n);
  s->solve(s, rhs, 2);
  ref = OSQPVectorf_new(data->test_solve_KKT_x, m+n);

  mu_assert(
    "Linear systems solve tests: error in forming and solving KKT system!",
    OSQPVectorf_norm_inf_diff(rhs, ref) < TESTS_TOL);


  // Cleanup
  s->free(s);
  c_free(settings);
  OSQPVectorf_free(rho_vec);
  OSQPVectorf_free(rhs);
  OSQPVectorf_free(ref);
  OSQPMatrix_free(Pu);
  OSQPMatrix_free(A);
  clean_problem_solve_linsys_sols_data(data);
}

#ifdef ENABLE_MKL_PARDISO
void test_solveKKT_pardiso() {

  c_int m, n, exitflag = 0;
  OSQPVectorf *rho_vec, *rhs, *ref;
  OSQPMatrix *Pu, *A;
  LinSysSolver *s;  // Private structure to form KKT factorization
  OSQPSettings *settings = (OSQPSettings *)c_malloc(sizeof(OSQPSettings)); // Settings
  solve_linsys_sols_data *data = generate_problem_solve_linsys_sols_data();

  // Settings
  settings->rho           = data->test_solve_KKT_rho;
  settings->sigma         = data->test_solve_KKT_sigma;
  settings->linsys_solver = MKL_PARDISO_SOLVER;

  // Set rho_vec
  m       = data->test_solve_KKT_A->m;
  n       = data->test_solve_KKT_Pu->n;
  rho_vec = OSQPVectorf_malloc(m);
  OSQPVectorf_set_scalar(rho_vec,settings->rho);

  //data Matrices
  Pu = OSQPMatrix_new_from_csc(data->test_solve_KKT_Pu,1);
  A  = OSQPMatrix_new_from_csc(data->test_solve_KKT_A,0);

  // Load Pardiso shared library
  exitflag = load_linsys_solver(MKL_PARDISO_SOLVER);
  mu_assert("Linear system solve test: error in loading Pardiso shared library",
            exitflag == 0);

  // Form and factorize KKT matrix
  exitflag = init_linsys_solver(&s, Pu, A, rho_vec, settings, OSQP_NULL, OSQP_NULL, 0);

  // Solve  KKT x = b via LDL given factorization
  rhs = OSQPVectorf_new(data->test_solve_KKT_rhs, m+n);
  s->solve(s, rhs, 1);
  ref = OSQPVectorf_new(data->test_solve_KKT_x, m+n);

  mu_assert(
    "Linear systems solve tests: error in forming and solving KKT system with PARDISO!",
    OSQPVectorf_norm_inf_diff(rhs, ref) < TESTS_TOL);


  // Cleanup
  s->free(s);
  c_free(settings);
  OSQPVectorf_free(rho_vec);
  OSQPVectorf_free(rhs);
  OSQPVectorf_free(ref);
  OSQPMatrix_free(Pu);
  OSQPMatrix_free(A);
  clean_problem_solve_linsys_sols_data(data);

  // Unload Pardiso shared library
  exitflag = unload_linsys_solver(MKL_PARDISO_SOLVER);
}
#endif