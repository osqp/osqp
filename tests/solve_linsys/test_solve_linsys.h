#include <stdio.h>
#include "osqp.h"
#include "osqp_tester.h"

#include "solve_linsys/data.h"


void test_solveKKT() {
  OSQPInt exitflag = 0;

  OSQPVectorf_ptr rho_vec{nullptr};
  OSQPVectorf_ptr rhs{nullptr};
  OSQPVectorf_ptr ref{nullptr};
  OSQPMatrix_ptr  Pu{nullptr};
  OSQPMatrix_ptr  A{nullptr};

  LinSysSolver* s;  // Private structure to form KKT factorization

  // Problem settings
  OSQPSettings_ptr settings{(OSQPSettings *)c_malloc(sizeof(OSQPSettings))};

  solve_linsys_sols_data_ptr data{generate_problem_solve_linsys_sols_data()};

  // Settings
  osqp_set_default_settings(settings.get());
  settings->rho   = data->test_solve_KKT_rho;
  settings->sigma = data->test_solve_KKT_sigma;

  /* Test all possible linear system solvers in this test case */
  settings->linsys_solver = GENERATE(filter(&isLinsysSupported, values({OSQP_DIRECT_SOLVER, OSQP_INDIRECT_SOLVER})));

  // Set rho_vec
  OSQPInt m = data->test_solve_KKT_A->m;
  OSQPInt n = data->test_solve_KKT_Pu->n;

  rho_vec.reset(OSQPVectorf_malloc(m));
  OSQPVectorf_set_scalar(rho_vec.get(), settings->rho);

  // data matrices
  Pu.reset(OSQPMatrix_new_from_csc(data->test_solve_KKT_Pu, 1));
  A.reset(OSQPMatrix_new_from_csc(data->test_solve_KKT_A, 0));

  // Set residuals to small values to enforce accurate solution by indirect solvers
  OSQPFloat prim_res = 1e-7;
  OSQPFloat dual_res = 1e-7;

  // Form and factorize KKT matrix
  exitflag = osqp_algebra_init_linsys_solver(&s, Pu.get(), A.get(), rho_vec.get(), settings.get(), &prim_res, &dual_res, 0);

  // Solve KKT x = rhs
  rhs.reset(OSQPVectorf_new(data->test_solve_KKT_rhs, m+n));
  s->solve(s, rhs.get(), 2);
  ref.reset(OSQPVectorf_new(data->test_solve_KKT_x, m+n));

  mu_assert("Linear systems solve tests: error in forming and solving KKT system!",
            OSQPVectorf_norm_inf_diff(rhs.get(), ref.get()) < TESTS_TOL);

  // Cleanup
  s->free(s);
}
