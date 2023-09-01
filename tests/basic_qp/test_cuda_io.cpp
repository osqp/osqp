#include <catch2/catch.hpp>

#include "osqp_api.h"    /* OSQP API wrapper (public + some private) */
#include "osqp_api_types.h"
#include "osqp_tester.h" /* Tester helpers */
#include "test_utils.h"  /* Testing Helper functions */

#include "cuda_memory.h"
#include "cuda_lin_alg.h"

#include "basic_qp_data.h"


TEST_CASE_METHOD(basic_qp_test_fixture, "Basic QP: Solve using CUDA in/out data", "[solve][qp][cuda]")
{
  OSQPInt exitflag;

  // Test-specific options
  settings->allocate_solution = 0;
  settings->polishing         = 1;
  settings->scaling           = 0;
  settings->warm_starting     = 0;

  /*
   * Move the data to the GPU ahead of time
   */

  // A matrix
  OSQPCscMatrix* cuA = (OSQPCscMatrix*) c_malloc(sizeof(OSQPCscMatrix));

  cuA->m = data->A->m;
  cuA->n = data->A->n;
  cuA->nz = data->A->nz;
  cuA->nzmax = data->A->nzmax;
  cuA->x = nullptr;
  cuA->i = nullptr;
  cuA->p = nullptr;

  cuda_malloc((void**) &(cuA->x), data->A->nzmax * sizeof(OSQPFloat));
  cuda_malloc((void**) &(cuA->i), data->A->nzmax * sizeof(OSQPInt));
  cuda_malloc((void**) &(cuA->p), (data->A->n + 1) * sizeof(OSQPInt));

  mu_assert("Vector Ax not on device", cuda_isdeviceptr(cuA->x));
  mu_assert("Vector Ai not on device", cuda_isdeviceptr(cuA->i));
  mu_assert("Vector Ap not on device", cuda_isdeviceptr(cuA->p));

  cuda_vec_copy_h2d(cuA->x, data->A->x, data->A->nzmax);
  cuda_vec_int_copy_h2d(cuA->i, data->A->i, data->A->nzmax);
  cuda_vec_int_copy_h2d(cuA->p, data->A->p, (data->A->n + 1));

  // P matrix
  OSQPCscMatrix* cuP = (OSQPCscMatrix*) c_malloc(sizeof(OSQPCscMatrix));

  cuP->m = data->P->m;
  cuP->n = data->P->n;
  cuP->nz = data->P->nz;
  cuP->nzmax = data->P->nzmax;
  cuP->x = nullptr;
  cuP->i = nullptr;
  cuP->p = nullptr;

  cuda_malloc((void**) &(cuP->x), data->P->nzmax * sizeof(OSQPFloat));
  cuda_malloc((void**) &(cuP->i), data->P->nzmax * sizeof(OSQPInt));
  cuda_malloc((void**) &(cuP->p), (data->P->n + 1) * sizeof(OSQPInt));

  mu_assert("Vector Px not on device", cuda_isdeviceptr(cuP->x));
  mu_assert("Vector Pi not on device", cuda_isdeviceptr(cuP->i));
  mu_assert("Vector Pp not on device", cuda_isdeviceptr(cuP->p));

  cuda_vec_copy_h2d(cuP->x, data->P->x, data->P->nzmax);
  cuda_vec_int_copy_h2d(cuP->i, data->P->i, data->P->nzmax);
  cuda_vec_int_copy_h2d(cuP->p, data->P->p, (data->P->n + 1));

  // Vectors
  OSQPFloat* cuq = nullptr;
  OSQPFloat* cul = nullptr;
  OSQPFloat* cuu = nullptr;

  cuda_malloc((void**) &cuq, data->n * sizeof(OSQPFloat));
  cuda_malloc((void**) &cul, data->m * sizeof(OSQPFloat));
  cuda_malloc((void**) &cuu, data->m * sizeof(OSQPFloat));

  mu_assert("Vector q not on device", cuda_isdeviceptr(cuq));
  mu_assert("Vector l not on device", cuda_isdeviceptr(cul));
  mu_assert("Vector u not on device", cuda_isdeviceptr(cuu));

  cuda_vec_copy_h2d(cuq, data->q, data->n);
  cuda_vec_copy_h2d(cul, data->l, data->m);
  cuda_vec_copy_h2d(cuu, data->u, data->m);


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
  mu_assert("Basic QP test solve: Solution improperly allocated", !(solver->solution));

  // Solve Problem
  osqp_solve(solver.get());

  // Compare solver statuses
  mu_assert("Basic QP test solve: Error in solver status!",
      solver->info->status_val == sols_data->status_test);

  /*
   * Get the solution out using GPU-allocated sotrage
   */
  OSQPSolution_ptr solution{(OSQPSolution*)c_calloc(1, sizeof(OSQPSolution))};
  solution->x             = nullptr;
  solution->y             = nullptr;
  solution->prim_inf_cert = nullptr;
  solution->dual_inf_cert = nullptr;

  cuda_malloc((void**) &(solution->x), data->n * sizeof(OSQPFloat));
  cuda_malloc((void**) &(solution->y), data->m * sizeof(OSQPFloat));
  cuda_malloc((void**) &(solution->prim_inf_cert), data->m * sizeof(OSQPFloat));
  cuda_malloc((void**) &(solution->dual_inf_cert), data->n * sizeof(OSQPFloat));

  mu_assert("Vector sol->x not on device", cuda_isdeviceptr(solution->x));
  mu_assert("Vector sol->y not on device", cuda_isdeviceptr(solution->y));
  mu_assert("Vector sol->prim_inf_cert not on device", cuda_isdeviceptr(solution->prim_inf_cert));
  mu_assert("Vector sol->dual_inf_cert not on device", cuda_isdeviceptr(solution->dual_inf_cert));

  osqp_get_solution(solver.get(), solution.get());

  // Move actual solution to the GPU for comparison
  OSQPFloat* cusolx = nullptr;
  OSQPFloat* cusoly = nullptr;

  cuda_malloc((void**) &cusolx, data->n * sizeof(OSQPFloat));
  cuda_malloc((void**) &cusoly, data->m * sizeof(OSQPFloat));

  mu_assert("Vector sol_test->x not on device", cuda_isdeviceptr(cusolx));
  mu_assert("Vector sol_test->y not on device", cuda_isdeviceptr(cusoly));

  cuda_vec_copy_h2d(cusolx, sols_data->x_test, data->n);
  cuda_vec_copy_h2d(cusoly, sols_data->y_test, data->m);

  // Compare primal solutions
  OSQPFloat res = 0.0;
  cuda_vec_diff_norm_inf(solution->x, cusolx, data->n, &res);
  mu_assert("Basic QP test solve: Error in primal solution!",
      res < TESTS_TOL);

  // Compare dual solutions
  res = 0.0;
  cuda_vec_diff_norm_inf(solution->y, cusoly, data->n, &res);
  mu_assert("Basic QP test solve: Error in dual solution!",
      res < TESTS_TOL);

  // Compare objective values
  mu_assert("Basic QP test solve: Error in objective value!",
      c_absval(solver->info->obj_val - sols_data->obj_value_test) <
      TESTS_TOL);

  /*
   * Free device-allocated data
   */
  cuda_free((void**) &cusolx);
  cuda_free((void**) &cusoly);

  cuda_free((void**) &cuq);
  cuda_free((void**) &cul);
  cuda_free((void**) &cuu);

  cuda_free((void**) &(cuA->x));
  cuda_free((void**) &(cuA->i));
  cuda_free((void**) &(cuA->p));
  c_free(cuA);

  cuda_free((void**) &(cuP->x));
  cuda_free((void**) &(cuP->i));
  cuda_free((void**) &(cuP->p));
  c_free(cuP);

  // Delete manually and zero the pointer since the solution unique_ptr assumes CPU-based memory
  cuda_free((void**) &(solution->x));
  cuda_free((void**) &(solution->y));
  cuda_free((void**) &(solution->prim_inf_cert));
  cuda_free((void**) &(solution->dual_inf_cert));

  solution->x = nullptr;
  solution->y = nullptr;
  solution->prim_inf_cert = nullptr;
  solution->dual_inf_cert = nullptr;
}
