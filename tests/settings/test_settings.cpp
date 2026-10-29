#include <catch2/catch.hpp>

#include "osqp_api.h"    /* OSQP API wrapper (public + some private) */
#include "osqp_tester.h" /* Tester helpers */
#include "test_utils.h"  /* Testing Helper functions */

TEST_CASE("Settings: defaults", "[settings],[defaults]")
{
/* This test only works when there is no padding in the struct. */
#ifdef OSQP_PACK_SETTINGS
  OSQPSettings_ptr settings{(OSQPSettings *)c_malloc(sizeof(OSQPSettings))};

  // All elements of the struct are multiples of 32-bits/4 bytes
  uint32_t* settings_int = (uint32_t*) settings.get();

  // Put sentinel values into the structure
  for(int i=0; i < sizeof(OSQPSettings)/4; i++)
  {
    settings_int[i] = 0xDEADBEEF;
  }

  // Define codegen settings
  osqp_set_default_settings(settings.get());

  // See if the sentinel value is still present after getting the default values
  for(int i=0; i < sizeof(OSQPSettings)/4; i++)
  {
    INFO("i = " << i );
    mu_assert("Settings value not initialized", settings_int[i] != 0xDEADBEEF );
  }
#endif
}
TEST_CASE("Settings: time-based adaptive rho interval", "[settings],[adaptive_rho]")
{
#if defined(OSQP_ENABLE_PROFILING) && (OSQP_EMBEDDED_MODE != 1)
  /* Regression test for https://github.com/osqp/osqp/issues/778
   *
   * When the time-based adaptive rho update computes the fixed iteration
   * interval, the clamp must keep the interval at least 1 and at least
   * check_termination. A c_min there forced the interval to always be 1,
   * making rho update on every single iteration instead.
   */

  /* Small QP (the OSQP demo problem) */
  OSQPFloat Px[3] = {4.0, 1.0, 2.0};
  OSQPInt   Pi[3] = {0, 0, 1};
  OSQPInt   Pp[3] = {0, 1, 3};
  OSQPFloat q[2]  = {1.0, 1.0};
  OSQPFloat Ax[4] = {1.0, 1.0, 1.0, 1.0};
  OSQPInt   Ai[4] = {0, 1, 0, 2};
  OSQPInt   Ap[3] = {0, 2, 4};
  OSQPFloat l[3]  = {1.0, 0.0, 0.0};
  OSQPFloat u[3]  = {1.0, 0.7, 0.7};

  OSQPInt n = 2;
  OSQPInt m = 3;

  std::unique_ptr<OSQPCscMatrix, decltype(&OSQPCscMatrix_free)> P{OSQPCscMatrix_new(n, n, 3, Px, Pi, Pp), &OSQPCscMatrix_free};
  std::unique_ptr<OSQPCscMatrix, decltype(&OSQPCscMatrix_free)> A{OSQPCscMatrix_new(m, n, 4, Ax, Ai, Ap), &OSQPCscMatrix_free};
  REQUIRE(P.get() != nullptr);
  REQUIRE(A.get() != nullptr);

  OSQPSettings_ptr settings{OSQPSettings_new()};
  osqp_set_default_settings(settings.get());
  settings->verbose               = 0;
  settings->max_iter              = 4000;
  settings->eps_abs               = 1e-12;
  settings->eps_rel               = 1e-12;
  settings->adaptive_rho          = OSQP_ADAPTIVE_RHO_UPDATE_TIME;
  settings->adaptive_rho_interval = 0;    /* compute the interval from the solve time */
  settings->adaptive_rho_fraction = 1e-4; /* tiny, so the time threshold is passed almost immediately */
  settings->check_termination     = 25;

  OSQPSolver* tmpSolver = nullptr;
  OSQPInt exitflag = osqp_setup(&tmpSolver, P.get(), q, A.get(), l, u, m, n, settings.get());
  OSQPSolver_ptr solver{tmpSolver};
  mu_assert("Settings adaptive rho: Setup error!", exitflag == 0);

  osqp_solve(solver.get());

  /* The interval is either still 0 (the time-based update never ran, for
   * example if the solve finished immediately) or it must respect the
   * documented floor: at least 1 and at least check_termination. Since the
   * interval is rounded to a multiple of check_termination, a legal value is
   * also a multiple of it. */
  const OSQPInt interval = solver->settings->adaptive_rho_interval;
  INFO("adaptive_rho_interval = " << interval);
  if (interval != 0) {
    mu_assert("Time-based adaptive rho interval was forced to 1 instead of being clamped to at least check_termination",
              interval >= settings->check_termination);
    mu_assert("Time-based adaptive rho interval is not a multiple of check_termination",
              interval % settings->check_termination == 0);
  }
#endif
}
