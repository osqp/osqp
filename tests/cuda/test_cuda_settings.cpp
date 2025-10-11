#include <catch2/catch.hpp>

#include "osqp_api.h"    /* OSQP API wrapper (public + some private) */
#include "osqp_tester.h" /* Tester helpers */
#include "test_utils.h"  /* Testing Helper functions */

#include "cuda_data.h"

TEST_CASE_METHOD(cuda_test_fixture, "CUDA: Device number", "[cuda][settings]")
{
  OSQPInt exitflag;

  // Default settings
  osqp_set_default_settings(settings.get());

  SECTION( "CUDA: Device number: Negative device" ) {
    // Try a negative device number - this is never valid
    settings->device = -1;

    // Setup solver
    exitflag = osqp_setup(&tmpSolver, data->P, data->q,
                          data->A, data->l, data->u,
                          data->m, data->n, settings.get());

    mu_assert("Negative device not caught", exitflag == OSQP_SETTINGS_VALIDATION_ERROR );
  }

  SECTION( "CUDA: Device number: Non-existent device" ) {
    // Try a very large device number
    settings->device = 100;

    // Setup solver
    exitflag = osqp_setup(&tmpSolver, data->P, data->q,
                          data->A, data->l, data->u,
                          data->m, data->n, settings.get());

    mu_assert("Non-existent device not caught", exitflag == OSQP_ALGEBRA_LOAD_ERROR );
  }
}
