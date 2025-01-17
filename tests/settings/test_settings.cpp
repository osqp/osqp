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