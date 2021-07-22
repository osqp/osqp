/* OSQP TESTER MODULE */

#include "osqp.h"
#include "osqp_tester.h"

#define CATCH_CONFIG_MAIN
#include "catch.hpp"

/* THE CODE FOR MINIMAL UNIT TESTING HAS BEEN TAKEN FROM
   http://www.jera.com/techinfo/jtns/jtn002.html */
#define mu_assert(message, test) \
  do { if (!(test)) return message; } while (0)


TEST_CASE( "1: All test cases reside in other .cpp files (empty)", "[multi-file:1]" ) {
}
