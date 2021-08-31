/* OSQP TESTER MODULE */

#define mu_assert(msg, pred) do { INFO(msg); REQUIRE(pred); } while((void)0, 0)
#define TESTS_TOL 1e-4 // Define tests tolerance