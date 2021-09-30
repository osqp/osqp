// Utilities for testing

#ifndef OSQP_TESTER_H
#define OSQP_TESTER_H

#define mu_assert(msg, pred) do { INFO(msg); REQUIRE(pred); } while((void)0, 0)
#define TESTS_TOL 1e-4 // Define tests tolerance

/* create structure to hold problem data */
/* similar to OSQP internal container, but */
/* holding only bare array types and csc */
typedef struct {
c_int    n;
c_int    m;
csc     *P;
c_float *q;
csc     *A;
c_float *l;
c_float *u;
} OSQPTestData;

#endif // #ifndef OSQP_TESTER_H