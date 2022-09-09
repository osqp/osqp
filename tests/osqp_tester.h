// Utilities for testing

#ifndef OSQP_TESTER_H
#define OSQP_TESTER_H

#include "osqp.h"

#define mu_assert(msg, pred) do { INFO(msg); REQUIRE(pred); } while((void)0, 0)

// Define tests tolerance
#ifndef OSQP_USE_FLOAT
#define TESTS_TOL 1e-4      // Tolerance for doubles
#else
#define TESTS_TOL 1e-3      // Slightly larger tolerance for floats
#endif

/* create structure to hold problem data */
/* similar to OSQP internal container, but */
/* holding only bare array types and csc */
typedef struct {
    OSQPInt        n;
    OSQPInt        m;
    OSQPCscMatrix* P;
    OSQPFloat*     q;
    OSQPCscMatrix* A;
    OSQPFloat*     l;
    OSQPFloat*     u;
} OSQPTestData;

#endif /* ifndef OSQP_TESTER_H */
