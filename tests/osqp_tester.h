// Utilities for testing

#ifndef OSQP_TESTER_H
#define OSQP_TESTER_H

#include "osqp.h"
#include "osqp_api.h"

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

/*
 * Test fixture to hold various types needed for OSQP tests
 */
class OSQPTestFixture {
public:
    OSQPTestFixture() {
        settings.reset((OSQPSettings*) c_malloc(sizeof(OSQPSettings)));

        // Initialize default test settings
        osqp_set_default_settings(settings.get());
        settings->device = deviceNumber;

        /*
         * Common solver settings
         */
        settings->rho   = 0.1;
        settings->alpha = 1.6;

        settings->max_iter = 2000;

        settings->scaling = 1;
        settings->verbose = 1;

        settings->eps_abs = 1e-5;
        settings->eps_rel = 1e-5;
    }

    static int deviceNumber;

protected:
    // Settings to use for the test
    OSQPSettings_ptr settings;

    // OSQP Solver itself
    OSQPSolver*    tmpSolver = nullptr;
    OSQPSolver_ptr solver{nullptr};   // Wrap solver inside memory management
};

#endif /* ifndef OSQP_TESTER_H */
