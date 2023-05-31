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

/* QP problem data */
class OSQPTestData {
public:
    OSQPTestData() {};

    virtual ~OSQPTestData() {
        // Clean vectors
        c_free(l);
        c_free(u);
        c_free(q);

        //Clean Matrices
        c_free(A->x);
        c_free(A->i);
        c_free(A->p);
        c_free(A);
        c_free(P->x);
        c_free(P->i);
        c_free(P->p);
        c_free(P);
    };

    OSQPInt        n;
    OSQPInt        m;
    OSQPCscMatrix* P;
    OSQPFloat*     q;
    OSQPCscMatrix* A;
    OSQPFloat*     l;
    OSQPFloat*     u;
};

#include <memory>

class OSQPBaseFixture {
public:
    OSQPBaseFixture() {}
    virtual ~OSQPBaseFixture() {}
};

/*
 * Test fixture to hold various types needed for OSQP tests
 */
class OSQPTestFixture{
public:
    OSQPTestFixture()
    {
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

    virtual ~OSQPTestFixture() {}

    /* Device number to use in the test suite */
    static int deviceNumber;

protected:
    // Settings to use for the test
    OSQPSettings_ptr settings;

    // OSQP Solver itself
    OSQPSolver*    tmpSolver = nullptr;
    OSQPSolver_ptr solver{nullptr};   // Wrap solver inside memory management

    // Test data
    std::unique_ptr<OSQPTestData> data;
};

#endif /* ifndef OSQP_TESTER_H */
