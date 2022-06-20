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

#include <memory.h>

struct OSQPSolver_deleter {
    void operator()(OSQPSolver* solver) {
        osqp_cleanup(solver);
    }
};

struct OSQPSettings_deleter {
    void operator()(OSQPSettings* settings) {
        c_free(settings);
    }
};

struct OSQPCodegenDefines_deleter {
    void operator()(OSQPCodegenDefines* defines) {
        c_free(defines);
    }
};

using OSQPSolver_ptr = std::unique_ptr<OSQPSolver, OSQPSolver_deleter>;
using OSQPSettings_ptr = std::unique_ptr<OSQPSettings, OSQPSettings_deleter>;
using OSQPCodegenDefines_ptr = std::unique_ptr<OSQPCodegenDefines, OSQPCodegenDefines_deleter>;

#endif /* ifndef OSQP_TESTER_H */
