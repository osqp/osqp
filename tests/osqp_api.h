#ifndef OSQP_API_H_
#define OSQP_API_H_

#include <memory>

/* OSQP Public API */
#include "osqp.h"

/* Needed since we define smart pointers for the linear algebra types */
#include "algebra_matrix.h"
#include "algebra_vector.h"

/* Needed to directly access workspace objects in the tests */
#include "types.h"

/* Needed for memory operations */
#include "glob_opts.h"

/*
 * Linear algebra smart pointers
 */
struct OSQPMatrix_deleter {
    void operator()(OSQPMatrix* mat) {
        OSQPMatrix_free(mat);
    }
};

struct OSQPVectorf_deleter {
    void operator()(OSQPVectorf* vec) {
        OSQPVectorf_free(vec);
    }
};

struct OSQPVectori_deleter {
    void operator()(OSQPVectori* vec) {
        OSQPVectori_free(vec);
    }
};

using OSQPMatrix_ptr  = std::unique_ptr<OSQPMatrix, OSQPMatrix_deleter>;
using OSQPVectorf_ptr = std::unique_ptr<OSQPVectorf, OSQPVectorf_deleter>;
using OSQPVectori_ptr = std::unique_ptr<OSQPVectori, OSQPVectori_deleter>;


/*
 * OSQP API types smart pointers
 */
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


#endif /* #ifndef OSQP_API_H_ */
