#include "glob_opts.h"
#include "algebra_impl.h"
#include "GraphBLAS.h"

/* A scalar value that is always 0 */
GrB_Scalar OSQP_GrB_FLOAT_ZERO;

/* A vector that is always empty */
GrB_Vector OSQP_GrB_FLOAT_EMPTY_VEC;

/* A unary operator to compute the square root */
GrB_UnaryOp OSQP_GrB_FLOAT_SQRT;
void osqp_unary_sqrt(void *res, const void *x) {
    c_float *resVal = (c_float*) res;
    c_float *xVal = (c_float*) x;

    *(resVal) = c_sqrt(*(xVal));
}

void init_custom_operators() {
    /* Create a zero scalar */
    if (!OSQP_GrB_FLOAT_ZERO) {
        GrB_Scalar_new(&OSQP_GrB_FLOAT_ZERO, OSQP_GrB_FLOAT);
        GrB_Scalar_setElement(OSQP_GrB_FLOAT_ZERO, 0.0);
    }

    /* Create an empty vector */
    if (!OSQP_GrB_FLOAT_EMPTY_VEC) {
        GrB_Vector_new(&OSQP_GrB_FLOAT_EMPTY_VEC, OSQP_GrB_FLOAT, 0);
        GrB_Vector_clear(OSQP_GrB_FLOAT_EMPTY_VEC);
    }

    /* Create the square root operator */
    if (!OSQP_GrB_FLOAT_SQRT) {
        GrB_UnaryOp_new(&OSQP_GrB_FLOAT_SQRT, osqp_unary_sqrt, OSQP_GrB_FLOAT, OSQP_GrB_FLOAT);
    }
}


void destroy_custom_operators() {
    GrB_free(&OSQP_GrB_FLOAT_ZERO);
    GrB_free(&OSQP_GrB_FLOAT_EMPTY_VEC);
    GrB_free(&OSQP_GrB_FLOAT_SQRT);
}
