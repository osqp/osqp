#include "glob_opts.h"
#include "algebra_impl.h"
#include "GraphBLAS.h"

/* A scalar value that is always 0 */
GrB_Scalar OSQP_GrB_FLOAT_ZERO;

/* Helper function form computing the infinity norm of a vector */
GrB_BinaryOp OSQP_GrB_MAXABS;
void maxabs_op_func(c_float *res, const c_float *a, const c_float *b) {
  c_float absa = c_absval(*a);
  c_float absb = c_absval(*b);

  (*res) = (absa > absb) ? absa : absb;
}

void init_custom_operators() {
    /* Create a zero scalar */
    if (!OSQP_GrB_FLOAT_ZERO) {
        GrB_Scalar_new(&OSQP_GrB_FLOAT_ZERO, OSQP_GrB_FLOAT);
        GrB_Scalar_setElement(OSQP_GrB_FLOAT_ZERO, 0.0);
    }

    /* Create a binary operator that returns the item that has the largest absolute value */
    if (!OSQP_GrB_MAXABS) {
    GrB_BinaryOp_new(&OSQP_GrB_MAXABS,
                     maxabs_op_func,
                     OSQP_GrB_FLOAT,
                     OSQP_GrB_FLOAT,
                     OSQP_GrB_FLOAT);
    }

}


void destroy_custom_operators() {
    GrB_free(&OSQP_GrB_FLOAT_ZERO);
    GrB_BinaryOp_free(&OSQP_GrB_MAXABS);
}
