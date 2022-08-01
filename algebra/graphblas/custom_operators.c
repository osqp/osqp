#include "glob_opts.h"
#include "algebra_impl.h"
#include "GraphBLAS.h"

/* Helper function form computing the infinity norm of a vector */
GrB_BinaryOp maxabs;
void maxabs_op_func(c_float *res, const c_float *a, const c_float *b) {
  c_float absa = c_absval(*a);
  c_float absb = c_absval(*b);

  (*res) = (absa > absb) ? absa : absb;
}

void init_custom_operators() {

    /* Create a binary operator that returns the item that has the largest absolute value */
    if (!maxabs) {
    GrB_BinaryOp_new(&maxabs,
                     maxabs_op_func,
                     GRFLOAT,
                     GRFLOAT,
                     GRFLOAT);
    }

}


void destroy_custom_operators() {
    GrB_BinaryOp_free(&maxabs);
}
