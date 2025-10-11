#ifndef ALGEBRA_IMPL_H
#define ALGEBRA_IMPL_H

#include "csc_math.h"

#ifdef __cplusplus
extern "C" {
#endif

/*********************************************
 *   Internal definition of algebra context.
 *********************************************/
struct OSQPAlgebraContext_ {
  // Nothing is needed in the built-in implementation
};


/*********************************************
*   Internal definition of OSQPVector types
*   and supporting definitions
*********************************************/

struct OSQPVectori_ {
  const OSQPAlgebraContext* context;  /* Not owned by the vector - owned by the solver */
  OSQPInt*                  values;
  OSQPInt                   length;
};

struct OSQPVectorf_ {
  const OSQPAlgebraContext* context;  /* Not owned by the vector - owned by the solver */
  OSQPFloat*                values;
  OSQPInt                   length;
};


/*********************************************
*   Internal definition of OSQPMatrix type
*   and supporting definitions
*********************************************/

/**
 *  An enum used to indicate whether a matrix is symmetric.   Options
 *  NONE : matrix is fully populated
 *  TRIU : matrix is symmetric and only upper triangle is stored
 */
typedef enum OSQPMatrix_symmetry_type {NONE,TRIU} OSQPMatrix_symmetry_type;

struct OSQPMatrix_ {
  const OSQPAlgebraContext* context;  /* Not owned by the matrix - owned by the solver */
  OSQPCscMatrix*            csc;
  OSQPMatrix_symmetry_type  symmetry;
};

#ifdef __cplusplus
}
#endif

#endif /* ifndef ALGEBRA_IMPL_H */
