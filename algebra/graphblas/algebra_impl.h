#ifndef ALGEBRA_IMPL_H
# define ALGEBRA_IMPL_H

#include "csc_math.h"

#include <GraphBLAS.h>
#include <stdint.h>

#ifdef DLONG
#define GRINT GrB_INT64
typedef int64_t gb_int;

#define GR_INT_SEMIRING GrB_PLUS_TIMES_SEMIRING_INT64

#define GR_INT_PLUS GrB_PLUS_INT64
#define GR_INT_MINUS GrB_PLUS_INT64
#else
#define GRINT GrB_INT32
typedef int32_t gb_int;

#define GR_INT_SEMIRING GrB_PLUS_TIMES_SEMIRING_INT64

#define GR_INT_PLUS GrB_PLUS_INT64
#define GR_INT_MINUS GrB_PLUS_INT64
#endif

#ifdef DFLOAT
#define GRFLOAT GrB_FP32


#define GR_FLOAT_SEMIRING GrB_PLUS_TIMES_SEMIRING_FP32

#define GR_FLOAT_PLUS  GrB_PLUS_FP32
#define GR_FLOAT_MINUS GrB_MINUS_FP32
#define GR_FLOAT_TIMES GrB_TIMES_FP32
#define GR_FLOAT_MAX   GrB_MAX_FP64
#else
#define GRFLOAT GrB_FP64

#define GR_FLOAT_SEMIRING GrB_PLUS_TIMES_SEMIRING_FP64

#define GR_FLOAT_PLUS  GrB_PLUS_FP64
#define GR_FLOAT_MINUS GrB_MINUS_FP64
#define GR_FLOAT_TIMES GrB_TIMES_FP64
#define GR_FLOAT_MAX   GrB_MAX_FP64
#endif

/*
 * Custom binary operators
 */

/* A binary operator that returns the item that has the largest absolute value */
extern GrB_BinaryOp maxabs;

/*********************************************
*   Internal definition of OSQPVector types
*   and supporting definitions
*********************************************/

struct OSQPVectori_ {
  GrB_Vector vec;
  c_int      length;
};

struct OSQPVectorf_ {
  GrB_Vector vec;
  c_int      length;
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
  GrB_Matrix mat;
  OSQPMatrix_symmetry_type    symmetry;
};


#endif /* ifndef ALGEBRA_IMPL_H */
