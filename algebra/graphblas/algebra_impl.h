#ifndef ALGEBRA_IMPL_H
# define ALGEBRA_IMPL_H

#include "csc_math.h"

#include <GraphBLAS.h>
#include <stdint.h>

#ifdef DLONG
#define OSQP_GrB_INT GrB_INT64
typedef int64_t osqp_grb_int_t;

#define OSQP_GrB_INT_SEMIRING GrB_PLUS_TIMES_SEMIRING_INT64

#define OSQP_GrB_INT_PLUS GrB_PLUS_INT64
#define OSQP_GrB_INT_MINUS GrB_PLUS_INT64
#else
#define OSQP_GrB_INT GrB_INT32
typedef int32_t osqp_grb_int_t;

#define OSQP_GrB_INT_SEMIRING GrB_PLUS_TIMES_SEMIRING_INT64

#define OSQP_GrB_INT_PLUS GrB_PLUS_INT64
#define OSQP_GrB_INT_MINUS GrB_PLUS_INT64
#endif

#ifdef DFLOAT
#define OSQP_GrB_FLOAT GrB_FP32


#define GR_FLOAT_SEMIRING GrB_PLUS_TIMES_SEMIRING_FP32

#define OSQP_GrB_FLOAT_PLUS  GrB_PLUS_FP32
#define OSQP_GrB_MINUS GrB_MINUS_FP32
#define OSQP_GrB_TIMES GrB_TIMES_FP32
#define OSQP_GrB_MAX   GrB_MAX_FP64
#else
#define OSQP_GrB_FLOAT GrB_FP64

#define OSQP_GrB_FLOAT_SEMIRING GrB_PLUS_TIMES_SEMIRING_FP64

#define OSQP_GrB_FLOAT_PLUS  GrB_PLUS_FP64
#define OSQP_GrB_FLOAT_MINUS GrB_MINUS_FP64
#define OSQP_GrB_FLOAT_TIMES GrB_TIMES_FP64
#define OSQP_GrB_FLOAT_MAX   GrB_MAX_FP64
#endif

/*
 * Custom GraphBLAS items
 * (Defined inside static_items.h)
 */

/* A scalar that is always 0.0 */
extern GrB_Scalar OSQP_GrB_FLOAT_ZERO;

/* A binary operator that returns the item that has the largest absolute value */
extern GrB_BinaryOp OSQP_GrB_MAXABS;

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
