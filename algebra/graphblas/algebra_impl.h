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
/* Types */
#define OSQP_GrB_FLOAT GrB_FP32

/* Semirings */
#define OSQP_GrB_PLUS_TIMES_FLOAT_SEMIRING GrB_PLUS_TIMES_SEMIRING_FP32

/* Monoids */
#define OSQP_GrB_FLOAT_MAX_MONOID  GrB_MAX_MONOID_FP32
#define OSQP_GrB_FLOAT_PLUS_MONOID GrB_PLUS_MONOID_FP32

/* Unary operators */
#define OSQP_GrB_FLOAT_ABS   GrB_ABS_FP32
#define OSQP_GrB_FLOAT_MINV  GrB_MINV_FP32

/* Binary operators */
#define OSQP_GrB_FLOAT_PLUS  GrB_PLUS_FP32
#define OSQP_GrB_FLOAT_MINUS GrB_MINUS_FP32
#define OSQP_GrB_FLOAT_TIMES GrB_TIMES_FP32
#define OSQP_GrB_FLOAT_MAX   GrB_MAX_FP32
#define OSQP_GrB_FLOAT_MIN   GrB_MIN_FP32

/* Unary index functions */
#define OSQP_GrB_VALUEGT GrB_VALUEGT_FP32
#define OSQP_GrB_VALUELT GrB_VALUELT_FP32

/* Type-dependent functions */
#define OSQP_GxB_Vector_Iterator_get_Float GxB_Iterator_get_FP64
#else
/* Types */
#define OSQP_GrB_FLOAT GrB_FP64

/* Semirings */
#define OSQP_GrB_PLUS_TIMES_FLOAT_SEMIRING GrB_PLUS_TIMES_SEMIRING_FP64

/* Monoids */
#define OSQP_GrB_FLOAT_MAX_MONOID  GrB_MAX_MONOID_FP64
#define OSQP_GrB_FLOAT_PLUS_MONOID GrB_PLUS_MONOID_FP64

/* Unary operators */
#define OSQP_GrB_FLOAT_ABS   GrB_ABS_FP64
#define OSQP_GrB_FLOAT_MINV  GrB_MINV_FP64

/* Binary operators */
#define OSQP_GrB_FLOAT_PLUS  GrB_PLUS_FP64
#define OSQP_GrB_FLOAT_MINUS GrB_MINUS_FP64
#define OSQP_GrB_FLOAT_TIMES GrB_TIMES_FP64
#define OSQP_GrB_FLOAT_MAX   GrB_MAX_FP64
#define OSQP_GrB_FLOAT_MIN   GrB_MIN_FP64

/* Unary index functions */
#define OSQP_GrB_VALUEGT GrB_VALUEGT_FP64
#define OSQP_GrB_VALUELT GrB_VALUELT_FP64

/* Type-dependent functions */
#define OSQP_GxB_Vector_Iterator_get_Float GxB_Iterator_get_FP64
#endif


/*
 * Custom GraphBLAS items
 * (Defined inside static_items.h)
 */

/* A scalar that is always 0.0 */
extern GrB_Scalar OSQP_GrB_FLOAT_ZERO;

/* A vector that is always empty */
extern GrB_Vector OSQP_GrB_FLOAT_EMPTY_VEC;

/* A unary operator to compute the square root */
extern GrB_UnaryOp OSQP_GrB_FLOAT_SQRT;


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
