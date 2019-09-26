#ifndef ALGEBRA_TYPES_H
# define ALGEBRA_TYPES_H

# ifdef __cplusplus
extern "C" {
# endif // ifdef __cplusplus

#include "csc_math.h"


/* CUDA headers */
#include "cuda_handler.h"

extern CUDA_Handle_t *CUDA_handle;


/*********************************************
*   Internal definition of OSQPVector types
*   and supporting definitions
*********************************************/

struct OSQPVectori_ {
  c_int *d_val;
  c_int *values;
  c_int  length;
};

struct OSQPVectorf_ {
  c_float *d_val;
  c_float *values;
  c_int    length;
};


/*********************************************
*   Internal definition of OSQPMatrix type
*   and supporting definitions
*********************************************/

/**
 *  An enum used to indicate whether a matrix is symmetric.   Options
 *  NONE : matrix is fully populated
 *  TRUI : matrix is symmetric and only upper triangle is stored
 */
typedef enum OSQPMatrix_symmetry_type {NONE,TRIU} OSQPMatrix_symmetry_type;

struct OSQPMatrix_ {
  csc*                             csc;
  OSQPMatrix_symmetry_type    symmetry;
};


# ifdef __cplusplus
}
# endif // ifdef __cplusplus

#endif // ifndef ALGEBRA_TYPES_H
