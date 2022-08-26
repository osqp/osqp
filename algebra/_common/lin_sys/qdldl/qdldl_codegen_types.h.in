/*
 * Fix the types that QDLDL uses for the generated code to match
 * the OSQP types and be C89 compliant.
 */

#ifndef QDLDL_TYPES_H
# define QDLDL_TYPES_H

#include "osqp_api_types.h"

# ifdef __cplusplus
extern "C" {
# endif /* ifdef __cplusplus */

#include <limits.h> //for the QDLDL_INT_TYPE_MAX

/* Set the QDLDL integer and float types the same as OSQP */
typedef OSQPInt    QDLDL_int;   /* for indices */
typedef OSQPFloat  QDLDL_float; /* for numerical values  */

/* Always use int for bool because we must be C89 compliant */
typedef int   QDLDL_bool;

/* Maximum value of the signed type QDLDL_int. */
#ifdef OSQP_USE_LONG
#define QDLDL_INT_MAX LLONG_MAX
#else
#define QDLDL_INT_MAX INT_MAX
#endif

# ifdef __cplusplus
}
# endif /* ifdef __cplusplus */

#endif /* ifndef QDLDL_TYPES_H */
