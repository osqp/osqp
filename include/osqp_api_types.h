#ifndef  OSQP_API_TYPES_H
#define  OSQP_API_TYPES_H

# ifdef __cplusplus
extern "C" {
# endif // ifdef __cplusplus

/*****************************
* OSQP API type definitions  *
******************************/

/* OSQP custom float definitions */
# ifdef DLONG            // long integers
typedef long long c_int; /* for indices */
# else // standard integers
typedef int c_int;       /* for indices */
# endif /* ifdef DLONG */


# ifndef DFLOAT         // Doubles
typedef double c_float; /* for numerical values  */
# else                  // Floats
typedef float c_float;  /* for numerical values  */
# endif /* ifndef DFLOAT */


# ifdef __cplusplus
}
# endif // ifdef __cplusplus

#endif // ifndef OSQP_API_TYPES_H