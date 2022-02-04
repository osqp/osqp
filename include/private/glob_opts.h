#ifndef GLOB_OPTS_H
#define GLOB_OPTS_H

/*
   Define OSQP compiler flags
 */

// cmake generated compiler flags
#include "osqp_configure.h"

/* DATA CUSTOMIZATIONS (depending on memory manager)-----------------------   */

// We do not need memory allocation functions if EMBEDDED is enabled
# ifndef EMBEDDED
# include "memory_defs.h"
# endif // end ifndef EMBEDDED


/* Use customized operations */

# ifndef c_absval
#  define c_absval(x) (((x) < 0) ? -(x) : (x))
# endif /* ifndef c_absval */

# ifndef c_max
#  define c_max(a, b) (((a) > (b)) ? (a) : (b))
# endif /* ifndef c_max */

# ifndef c_min
#  define c_min(a, b) (((a) < (b)) ? (a) : (b))
# endif /* ifndef c_min */

// Round x to the nearest multiple of N
# ifndef c_roundmultiple
#  define c_roundmultiple(x, N) ((x) + .5 * (N)-c_fmod((x) + .5 * (N), (N)))
# endif /* ifndef c_roundmultiple */


/* Use customized functions -----------------------------------------------   */

# if EMBEDDED != 1

#  include <math.h>
#  ifndef DFLOAT // Doubles
#   define c_sqrt sqrt
#   define c_fmod fmod
#  else          // Floats
#   define c_sqrt sqrtf
#   define c_fmod fmodf
#  endif /* ifndef DFLOAT */

# endif // end EMBEDDED

# ifdef PRINTING
#  include <stdio.h>
#  include <string.h>

/* informational print function */
#  ifdef MATLAB
#   define c_print mexPrintf
#  elif defined PYTHON
#   include <Python.h>
# define c_print(...)                              \
  {                                                  \
    PyGILState_STATE gilstate = PyGILState_Ensure(); \
    PySys_WriteStdout(__VA_ARGS__);                  \
    PyGILState_Release(gilstate);                    \
  }
#  elif defined R_LANG
#   include <R_ext/Print.h>
#   define c_print Rprintf
#  else  /* ifdef MATLAB */
#   define c_print printf
#  endif /* c_print configuration */

/* error printing function */
#  ifdef R_LANG
    /* Some CRAN builds complain about __VA_ARGS__, so just print */
    /* out the error messages on R without the __FUNCTION__ trace */
#   define c_eprint Rprintf
#  else
#   define c_eprint(...) c_print("ERROR in %s: ", __FUNCTION__); \
            c_print(__VA_ARGS__); c_print("\n");
#  endif /* c_eprint configuration */

# endif  /* PRINTING */


#endif /* ifndef GLOB_OPTS_H */
