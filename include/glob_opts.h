#ifndef GLOB_OPTS_H
#define GLOB_OPTS_H

/* DATA CUSTOMIZATIONS (depending on memory manager)-----------------------   */
/* define custom printfs and memory allocation (e.g. matlab or python) */
#ifdef MATLAB_MEX_FILE
    #include "mex.h"
    #define c_malloc mxMalloc
    #define c_calloc mxCalloc
    #define c_free mxFree
    #define c_realloc mxRealloc
#elif defined PYTHON
    // Define memory allocation for python. Note that in Python 2 memory manager
    // Calloc is not implemented
    #include <Python.h>
    #define c_malloc PyMem_Malloc
    #define c_calloc(n,s) ({                     \
            void * p_calloc = c_malloc((n)*(s)); \
            memset(p_calloc, 0, (n)*(s));        \
            p_calloc;                            \
        })
    #define c_free PyMem_Free
    #define c_realloc PyMem_Realloc
#else
    #define c_malloc malloc
    #define c_calloc calloc
    #define c_free free
    #define c_realloc realloc
#endif
/* Use customized constants -----------------------------------------------   */
#define OSQP_NULL 0
#define OSQP_INFTY 1.0e20;  // Numerical value of infinity


/* Use customized number representation -----------------------------------   */
typedef int c_int;                   /* for indeces */
typedef double c_float;              /* for numerical values  */

/* Use customized operations */
#define c_sqrt sqrt  // Doubles
#define c_sqrtf sqrtf  // Floats

#ifndef c_absval
#define c_absval(x) (((x) < 0) ? -(x) : (x))
#endif

#ifndef c_max
#define c_max(a, b) (((a) > (b)) ? (a) : (b))
#endif

#ifndef c_min
#define c_min(a, b) (((a) < (b)) ? (a) : (b))
#endif

/* Use customized functions -----------------------------------------------   */

#if PRINTLEVEL > 0
#ifdef MATLAB_MEX_FILE
#define c_print mexPrintf
/* #elif defined PYTHON
  #define c_print(...)                                                           \
    {                                                                          \
        PyGILState_STATE gilstate = PyGILState_Ensure();                       \
        PySys_WriteStdout(__VA_ARGS__);                                        \
        PyGILState_Release(gilstate);                                          \
    }
*/
#else
#define c_print printf
#endif
#endif


/* Use customized structures names */
typedef struct OSQP_PROBLEM_DATA Data;
typedef struct OSQP_POLISH Polish;
typedef struct OSQP_SETTINGS Settings;
typedef struct OSQP_SOLUTION Solution;
typedef struct OSQP_INFO Info;
typedef struct OSQP_SCALING Scaling;
typedef struct OSQP_WORK Work;
typedef struct OSQP_TIMER Timer;




#endif
