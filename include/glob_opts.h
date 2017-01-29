#ifndef GLOB_OPTS_H
#define GLOB_OPTS_H

/* DATA CUSTOMIZATIONS (depending on memory manager)-----------------------   */
/* define custom printfs and memory allocation (e.g. matlab or python) */
#ifdef MATLAB_MEX_FILE
    #include "mex.h"
    static void* c_calloc(size_t num, size_t size){
        void *m = mxCalloc(num,size);
        mexMakeMemoryPersistent(m);
        return m;
    }
    static void* c_malloc(size_t size){
        void *m = mxMalloc(size);
        mexMakeMemoryPersistent(m);
        return m;
    }
    static void* c_realloc(void *ptr, size_t size){
        void *m = mxRealloc(ptr,size);
        mexMakeMemoryPersistent(m);
        return m;
    }
    #define c_free mxFree
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
#define OSQP_INFTY 1.0e20  // Numerical value of infinity


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

#ifdef PRINTING
#ifdef MATLAB_MEX_FILE
//call mexPrintf and immediately force print buffer flush
//otherwise messages don't appear until solver termination
//ugly because matlab does not provide a vprintf mex interface
#include <stdarg.h>
static int c_print(char *msg, ...)
{
  va_list argList;
  va_start(argList, msg);
  //message buffer
  int bufferSize = 256;
  char buffer[bufferSize];
  vsnprintf(buffer,bufferSize-1, msg, argList);
  va_end(argList);
  int out = mexPrintf(buffer); //print to matlab display
  mexEvalString("drawnow;");   // flush matlab print buffer
  return out;
}
/* #elif defined PYTHON
  #define c_print(...)                                                         \
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
